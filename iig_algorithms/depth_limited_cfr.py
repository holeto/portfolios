"""Layered CFR+ over trees built by `tree_builder`.

The tree is processed layer by layer (BFS order), each layer being a set of
histories with joint actions [H, A1, A2]. Infoset level quantities are
aggregated with segment sums. Supports chance histories (player 2 "acts"
for chance with fixed probabilities) and depth-limit leaves (the leaf
utility matrix is over the portfolio policies of both players).

Variant: regret matching+, linear averaging, alternating updates.
Utilities are always from the perspective of player 1.
"""
from __future__ import annotations

import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np

from iig_algorithms.tree_builder import LEAF, TreeLayer, bucket_size


@chex.dataclass(frozen=True)
class DepthLimitedCFRConstants:
  """Padded tree arrays. Symbols: D depth, Pl players, H(D) histories at
  depth D, I(D) infosets at depth D, W(D) action width at depth D."""
  init_reaches: chex.Array = ()                  # [Pl + 1, H(0)], last is chance
  depth_iset_legal: chex.ArrayTree = ()          # D x Pl x [I(D), W(D)]
  depth_history_iset: chex.ArrayTree = ()        # D x [Pl, H(D)]
  depth_history_player_legal: chex.ArrayTree = ()  # D x [Pl, H(D), W(D)]
  depth_history_chance_probs: chex.ArrayTree = ()  # D x [H(D), W(D)]
  depth_history_is_chance: chex.ArrayTree = ()   # D x [H(D)]
  depth_history_action_utility: chex.ArrayTree = ()  # D x [H(D), W(D), W(D)]
  depth_history_next_history: chex.ArrayTree = ()  # D x [H(D), W(D), W(D)]
  depth_history_parent: chex.ArrayTree = ()      # D x [H(D)]
  depth_history_parent_action: chex.ArrayTree = ()  # D x [Pl, H(D)]
  depth_history_is_leaf: chex.ArrayTree = ()     # D x [H(D)] multi valued depth-limit leaves
  depth_history_leaf_values: chex.ArrayTree = ()  # D x [H(D), Pl, F(D)] P1 values of the opponent's options


def regret_matching(regrets: chex.Array, legal: chex.Array) -> chex.Array:
  positive = jnp.maximum(regrets, 0) * legal
  total = jnp.sum(positive, axis=-1, keepdims=True)
  uniform = legal / jnp.sum(legal, axis=-1, keepdims=True)
  return jnp.where(total > 0, positive / jnp.where(total > 0, total, 1), uniform)


def constants_from_layers(layers: list[TreeLayer], init_reaches: np.ndarray) -> DepthLimitedCFRConstants:
  """Converts tree layers into padded CFR constants. Histories and infosets
  are padded to powers of two, the padding histories have a single legal
  joint action, no utility and point into a padding infoset."""
  fields = {k: [] for k in DepthLimitedCFRConstants.__dataclass_fields__ if k != "init_reaches"}
  for layer in layers:
    num_h, width = layer.num_histories, layer.width
    padded_h = bucket_size(num_h)
    pad = padded_h - num_h

    iset_legal, history_iset = [], []
    for pl in range(2):
      num_i = layer.iset_legal[pl].shape[0]
      padded_i = bucket_size(num_i + 1)
      legal = np.zeros((padded_i, width), dtype=np.float32)
      legal[:num_i] = layer.iset_legal[pl]
      legal[num_i:, 0] = 1
      iset_legal.append(jnp.asarray(legal))
      history_iset.append(np.concatenate([layer.history_iset[pl], np.full(pad, num_i)]))
    fields["depth_iset_legal"].append(iset_legal)
    fields["depth_history_iset"].append(jnp.asarray(np.stack(history_iset), dtype=jnp.int32))

    player_legal = np.zeros((2, padded_h, width), dtype=np.float32)
    player_legal[:, :num_h] = layer.player_legal
    player_legal[:, num_h:, 0] = 1
    fields["depth_history_player_legal"].append(jnp.asarray(player_legal))

    def pad_rows(x, value=0, dtype=None):
      out = np.full((padded_h,) + x.shape[1:], value, dtype=dtype or x.dtype)
      out[:num_h] = x
      return jnp.asarray(out)
    fields["depth_history_chance_probs"].append(pad_rows(layer.chance_probs, dtype=np.float32))
    fields["depth_history_is_chance"].append(pad_rows(layer.kind == 1, False))
    fields["depth_history_action_utility"].append(pad_rows(layer.utility, dtype=np.float32))
    fields["depth_history_next_history"].append(pad_rows(layer.next_history, -1, np.int32))
    fields["depth_history_parent"].append(pad_rows(np.maximum(layer.parent, 0), 0, np.int32))
    parent_action = np.zeros((2, padded_h), dtype=np.int32)
    parent_action[:, :num_h] = np.maximum(layer.parent_action, 0)
    fields["depth_history_parent_action"].append(jnp.asarray(parent_action))
    if layer.leaf_values is not None:
      fields["depth_history_is_leaf"].append(pad_rows(layer.kind == LEAF, False))
      fields["depth_history_leaf_values"].append(pad_rows(layer.leaf_values, dtype=np.float32))
    else:
      fields["depth_history_is_leaf"].append(jnp.zeros(padded_h, dtype=bool))
      fields["depth_history_leaf_values"].append(jnp.zeros((padded_h, 2, 1), dtype=jnp.float32))

  padded_reaches = np.zeros((3, bucket_size(layers[0].num_histories)), dtype=np.float32)
  padded_reaches[:, :layers[0].num_histories] = init_reaches
  return DepthLimitedCFRConstants(init_reaches=jnp.asarray(padded_reaches), **fields)


def _history_strategies(c: DepthLimitedCFRConstants, iset_strategies) -> list[chex.Array]:
  """Per history strategies [Pl, H, W]. In chance histories player 1 plays
  its dummy action and player 2 the chance probabilities."""
  result = []
  for d in range(len(c.depth_history_iset)):
    player_legal = c.depth_history_player_legal[d]
    strategies = []
    for pl in range(2):
      legal = player_legal[pl]
      strategy = iset_strategies[d][pl][c.depth_history_iset[d][pl]] * legal
      total = jnp.sum(strategy, axis=-1, keepdims=True)
      # Histories of an infoset may in general have different legal actions.
      strategy = jnp.where(total > 1e-12, strategy / jnp.where(total > 1e-12, total, 1),
                           legal / jnp.sum(legal, axis=-1, keepdims=True))
      strategies.append(strategy)
    is_chance = c.depth_history_is_chance[d][:, None]
    p1_strategy = jnp.where(is_chance, player_legal[0], strategies[0])
    p2_strategy = jnp.where(is_chance, c.depth_history_chance_probs[d], strategies[1])
    result.append(jnp.stack([p1_strategy, p2_strategy], axis=0))
  return result


def _history_reaches(c: DepthLimitedCFRConstants, history_strategies) -> list[chex.Array]:
  """Reach probabilities [Pl + 1, H] of both players and chance."""
  reaches = [c.init_reaches]
  for d in range(1, len(c.depth_history_iset)):
    parent = c.depth_history_parent[d]
    a1, a2 = c.depth_history_parent_action[d][0], c.depth_history_parent_action[d][1]
    prev = reaches[-1]
    strategy = history_strategies[d - 1]
    parent_chance = c.depth_history_is_chance[d - 1][parent]
    p1_prob = strategy[0][parent, a1]
    p2_prob = strategy[1][parent, a2]
    reaches.append(jnp.stack([
        prev[0][parent] * p1_prob,
        prev[1][parent] * jnp.where(parent_chance, 1.0, p2_prob),
        prev[2][parent] * jnp.where(parent_chance, p2_prob, 1.0)], axis=0))
  return reaches


def _multi_valued_leaf_values(c: DepthLimitedCFRConstants, d: int, reaches, player: int) -> chex.Array:
  """Values [H] of multi valued leaves from the perspective of the given player. In each
  infoset s of the player, the opponent picks its option minimizing the counterfactual
  value of s (for player 2 maximizing the P1 value)."""
  option_values = c.depth_history_leaf_values[d][:, player, :]
  cf_reach = reaches[d][1 - player] * reaches[d][2]
  iset_ids = c.depth_history_iset[d][player]
  num_isets = c.depth_iset_legal[d][player].shape[0]
  iset_option_values = jax.ops.segment_sum(option_values * cf_reach[:, None], iset_ids, num_segments=num_isets)
  if player == 0:
    choice = jnp.argmin(iset_option_values, axis=-1)
  else:
    choice = jnp.argmax(iset_option_values, axis=-1)
  return jnp.take_along_axis(option_values, choice[iset_ids][:, None], axis=-1)[:, 0]


def _history_values(c: DepthLimitedCFRConstants, history_strategies, reaches, player: int):
  """Backward pass from the perspective of the given player (static int), which
  only matters for multi valued leaves. Returns per layer (q1 [H, W], q2 [H, W], v [H])."""
  depth = len(c.depth_history_iset)
  result = [None] * depth
  child_values = jnp.zeros((1,))
  for d in range(depth - 1, -1, -1):
    next_history = c.depth_history_next_history[d]
    action_value = c.depth_history_action_utility[d] + jnp.where(
        next_history >= 0, child_values[jnp.maximum(next_history, 0)], 0.0)
    p1_strategy, p2_strategy = history_strategies[d][0], history_strategies[d][1]
    q1 = jnp.einsum("hab,hb->ha", action_value, p2_strategy)
    q2 = jnp.einsum("hab,ha->hb", action_value, p1_strategy)
    value = jnp.sum(q1 * p1_strategy, axis=-1)
    is_leaf = c.depth_history_is_leaf[d]
    leaf_value = _multi_valued_leaf_values(c, d, reaches, player)
    value = jnp.where(is_leaf, leaf_value, value)
    q1 = jnp.where(is_leaf[:, None], leaf_value[:, None], q1)
    q2 = jnp.where(is_leaf[:, None], leaf_value[:, None], q2)
    result[d] = (q1, q2, value)
    child_values = value
  return result


def _player_update(c: DepthLimitedCFRConstants, regrets, averages, cf_values, player: int, iteration):
  """A single CFR+ update of the given player (static int)."""
  depth = len(c.depth_history_iset)
  current = [[regret_matching(regrets[d][pl], c.depth_iset_legal[d][pl]) for pl in range(2)] for d in range(depth)]
  history_strategies = _history_strategies(c, current)
  reaches = _history_reaches(c, history_strategies)
  values = _history_values(c, history_strategies, reaches, player)

  regrets = [list(r) for r in regrets]
  averages = [list(a) for a in averages]
  cf_values = [list(v) for v in cf_values]
  opponent = 1 - player
  sign = 1.0 if player == 0 else -1.0
  for d in range(depth):
    q1, q2, value = values[d]
    q = q1 if player == 0 else q2
    non_chance = 1.0 - c.depth_history_is_chance[d].astype(jnp.float32)
    num_isets = regrets[d][player].shape[0]
    iset_ids = c.depth_history_iset[d][player]

    cf_reach = reaches[d][opponent] * reaches[d][2] * non_chance
    instant_regret = sign * (q - value[:, None]) * cf_reach[:, None] * c.depth_history_player_legal[d][player]
    iset_regret = jax.ops.segment_sum(instant_regret, iset_ids, num_segments=num_isets)
    regrets[d][player] = jnp.maximum(regrets[d][player] + iset_regret * c.depth_iset_legal[d][player], 0.0)

    own_reach = reaches[d][player] * non_chance
    realization = history_strategies[d][player] * own_reach[:, None]
    averages[d][player] = averages[d][player] + iteration * jax.ops.segment_sum(realization, iset_ids, num_segments=num_isets)

    value_sum = jax.ops.segment_sum(value * cf_reach, iset_ids, num_segments=num_isets)
    reach_sum = jax.ops.segment_sum(cf_reach, iset_ids, num_segments=num_isets)
    iset_value = jnp.where(reach_sum > 1e-12, value_sum / jnp.where(reach_sum > 1e-12, reach_sum, 1), 0.0)
    cf_values[d][player] = cf_values[d][player] + (iset_value - cf_values[d][player]) * (2.0 / (iteration + 1.0))
  return regrets, averages, cf_values


@jax.jit
def _run_iterations(c: DepthLimitedCFRConstants, regrets, averages, cf_values, start_iteration, num_iterations):
  def body(i, carry):
    regrets, averages, cf_values = carry
    iteration = (start_iteration + i).astype(jnp.float32)
    for player in range(2):
      regrets, averages, cf_values = _player_update(c, regrets, averages, cf_values, player, iteration)
    return regrets, averages, cf_values
  return jax.lax.fori_loop(0, num_iterations, body, (regrets, averages, cf_values))


def _normalize(strategy, legal):
  total = jnp.sum(strategy, axis=-1, keepdims=True)
  return jnp.where(total > 1e-12, strategy / jnp.where(total > 1e-12, total, 1),
                   legal / jnp.sum(legal, axis=-1, keepdims=True))


@functools.partial(jax.jit, static_argnames=("player",))
def _analyze(c: DepthLimitedCFRConstants, iset_strategies, player: int = 0):
  """History strategies, reaches and values (from the perspective of the given
  player for multi valued leaves) of a given infoset strategy."""
  history_strategies = _history_strategies(c, iset_strategies)
  reaches = _history_reaches(c, history_strategies)
  values = _history_values(c, history_strategies, reaches, player)
  return history_strategies, reaches, [v[2] for v in values]


class DepthLimitedCFR:
  """CFR+ solver over a (possibly depth-limited and gadget-prefixed) tree."""

  def __init__(self, layers: list[TreeLayer], init_reaches: np.ndarray):
    self.layers = layers
    self.max_depth = len(layers)
    self.constants = constants_from_layers(layers, np.asarray(init_reaches))
    c = self.constants
    self.regrets = [[jnp.zeros_like(c.depth_iset_legal[d][pl]) for pl in range(2)] for d in range(self.max_depth)]
    self.averages = [[jnp.zeros_like(c.depth_iset_legal[d][pl]) for pl in range(2)] for d in range(self.max_depth)]
    self.cf_values = [[jnp.zeros(c.depth_iset_legal[d][pl].shape[0]) for pl in range(2)] for d in range(self.max_depth)]
    self.timestep = 1

  def multiple_steps(self, iterations: int):
    self.regrets, self.averages, self.cf_values = _run_iterations(
        self.constants, self.regrets, self.averages, self.cf_values,
        jnp.asarray(self.timestep, dtype=jnp.int32), jnp.asarray(iterations, dtype=jnp.int32))
    self.timestep += iterations

  def step(self):
    self.multiple_steps(1)

  def _real_isets(self, d, pl):
    return self.layers[d].iset_legal[pl].shape[0]

  def _real_histories(self, d):
    return self.layers[d].num_histories

  def _padded_average_strategies(self):
    c = self.constants
    return [[_normalize(self.averages[d][pl], c.depth_iset_legal[d][pl]) for pl in range(2)] for d in range(self.max_depth)]

  def _padded_current_strategies(self):
    c = self.constants
    return [[regret_matching(self.regrets[d][pl], c.depth_iset_legal[d][pl]) for pl in range(2)] for d in range(self.max_depth)]

  def average_strategies(self) -> list[list[np.ndarray]]:
    """Normalized average strategy, D x Pl x [I, W]."""
    strategies = self._padded_average_strategies()
    return [[np.asarray(strategies[d][pl])[:self._real_isets(d, pl)] for pl in range(2)] for d in range(self.max_depth)]

  def current_strategies(self) -> list[list[np.ndarray]]:
    strategies = self._padded_current_strategies()
    return [[np.asarray(strategies[d][pl])[:self._real_isets(d, pl)] for pl in range(2)] for d in range(self.max_depth)]

  def analyze(self, average: bool = True, player: int = 0):
    """Returns per layer history strategies [Pl, H, W], reaches [Pl + 1, H]
    and history values [H] (P1 utility) of the average (or current) strategy.
    With multi valued leaves, the values are from the perspective of the given player."""
    strategies = self._padded_average_strategies() if average else self._padded_current_strategies()
    history_strategies, reaches, values = _analyze(self.constants, strategies, player=player)
    sizes = [self._real_histories(d) for d in range(self.max_depth)]
    return ([np.asarray(s)[:, :n] for s, n in zip(history_strategies, sizes)],
            [np.asarray(r)[:, :n] for r, n in zip(reaches, sizes)],
            [np.asarray(v)[:n] for v, n in zip(values, sizes)])

  def history_reaches(self, average: bool = True) -> list[np.ndarray]:
    return self.analyze(average)[1]

  def root_value(self, average: bool = True, player: int = 0) -> float:
    """Expected P1 utility of the root layer under the initial reaches (from the
    perspective of the given player with multi valued leaves)."""
    _, reaches, values = self.analyze(average, player)
    weights = np.prod(reaches[0], axis=0)
    return float(np.sum(weights * values[0]) / np.sum(weights))

  def iset_cf_values(self) -> list[list[np.ndarray]]:
    """Counterfactual values of infosets, normalized by the counterfactual reach, P1 utility."""
    return [[np.asarray(self.cf_values[d][pl])[:self._real_isets(d, pl)] for pl in range(2)] for d in range(self.max_depth)]

  def get_strategy(self, depth: int, player: int, infoset: np.ndarray, kind=None) -> np.ndarray:
    """Average strategy [W] of the infoset with the given tensor at the given depth."""
    iset_id = self.layers[depth].find_iset(player, infoset, kind)
    if iset_id < 0:
      raise KeyError(f"Infoset not found at depth {depth} for player {player}.")
    return self.average_strategies()[depth][player][iset_id]
