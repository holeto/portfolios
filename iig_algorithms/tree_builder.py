"""Layered (BFS) construction of game trees over the real JAX game states.

Each layer of the tree holds all histories at a given depth together with
everything that CFR, best response and the depth-limited resolving need:
joint action legality, chance probabilities, immediate utilities, links to the
next layer and infoset ids. Layers may mix decision, chance and depth-limit
leaf histories, so each layer is padded to the widest action space it contains.

Conventions (same as the games):
  - All utilities are from the perspective of player 1.
  - In chance histories player 1 has a single dummy action 0 and player 2
    "plays" the chance outcomes, outcome k being the action at index k.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

from games.jax_game import JaxGame, GameState

DECISION = 0
CHANCE = 1
LEAF = 2

MULTI_VALUED = "multi_valued"
MATRIX_VALUED = "matrix_valued"
VALUE_TYPES = (MULTI_VALUED, MATRIX_VALUED)

# (game states [H, ...], state tensors [H, S], game legal [H, Pl, A]) -> P1 values,
# [H, Pl, K] for multi valued and [H, K, K] for matrix valued states, optionally
# together with the legal options of each player [H, Pl, K] as a tuple (values, legal).
LeafValueFn = Callable[[GameState, np.ndarray, np.ndarray], np.ndarray]
# (game states [H, ...], decisions [H], chances [H]) -> leaf mask [H]
LeafPredicate = Callable[[GameState, np.ndarray, np.ndarray], np.ndarray]


@dataclasses.dataclass(frozen=True)
class LeafValues:
  """Values of the depth-limit leaves.

  multi_valued: fn returns [H, Pl, K], where [h, i, f] is the P1 value of
    history h when player i plays its blueprint and the opponent its option f
    (option 0 being the opponent's blueprint). In each infoset s_i the opponent
    picks the option minimizing the counterfactual value of s_i:
      v[s_i] = min_f sum_{h in s_i} P_{-i}(h) u_i(h)[f] / sum_{h in s_i} P_{-i}(h).
    The leaves have no actions.
  matrix_valued: fn returns [H, K, K], the P1 values of all pairs of options
    (P1 option in rows). Both players choose their option at the leaf by CFR,
    as in an additional layer.
  """
  fn: LeafValueFn
  value_type: str
  # The amount of options K, None if it may differ between calls (inferred from the values).
  num_options: Optional[int] = None

  def __post_init__(self):
    if self.value_type not in VALUE_TYPES:
      raise ValueError(f"Unknown value type {self.value_type}, expected one of {VALUE_TYPES}.")


def bucket_size(n: int, minimum: int = 8) -> int:
  """Next power of two, used to pad batch sizes and limit recompilations."""
  return max(minimum, 1 << max(n - 1, 0).bit_length())


def tree_take(tree: chex.ArrayTree, indices: np.ndarray) -> chex.ArrayTree:
  indices = jnp.asarray(indices, dtype=jnp.int32)
  return jax.tree.map(lambda x: x[indices], tree)


def tree_concat(trees: list[chex.ArrayTree]) -> chex.ArrayTree:
  return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)


class GameFunctions:
  """Batched (vmapped and jitted) game functions. The batch is padded to
  a power of two, so that the compiled functions get reused."""

  def __init__(self, game: JaxGame):
    self.game = game
    self._info = jax.jit(jax.vmap(game.get_info))
    self._apply = jax.jit(jax.vmap(game.apply_action))

    def chance_info(state):
      outcomes, probs = game.get_outcomes_and_probs(state)
      return jnp.asarray(game.is_chance(state), dtype=bool), outcomes, probs
    self._chance = jax.jit(jax.vmap(chance_info))

  @staticmethod
  def _batched_call(fn, *args):
    n = jax.tree.leaves(args[0])[0].shape[0]
    padded = bucket_size(n)
    pad_idx = np.concatenate([np.arange(n), np.zeros(padded - n, dtype=int)])
    args = [tree_take(a, pad_idx) for a in args]
    out = fn(*args)
    return jax.tree.map(lambda x: x[:n], out)

  def info(self, states):
    """Returns numpy state tensors, p1 infosets, p2 infosets and public states."""
    out = self._batched_call(self._info, states)
    return tuple(np.asarray(x, dtype=np.float32) for x in out)

  def apply(self, states, actions):
    new_states, terminal, rewards, legals = self._batched_call(self._apply, states, actions)
    return new_states, np.asarray(terminal, dtype=bool), np.asarray(rewards, dtype=np.float64), np.asarray(legals)

  def chance(self, states):
    is_chance, outcomes, probs = self._batched_call(self._chance, states)
    return np.asarray(is_chance, dtype=bool), np.asarray(outcomes), np.asarray(probs, dtype=np.float64)


_GAME_FUNCTIONS: dict[int, GameFunctions] = {}


def game_functions(game: JaxGame) -> GameFunctions:
  key = id(game)
  if key not in _GAME_FUNCTIONS or _GAME_FUNCTIONS[key].game is not game:
    _GAME_FUNCTIONS[key] = GameFunctions(game)
  return _GAME_FUNCTIONS[key]


@dataclasses.dataclass
class TreeLayer:
  """All histories at one depth of the tree.

  Symbols: H histories, W layer action width, Pl players, I(p) infosets
  of player p, A game actions.
  """
  states: GameState                 # pytree with leading dimension H
  game_legal: np.ndarray            # [H, Pl, A] legal actions as returned by the game
  kind: np.ndarray                  # [H] DECISION / CHANCE / LEAF
  decisions: np.ndarray             # [H] decision layers passed since the root
  parent: np.ndarray                # [H] index of the parent in the previous layer, -1 for roots
  parent_action: np.ndarray         # [Pl, H] joint action leading here from the parent, -1 for roots
  player_legal: np.ndarray          # [Pl, H, W]
  chance_probs: np.ndarray          # [H, W] zero for non-chance histories
  utility: np.ndarray               # [H, W, W] immediate P1 reward, or leaf value matrix
  next_history: np.ndarray          # [H, W, W] index into the next layer, -1 if none
  state_tensors: np.ndarray         # [H, S]
  infosets: np.ndarray              # [Pl, H, I] raw infoset tensors (zeros in chance histories)
  public_states: np.ndarray         # [H, P]
  history_iset: np.ndarray          # [Pl, H] infoset id of each history
  iset_legal: list[np.ndarray]      # Pl x [I(p), W]
  iset_kind: list[np.ndarray]       # Pl x [I(p)]
  iset_tensors: list[np.ndarray]    # Pl x [I(p), I] infoset tensors of each infoset id
  chances: Optional[np.ndarray] = None      # [H] chance layers passed since the root
  leaf_values: Optional[np.ndarray] = None  # [H, Pl, K] multi valued leaf values (zero elsewhere)

  @property
  def num_histories(self) -> int:
    return self.kind.shape[0]

  @property
  def width(self) -> int:
    return self.player_legal.shape[-1]

  @property
  def joint_legal(self) -> np.ndarray:
    """[H, W, W]"""
    return self.player_legal[0][:, :, None] * self.player_legal[1][:, None, :]

  def find_iset(self, player: int, infoset: np.ndarray, kind: Optional[int] = None) -> int:
    """Exact lookup of an infoset id from its tensor, -1 if not present."""
    infoset = np.asarray(infoset, dtype=np.float32)
    matches = np.all(self.iset_tensors[player] == infoset[None], axis=-1)
    if kind is not None:
      matches = matches & (self.iset_kind[player] == kind)
    found = np.flatnonzero(matches)
    return int(found[0]) if found.size > 0 else -1


def _assign_isets(kind: np.ndarray, infosets: np.ndarray, player_legal: np.ndarray):
  """Exact infoset ids. Infosets of different kinds (decision, leaf, chance dummy)
  never share an id, even if their tensors are equal."""
  history_iset, iset_legal, iset_kind, iset_tensors = [], [], [], []
  for pl in range(infosets.shape[0]):
    keys = np.concatenate([kind[:, None].astype(np.float32), infosets[pl]], axis=-1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    legal = np.zeros((unique_keys.shape[0], player_legal.shape[-1]), dtype=np.float32)
    np.maximum.at(legal, inverse, player_legal[pl])
    history_iset.append(inverse)
    iset_legal.append(legal)
    iset_kind.append(unique_keys[:, 0].astype(np.int8))
    iset_tensors.append(unique_keys[:, 1:])
  return np.stack(history_iset, axis=0), iset_legal, iset_kind, iset_tensors


def build_tree(game: JaxGame,
               root_states: GameState,
               root_legal: np.ndarray,
               depth_limit: Optional[int] = None,
               leaf_values: Optional[LeafValues] = None,
               is_leaf: Optional[LeafPredicate] = None) -> list[TreeLayer]:
  """Builds the tree below the given root histories layer by layer.

  Args:
    root_states: game states pytree with leading dimension H0.
    root_legal: [H0, Pl, A] game legal actions of the root states.
    depth_limit: amount of decision layers after which the histories become
      depth-limit leaves. Chance layers are not counted. None means no limit.
    leaf_values: values of the depth-limit leaves.
    is_leaf: additional predicate marking non-chance histories as leaves,
      e.g. the first decisions after a given amount of chance events.
  """
  if (depth_limit is not None or is_leaf is not None) and leaf_values is None:
    raise ValueError("Depth limited tree requires leaf values.")
  multi_valued = leaf_values is not None and leaf_values.value_type == MULTI_VALUED
  fns = game_functions(game)
  num_actions = game.num_distinct_actions()

  states = root_states
  game_legal = np.asarray(root_legal)
  num_roots = game_legal.shape[0]
  decisions = np.zeros(num_roots, dtype=np.int32)
  chances = np.zeros(num_roots, dtype=np.int32)
  parent = np.full(num_roots, -1, dtype=np.int32)
  parent_action = np.full((2, num_roots), -1, dtype=np.int32)

  layers = []
  while True:
    num_histories = game_legal.shape[0]
    is_chance, outcomes, probs = fns.chance(states)
    state_tensors, p1_infosets, p2_infosets, public_states = fns.info(states)

    kind = np.where(is_chance, CHANCE, DECISION).astype(np.int8)
    if depth_limit is not None:
      kind[(~is_chance) & (decisions >= depth_limit)] = LEAF
    if is_leaf is not None:
      kind[(~is_chance) & np.asarray(is_leaf(states, decisions, chances), dtype=bool)] = LEAF
    chance_mask, decision_mask, leaf_mask = kind == CHANCE, kind == DECISION, kind == LEAF

    # Leaf values (and optionally the legal options [H, Pl, K] of each player).
    leaf_idx = np.flatnonzero(leaf_mask)
    values, option_legal, num_options = None, None, 0
    if leaf_idx.size > 0:
      result = leaf_values.fn(tree_take(states, leaf_idx), state_tensors[leaf_idx], game_legal[leaf_idx])
      if isinstance(result, tuple):
        result, option_legal = result
      values = np.asarray(result, dtype=np.float64)
      num_options = values.shape[-1]
      if leaf_values.num_options is not None and num_options != leaf_values.num_options:
        raise ValueError(f"Leaf values have {num_options} options, expected {leaf_values.num_options}.")
      expected_shape = (leaf_idx.size, 2, num_options) if multi_valued else (leaf_idx.size, num_options, num_options)
      chex.assert_shape(values, expected_shape)
      if option_legal is None:
        option_legal = np.ones((leaf_idx.size, 2, num_options), dtype=np.float32)
      chex.assert_shape(option_legal, (leaf_idx.size, 2, num_options))
      if multi_valued and not np.all(option_legal > 0):
        raise ValueError("Option masks are only supported for matrix valued states.")

    width = 1
    if np.any(decision_mask):
      width = max(width, num_actions)
    if np.any(leaf_mask) and not multi_valued:
      width = max(width, num_options)
    if np.any(chance_mask):
      nonzero_outcomes = np.nonzero(probs[chance_mask] > 0)[1]
      width = max(width, int(nonzero_outcomes.max()) + 1)

    player_legal = np.zeros((2, num_histories, width), dtype=np.float32)
    if np.any(decision_mask):
      player_legal[:, decision_mask, :num_actions] = np.transpose(game_legal[decision_mask], (1, 0, 2)) > 0
    # Multi valued leaves have a single dummy action, matrix valued ones the options.
    if leaf_idx.size > 0:
      if multi_valued:
        player_legal[:, leaf_idx, 0] = 1
      else:
        player_legal[:, leaf_idx, :num_options] = np.transpose(option_legal, (1, 0, 2)) > 0
    chance_probs = np.zeros((num_histories, width), dtype=np.float64)
    chance_width = min(width, probs.shape[-1])
    chance_probs[chance_mask, :chance_width] = probs[chance_mask, :chance_width]
    player_legal[0, chance_mask, 0] = 1
    player_legal[1, chance_mask] = chance_probs[chance_mask] > 0

    utility = np.zeros((num_histories, width, width), dtype=np.float64)
    next_history = np.full((num_histories, width, width), -1, dtype=np.int32)

    layer_leaf_values = None
    if leaf_idx.size > 0:
      if multi_valued:
        layer_leaf_values = np.zeros((num_histories, 2, num_options), dtype=np.float64)
        layer_leaf_values[leaf_idx] = values
      else:
        utility[leaf_idx, :num_options, :num_options] = values

    infosets = np.stack([p1_infosets, p2_infosets], axis=0)
    infosets[:, chance_mask] = 0
    history_iset, iset_legal, iset_kind, iset_tensors = _assign_isets(kind, infosets, player_legal)

    # Expand all legal joint actions of non-leaf histories.
    joint_legal = player_legal[0][:, :, None] * player_legal[1][:, None, :]
    joint_legal[leaf_mask] = 0
    h_idx, a1_idx, a2_idx = np.nonzero(joint_legal)

    new_states = None
    if h_idx.size > 0:
      expand_chance = chance_mask[h_idx]
      chance_actions = outcomes[h_idx, np.minimum(a2_idx, outcomes.shape[1] - 1)].astype(np.int32)
      decision_actions = np.stack([a1_idx, a2_idx], axis=-1).astype(np.int32)
      actions = np.where(expand_chance[:, None], chance_actions, decision_actions)
      child_states, terminal, rewards, child_legal = fns.apply(tree_take(states, h_idx), jnp.asarray(actions))
      utility[h_idx, a1_idx, a2_idx] = np.where(expand_chance, 0.0, rewards)
      non_terminal = ~terminal
      next_history[h_idx[non_terminal], a1_idx[non_terminal], a2_idx[non_terminal]] = np.arange(np.sum(non_terminal))
      if np.any(non_terminal):
        keep = np.flatnonzero(non_terminal)
        new_states = tree_take(child_states, keep)
        new_game_legal = child_legal[keep]
        new_decisions = decisions[h_idx[keep]] + (kind[h_idx[keep]] == DECISION)
        new_chances = chances[h_idx[keep]] + (kind[h_idx[keep]] == CHANCE)
        new_parent = h_idx[keep].astype(np.int32)
        new_parent_action = np.stack([a1_idx[keep], a2_idx[keep]], axis=0).astype(np.int32)

    layers.append(TreeLayer(
        states=states, game_legal=game_legal, kind=kind, decisions=decisions,
        parent=parent, parent_action=parent_action, player_legal=player_legal,
        chance_probs=chance_probs, utility=utility, next_history=next_history,
        state_tensors=state_tensors, infosets=infosets, public_states=public_states,
        history_iset=history_iset, iset_legal=iset_legal, iset_kind=iset_kind,
        iset_tensors=iset_tensors, chances=chances, leaf_values=layer_leaf_values))

    if new_states is None:
      return layers
    states, game_legal, decisions, chances = new_states, new_game_legal, new_decisions, new_chances
    parent, parent_action = new_parent, new_parent_action


def build_full_tree(game: JaxGame) -> list[TreeLayer]:
  """The whole game tree from the initial state."""
  init_state, init_legal = game.initialize_structures()
  root = jax.tree.map(lambda x: jnp.asarray(x)[None], init_state)
  return build_tree(game, root, np.asarray(init_legal)[None])


def gadget_layer(root: TreeLayer, resolving_player: int, opponent_values: np.ndarray) -> TreeLayer:
  """Resolving gadget in front of the root layer. In each root history
  the opponent chooses to terminate (action 0, receiving the given
  counterfactual value of its infoset) or to follow (action 1) into the
  root history. The resolving player has a single dummy action.

  Args:
    opponent_values: [H] P1 values of terminating in each root history.
  """
  num_histories = root.num_histories
  opponent = 1 - resolving_player
  player_legal = np.zeros((2, num_histories, 2), dtype=np.float32)
  player_legal[resolving_player, :, 0] = 1
  player_legal[opponent] = 1
  utility = np.zeros((num_histories, 2, 2), dtype=np.float64)
  next_history = np.full((num_histories, 2, 2), -1, dtype=np.int32)
  if opponent == 0:
    utility[:, 0, 0] = opponent_values
    next_history[:, 1, 0] = np.arange(num_histories)
  else:
    utility[:, 0, 0] = opponent_values
    next_history[:, 0, 1] = np.arange(num_histories)
  kind = np.full(num_histories, DECISION, dtype=np.int8)
  history_iset, iset_legal, iset_kind, iset_tensors = _assign_isets(kind, root.infosets, player_legal)
  return TreeLayer(
      states=root.states, game_legal=root.game_legal, kind=kind,
      decisions=np.zeros(num_histories, dtype=np.int32),
      parent=np.full(num_histories, -1, dtype=np.int32),
      parent_action=np.full((2, num_histories), -1, dtype=np.int32),
      player_legal=player_legal, chance_probs=np.zeros((num_histories, 2)),
      utility=utility, next_history=next_history, state_tensors=root.state_tensors,
      infosets=root.infosets, public_states=root.public_states,
      history_iset=history_iset, iset_legal=iset_legal, iset_kind=iset_kind,
      iset_tensors=iset_tensors, chances=np.zeros(num_histories, dtype=np.int32))


def link_gadget(root: TreeLayer, resolving_player: int) -> TreeLayer:
  """Returns the root layer with parents pointing into the gadget layer,
  where the opponent reached it by the follow action."""
  parent_action = np.zeros((2, root.num_histories), dtype=np.int32)
  parent_action[1 - resolving_player] = 1
  return dataclasses.replace(root, parent=np.arange(root.num_histories, dtype=np.int32), parent_action=parent_action)
