"""Gameplay with depth-limited resolving.

In each of its decisions the search agent:
  1. Takes the histories of the current public state from the tree of the
     previous solve, with the ranges of the resolving player and chance
     given by the previous average strategy.
  2. Builds a depth-limited tree below them, prefixed by a resolving gadget
     where the opponent may terminate with its counterfactual values from
     the previous solve. The depth-limit leaves have values of the pairs of
     portfolio policies (e.g. the matrix valued states of RNaD-SePoT).
  3. Solves it with CFR and plays according to the average strategy.
The very first solve starts from the initial state of the game, without a gadget.

The game step counts all transitions since the initial state, including chance.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from games.jax_game import JaxGame
from iig_algorithms.depth_limited_cfr import DepthLimitedCFR
from iig_algorithms.tree_builder import (CHANCE, DECISION, LeafValues, TreeLayer, build_tree, gadget_layer, link_gadget,
                          tree_take)


@dataclasses.dataclass(frozen=True)
class TestTimeSearchConfig:
  player: int = 0
  resolve_iterations: int = 1000
  # Amount of decision layers in each solve, chance layers are not counted.
  # None solves the rest of the game.
  depth_limit: Optional[int] = 1
  # Alternatively (or additionally), the depth-limit leaves are the first decisions
  # after this amount of chance events since the root of each solve, e.g. 2 in Leduc
  # solves the first betting round from the initial state and the rest of the game
  # from the public states after the public card. None disables.
  leaf_after_chances: Optional[int] = None
  # Resolve in every decision. Otherwise the strategy from the last solve is
  # reused while the current decision is within its depth limit.
  resolve_every_step: bool = False
  # Root histories where the resolving player and chance reach is below are pruned.
  prune_threshold: float = 1e-10


@dataclasses.dataclass
class Solution:
  layers: list[TreeLayer]
  cfr: DepthLimitedCFR
  root_step: int    # Game step of the root layer (after the gadget).
  offset: int       # 1 if the gadget layer precedes the root layer.
  used: bool = False

  def layer_index(self, step: int) -> int:
    return step - self.root_step + self.offset


class TestTimeSearch:
  """Search agent with depth-limited resolving."""

  def __init__(self, game: JaxGame, config: TestTimeSearchConfig, leaf_values: Optional[LeafValues] = None):
    if (config.depth_limit is not None or config.leaf_after_chances is not None) and leaf_values is None:
      raise ValueError("Depth-limited solving requires leaf values.")
    self.game = game
    self.config = config
    self.leaf_values = leaf_values
    self.actions = game.num_distinct_actions()
    self.solution: Optional[Solution] = None

  def reset(self):
    self.solution = None

  def build_tree(self, root_states, root_legal) -> list[TreeLayer]:
    """The (depth-limited) tree below the given root histories, as built by the solves."""
    is_leaf = None
    if self.config.leaf_after_chances is not None:
      leaf_after_chances = self.config.leaf_after_chances
      is_leaf = lambda states, decisions, chances: chances >= leaf_after_chances
    return build_tree(self.game, root_states, root_legal, self.config.depth_limit, self.leaf_values, is_leaf)

  def _solve(self, layers: list[TreeLayer], init_reaches: np.ndarray) -> DepthLimitedCFR:
    cfr = DepthLimitedCFR(layers, init_reaches)
    cfr.multiple_steps(self.config.resolve_iterations)
    return cfr

  def initial_tree(self) -> list[TreeLayer]:
    """The tree of the first solve, from the initial state of the game."""
    init_state, init_legal = self.game.initialize_structures()
    root = jax.tree.map(lambda x: jnp.asarray(x)[None], init_state)
    return self.build_tree(root, np.asarray(init_legal)[None])

  def initial_solution(self) -> Solution:
    layers = self.initial_tree()
    return Solution(layers=layers, cfr=self._solve(layers, np.ones((3, 1))), root_step=0, offset=0)

  def next_solution(self, solution: Solution, step: int, public_state: np.ndarray) -> Solution:
    """Resolves the public state reached at the given step of the game."""
    player, opponent = self.config.player, 1 - self.config.player
    layer_idx = solution.layer_index(step)
    if layer_idx >= len(solution.layers):
      raise ValueError(f"Step {step} is beyond the tree of the previous solve.")
    layer = solution.layers[layer_idx]
    public_state = np.asarray(public_state, dtype=np.float32)
    in_public_state = np.all(layer.public_states == public_state[None], axis=-1) & (layer.kind != CHANCE)
    if not np.any(in_public_state):
      raise ValueError(f"Public state at step {step} not found in the tree of the previous solve.")

    reaches = solution.cfr.history_reaches()[layer_idx]
    keep = in_public_state & (reaches[player] * reaches[2] > self.config.prune_threshold)
    if not np.any(keep):
      keep = in_public_state
    root_idx = np.flatnonzero(keep)

    opponent_iset_values = solution.cfr.iset_cf_values()[layer_idx][opponent]
    gadget_values = opponent_iset_values[layer.history_iset[opponent][root_idx]]

    init_reaches = np.ones((3, root_idx.size))
    init_reaches[player] = reaches[player][root_idx] / max(np.max(reaches[player][root_idx]), 1e-30)
    init_reaches[2] = reaches[2][root_idx] / max(np.sum(reaches[2][root_idx]), 1e-30)

    tree = self.build_tree(tree_take(layer.states, root_idx), layer.game_legal[root_idx])
    gadget = gadget_layer(tree[0], player, gadget_values)
    tree[0] = link_gadget(tree[0], player)
    layers = [gadget] + tree
    return Solution(layers=layers, cfr=self._solve(layers, init_reaches), root_step=step, offset=1)

  def _can_reuse(self, step: int, infoset: np.ndarray) -> bool:
    solution = self.solution
    if solution.used and self.config.resolve_every_step:
      return False
    layer_idx = solution.layer_index(step)
    if layer_idx >= len(solution.layers):
      return False
    return solution.layers[layer_idx].find_iset(self.config.player, infoset, DECISION) >= 0

  def get_policy(self, public_state: np.ndarray, iset: np.ndarray, legal: np.ndarray, step: int) -> np.ndarray:
    """Policy [A] in the current decision, resolving if needed."""
    if self.solution is None:
      self.solution = self.initial_solution()
    # The decision may already be beyond the depth limit of the initial solve.
    if not self._can_reuse(step, iset):
      self.solution = self.next_solution(self.solution, step, public_state)
    solution = self.solution
    solution.used = True
    strategy = solution.cfr.get_strategy(solution.layer_index(step), self.config.player, iset, DECISION)
    policy = np.asarray(strategy[:self.actions], dtype=np.float64) * (np.asarray(legal) > 0)
    return policy / policy.sum()

  def act(self, public_state, iset, legal, step: int, rng: np.random.Generator) -> int:
    policy = self.get_policy(public_state, iset, legal, step)
    return int(rng.choice(self.actions, p=policy))


class RandomAgent:
  """Plays uniformly over the legal actions."""

  def reset(self):
    pass

  def act(self, public_state, iset, legal, step: int, rng: np.random.Generator) -> int:
    legal = np.asarray(legal, dtype=np.float64) > 0
    return int(rng.choice(np.flatnonzero(legal)))


class PolicyAgent:
  """Plays according to a policy function, e.g. the RNaD blueprint `solver.policy`."""

  def __init__(self, policy_fn):
    self.policy_fn = policy_fn

  def reset(self):
    pass

  def act(self, public_state, iset, legal, step: int, rng: np.random.Generator) -> int:
    policy = np.asarray(self.policy_fn(np.asarray(iset)[None], np.asarray(legal, dtype=np.float32)[None])[0],
                        dtype=np.float64)
    policy = policy * (np.asarray(legal) > 0)
    return int(rng.choice(policy.shape[0], p=policy / policy.sum()))


def play_game(game: JaxGame, agents, rng: np.random.Generator) -> float:
  """Plays a single game, returns the P1 return."""
  for agent in agents:
    agent.reset()
  state, legal = game.initialize_structures()
  step, total_reward, terminal = 0, 0.0, False
  while not terminal:
    if bool(game.is_chance(state)):
      outcomes, probs = game.get_outcomes_and_probs(state)
      probs = np.asarray(probs, dtype=np.float64)
      outcome = rng.choice(probs.shape[0], p=probs / probs.sum())
      state, terminal, reward, legal = game.apply_action(state, jnp.asarray(outcomes[outcome], dtype=jnp.int32))
    else:
      _, p1_iset, p2_iset, public_state = (np.asarray(x) for x in game.get_info(state))
      legal_np = np.asarray(legal)
      actions = [agents[pl].act(public_state, iset, legal_np[pl], step, rng)
                 for pl, iset in enumerate((p1_iset, p2_iset))]
      state, terminal, reward, legal = game.apply_action(state, jnp.asarray(actions, dtype=jnp.int32))
    total_reward += float(reward)
    terminal = bool(terminal)
    step += 1
  return total_reward


def play_match(game: JaxGame, agents, num_games: int, seed: int = 0) -> dict:
  """Mean P1 return and its standard error over the given amount of games."""
  rng = np.random.default_rng(seed)
  returns = np.array([play_game(game, agents, rng) for _ in range(num_games)])
  return {"mean": float(returns.mean()), "stderr": float(returns.std() / np.sqrt(num_games)), "returns": returns}
