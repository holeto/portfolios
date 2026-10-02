"""Exact matrix valued leaves with the whole continuation in normal form.

In each depth-limit leaf infoset s_i, the options of player i are all its pure
strategies in the rest of the game below s_i (one action in each of its decision
points there). The leaf value of a history is the exact P1 value of each pair
of pure strategies. With perfect recall, the depth-limited game is then
strategically equivalent to the full game, so it is useful to validate the
depth-limited solving. The mixed strategies over the options found by CFR are
converted back into behavioral strategies by `behavioral_tables`.

Only feasible when the continuation is short, the amount of options is the
product of the amounts of legal actions over all decision points below s_i.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np

from games.jax_game import JaxGame
from iig_algorithms.exploitability import infoset_key, portfolio_pair_values
from iig_algorithms.tree_builder import DECISION, LEAF, MATRIX_VALUED, LeafValues, TreeLayer, build_tree


@dataclasses.dataclass
class LeafNormalForm:
  """The normal form of a player's continuation below one of its leaf infosets."""
  decision_points: list   # infoset keys of the player's decision points below the leaf infoset
  legal_actions: list     # per decision point the legal actions
  paths: dict             # decision point key -> ((decision point index, action), ...) of the player before it
  pure_strategies: list   # per option an action for each decision point


class NormalFormLeaves:
  """Factory of the exact normal form leaf values, remembering the normal forms
  of all leaf infosets for converting the leaf strategies back."""

  def __init__(self, game: JaxGame):
    self.game = game
    self.normal_forms: dict = {}  # (player, leaf infoset key) -> LeafNormalForm

  def leaf_values(self) -> LeafValues:
    return LeafValues(fn=self._leaf_value_fn, value_type=MATRIX_VALUED, num_options=None)

  def _leaf_value_fn(self, states, state_tensors, game_legal):
    layers = build_tree(self.game, states, game_legal)
    num_leaves = layers[0].num_histories
    num_actions = self.game.num_distinct_actions()
    # The leaf (root of the continuation) of every history.
    roots = [np.arange(num_leaves)]
    for layer in layers[1:]:
      roots.append(roots[-1][layer.parent])

    leaf_forms = [[None] * num_leaves for _ in range(2)]
    history_dp_keys = [[] for _ in range(2)]  # per layer, per history the player's decision point key or None
    for player in range(2):
      forms = {}
      paths_by_layer = [[() for _ in range(num_leaves)]]
      for d, layer in enumerate(layers):
        if d > 0:
          prev_layer, prev_paths = layers[d - 1], paths_by_layer[d - 1]
          paths = []
          for h in range(layer.num_histories):
            parent, action = layer.parent[h], layer.parent_action[player, h]
            path = prev_paths[parent]
            if self._decides(prev_layer, player, parent):
              path = path + ((infoset_key(prev_layer.infosets[player, parent]), int(action)),)
            paths.append(path)
          paths_by_layer.append(paths)
        dp_keys = [None] * layer.num_histories
        for h in range(layer.num_histories):
          if not self._decides(layer, player, h):
            continue
          leaf_key = infoset_key(layers[0].infosets[player, roots[d][h]])
          form = forms.setdefault(leaf_key, {"keys": [], "legal": [], "paths": {}})
          key = infoset_key(layer.infosets[player, h])
          if key not in form["paths"]:
            form["keys"].append(key)
            form["legal"].append(np.flatnonzero(layer.player_legal[player, h, :num_actions] > 0).tolist())
            form["paths"][key] = paths_by_layer[d][h]
          dp_keys[h] = key
        history_dp_keys[player].append(dp_keys)

      for h in range(num_leaves):
        leaf_key = infoset_key(layers[0].infosets[player, h])
        if (player, leaf_key) not in self.normal_forms:
          form = forms.get(leaf_key, {"keys": [], "legal": [], "paths": {}})
          index = {k: i for i, k in enumerate(form["keys"])}
          paths = {k: tuple((index[q], a) for q, a in path) for k, path in form["paths"].items()}
          self.normal_forms[(player, leaf_key)] = LeafNormalForm(
              decision_points=form["keys"], legal_actions=form["legal"], paths=paths,
              pure_strategies=list(itertools.product(*form["legal"])))
        leaf_forms[player][h] = self.normal_forms[(player, leaf_key)]

    num_options = max(len(f.pure_strategies) for forms in leaf_forms for f in forms)
    option_legal = np.zeros((num_leaves, 2, num_options), dtype=np.float32)
    for player in range(2):
      for h in range(num_leaves):
        option_legal[h, player, :len(leaf_forms[player][h].pure_strategies)] = 1

    # One-hot policies of every pure strategy (padding options repeat the first one).
    portfolio_policies = []
    for d, layer in enumerate(layers):
      policies = np.zeros((2, num_options, layer.num_histories, layer.width))
      for player in range(2):
        for h, key in enumerate(history_dp_keys[player][d]):
          if key is None:
            continue
          form = leaf_forms[player][roots[d][h]]
          j = form.decision_points.index(key)
          for k in range(num_options):
            policies[player, k, h, form.pure_strategies[k % len(form.pure_strategies)][j]] = 1
      portfolio_policies.append(policies)
    return portfolio_pair_values(layers, portfolio_policies), option_legal

  @staticmethod
  def _decides(layer: TreeLayer, player: int, h: int) -> bool:
    return layer.kind[h] == DECISION and layer.player_legal[player, h].sum() > 1

  def behavioral_tables(self, layers: list[TreeLayer], average_strategies) -> tuple[dict, dict]:
    """Behavioral strategies (infoset key -> probs) in the continuation, from the
    mixed strategies over the pure strategies in the leaf infosets of the tree."""
    num_actions = self.game.num_distinct_actions()
    tables = ({}, {})
    for d, layer in enumerate(layers):
      for player in range(2):
        for iset_id in np.flatnonzero(layer.iset_kind[player] == LEAF):
          form = self.normal_forms[(player, infoset_key(layer.iset_tensors[player][iset_id]))]
          mixed = np.asarray(average_strategies[d][player][iset_id][:len(form.pure_strategies)], dtype=np.float64)
          pure = np.array(form.pure_strategies, dtype=np.int64).reshape(len(form.pure_strategies), -1)
          for j, key in enumerate(form.decision_points):
            # The pure strategies playing all of the player's actions leading to the decision point.
            reaching = np.ones(len(pure), dtype=bool)
            for q, action in form.paths[key]:
              reaching &= pure[:, q] == action
            probs = np.zeros(num_actions)
            np.add.at(probs, pure[reaching, j], mixed[reaching])
            if probs.sum() > 1e-12:
              probs /= probs.sum()
            else:
              probs[form.legal_actions[j]] = 1.0 / len(form.legal_actions[j])
            tables[player][key] = probs
    return tables
