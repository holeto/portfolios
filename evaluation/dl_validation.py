"""Validation of depth-limited CFR against a reference Nash equilibrium.

The depth-limited tree is solved from the initial state (no resolving) up to the
leaves after a given amount of chance events (in Leduc 2: the first decisions
after the public card is dealt, i.e. the whole first betting round). Leaf values:
  normal_form: exact matrix valued states, where the options of each player in
    its leaf infoset are all its pure strategies in the rest of the game. The
    depth-limited game is then equivalent to the full game, so CFR should find a
    Nash equilibrium. The continuation is converted back to behavioral strategies.
The following multi valued leaves use the reference continuation instead:
  nash_full_br: player i continues with the reference, the opponent minimizes over
    all of its continuation strategies separately for each infoset s_i. Such an
    opponent may condition on s_i, so it effectively best responds knowing the
    history (its own private information included).
  nash: both players continue with the reference (a single option).
The depth-limited strategy is combined with the continuation and evaluated in the
full game (NashConv, value), and compared per infoset with the reference. The
continuation (--continuation) is either:
  leaves:  taken from the leaves (converted from the normal form, or the reference).
  resolve: each player resolves every public subgame at the leaves with the gadget
           (its own search, as the gadget depends on the resolving player), solving
           the rest of the game below it. The solves use the given iterations.

Usage:
  uv run python evaluation/dl_validation.py --iterations 1000 10000 [--leaf_values normal_form nash nash_full_br]
      [--continuation leaves resolve]
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from evaluation.cfr_validation import (DEFAULT_REFERENCE, compare_tables, decision_tables, format_summary,
                                       infoset_reach, load_reference)
from games import make_game
from games.jax_game import JaxGame
from evaluation.search_exploitability import search_agent_tables
from gameplay.test_time_search import TestTimeSearchConfig
from iig_algorithms.depth_limited_cfr import DepthLimitedCFR
from iig_algorithms.exploitability import (best_response_values, fixed_continuation_leaf_values,
                                           history_policies_from_tables)
from iig_algorithms.normal_form_leaves import NormalFormLeaves
from iig_algorithms.tree_builder import LEAF, LeafValues, build_full_tree, build_tree

LEAF_MODES = ("normal_form", "nash_full_br", "nash")


def reference_leaf_values(game: JaxGame, reference_tables, mode: str) -> LeafValues:
  """Exact multi valued leaf values (a single option) from the reference continuation."""
  if mode not in ("nash_full_br", "nash"):
    raise ValueError(f"Unknown leaf values {mode}, expected 'nash_full_br' or 'nash'.")
  return fixed_continuation_leaf_values(game, reference_tables, opponent_best_response=mode == "nash_full_br")


def build_depth_limited_tree(game: JaxGame, leaf_values: LeafValues, leaf_after_chances: int):
  init_state, init_legal = game.initialize_structures()
  root = jax.tree.map(lambda x: jnp.asarray(x)[None], init_state)
  return build_tree(game, root, np.asarray(init_legal)[None], leaf_values=leaf_values,
                    is_leaf=lambda states, decisions, chances: chances >= leaf_after_chances)


def evaluate_resolving(game, leaf_values: LeafValues, iterations: int, args, full_layers, reference, reference_reach,
                       round1_keys, scale: float):
  """The root solve continued by resolving every public subgame at its leaves, for both players."""
  config = TestTimeSearchConfig(depth_limit=None, leaf_after_chances=args.leaf_after_chances,
                                resolve_iterations=iterations, resolve_every_step=False)
  start = time.time()
  tables, resolves = search_agent_tables(game, config, leaf_values)
  result = best_response_values(full_layers, history_policies_from_tables(full_layers, tables))
  print(f"Resolving every public subgame, {iterations} iterations per solve ({resolves} resolves per player, "
        f"{time.time() - start:.1f}s): NashConv {result['nash_conv'] * scale:.3e}, value {result['value'] * scale:.6f} "
        f"(reference {reference['value_p1']:.6f})")
  for name, in_round1 in (("Depth-limited part", True), ("Resolved continuation", False)):
    part = tuple({k: v for k, v in tables[pl].items() if (k in round1_keys[pl]) == in_round1} for pl in range(2))
    summary = compare_tables(part, reference["tables"], reference_reach, args.game, args.top_k, args.tv_threshold)
    print(f"  {name} vs reference: " + format_summary(summary))


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--game", default="leduc")
  parser.add_argument("--reference", default=DEFAULT_REFERENCE)
  parser.add_argument("--iterations", type=int, nargs="+", default=[1000, 10000])
  parser.add_argument("--leaf_values", nargs="+", default=["normal_form"], choices=LEAF_MODES)
  parser.add_argument("--leaf_after_chances", type=int, default=2,
                      help="Leaves are the first decisions after this amount of chance events.")
  parser.add_argument("--value_scale", type=float, default=None,
                      help="Multiplier from environment rewards to the reference units "
                           "(default: the game's max_bet_amount if it has one, else 1).")
  parser.add_argument("--continuation", nargs="+", default=["leaves", "resolve"], choices=["leaves", "resolve"])
  parser.add_argument("--top_k", type=int, default=5)
  parser.add_argument("--tv_threshold", type=float, default=0.05)
  args = parser.parse_args()

  game = make_game(args.game)
  scale = args.value_scale if args.value_scale is not None else float(getattr(game, "max_bet_amount", 1.0))
  num_actions = game.num_distinct_actions()
  reference = load_reference(args.reference)
  full_layers = build_full_tree(game)
  reference_reach = infoset_reach(full_layers, reference["tables"])
  print(f"Reference value {reference['value_p1']:.6f} (scale {scale})")

  for mode in args.leaf_values:
    start = time.time()
    normal_form = NormalFormLeaves(game) if mode == "normal_form" else None
    leaf_values = normal_form.leaf_values() if normal_form else reference_leaf_values(game, reference["tables"], mode)
    layers = build_depth_limited_tree(game, leaf_values, args.leaf_after_chances)
    num_leaves = sum(int(np.sum(layer.kind == LEAF)) for layer in layers)
    options = (f", options per leaf infoset up to {max(l.width for l in layers if np.any(l.kind == LEAF))}"
               if normal_form else "")
    print(f"\n=== Leaf values: {mode} | depth-limited tree {[l.num_histories for l in layers]} histories per layer, "
          f"{num_leaves} leaves{options}, built in {time.time() - start:.1f}s")
    round1_keys = tuple(set(table) for table in decision_tables(layers, DepthLimitedCFR(layers, np.ones((3, 1)))
                                                                 .average_strategies(), num_actions))
    if "resolve" in args.continuation:
      for iterations in sorted(args.iterations):
        evaluate_resolving(game, leaf_values, iterations, args, full_layers, reference, reference_reach, round1_keys,
                           scale)
    if "leaves" not in args.continuation:
      continue
    cfr = DepthLimitedCFR(layers, np.ones((3, 1)))
    done, cfr_time = 0, 0.0
    for checkpoint in sorted(args.iterations):
      start = time.time()
      cfr.multiple_steps(checkpoint - done)
      cfr.regrets[0][0].block_until_ready()
      cfr_time += time.time() - start
      done = checkpoint
      average = cfr.average_strategies()
      dl_tables = decision_tables(layers, average, num_actions)
      if normal_form:
        continuation, source = normal_form.behavioral_tables(layers, average), "the normal form leaves"
      else:
        continuation, source = reference["tables"], "the reference"
      combined = tuple({**continuation[pl], **dl_tables[pl]} for pl in range(2))
      result = best_response_values(full_layers, history_policies_from_tables(full_layers, combined))
      print(f"Iterations {done} ({cfr_time:.1f}s): depth-limited game value P1 view {cfr.root_value(player=0) * scale:.6f},"
            f" P2 view {cfr.root_value(player=1) * scale:.6f} | with the continuation from {source}: "
            f"NashConv {result['nash_conv'] * scale:.3e}, value {result['value'] * scale:.6f} "
            f"(reference {reference['value_p1']:.6f})")
      summary = compare_tables(dl_tables, reference["tables"], reference_reach, args.game, args.top_k,
                               args.tv_threshold)
      print("  Depth-limited part vs reference: " + format_summary(summary))
      if normal_form:
        summary = compare_tables(continuation, reference["tables"], reference_reach, args.game, args.top_k,
                                 args.tv_threshold)
        print("  Continuation vs reference: " + format_summary(summary))


if __name__ == "__main__":
  main()
