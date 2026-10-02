"""Validation that resolving with the gadget is robust to approximation errors.

Turn-based RPS (player 1 picks, then player 2 picks without observing it):
  1. The root solve (true game, depth limit 1) has the depth-limit leaves at the
     decision of player 2, with values of the uniform continuation (Nash). Player 1
     plays uniformly and its counterfactual values at the leaves are 0.
  2. Player 2 resolves its decision in a game with an approximation error: winning
     with rock pays 1 + 3 * epsilon. Against the uniform range of player 1, the values
     of player 2 are epsilon for rock and 0 for paper and scissors.
     - Unsafe resolve (fixed ranges, no gadget): player 2 learns to play pure rock,
       exploitable by 1 in the true game.
     - Gadget resolve: player 1 may terminate with its counterfactual value 0, so player
       2 has to keep every action of player 1 at value <= 0, which only allows strategies
       within O(epsilon) of uniform, and player 1 terminates instead of playing scissors.
The exploitability of player 2 is evaluated in the true game. Since the values in the
resolve differ by at most 3 * epsilon, sound resolving keeps it at most the
exploitability of the blueprint (0) plus 3 * epsilon.

Usage:
  uv run python evaluation/gadget_validation.py [--epsilon 0.05] [--iterations 10000]
"""
from __future__ import annotations

import argparse

import numpy as np

from games import make_game
from gameplay.test_time_search import TestTimeSearch, TestTimeSearchConfig
from iig_algorithms.depth_limited_cfr import DepthLimitedCFR
from iig_algorithms.exploitability import ExploitabilityEvaluator, fixed_continuation_leaf_values, infoset_key
from iig_algorithms.tree_builder import DECISION, LEAF, build_tree

RESOLVING_PLAYER = 1
STEP = 1  # The decision of player 2 is after a single transition.


def _round(x) -> list:
  return (np.round(np.asarray(x, dtype=np.float64), 4) + 0.0).tolist()


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--epsilon", type=float, default=0.05, help="The approximation error in the resolve.")
  parser.add_argument("--iterations", type=int, default=10000, help="CFR iterations of each solve.")
  parser.add_argument("--tolerance", type=float, default=1e-3)
  args = parser.parse_args()

  true_game = make_game("turn_based_rps")
  perturbed_game = make_game("turn_based_rps", p2_rock_win_error=args.epsilon)
  evaluator = ExploitabilityEvaluator(true_game)
  leaf_values = fixed_continuation_leaf_values(true_game, ({}, {}))  # uniform continuation

  # 1. Root solve in the true game.
  config = TestTimeSearchConfig(player=RESOLVING_PLAYER, depth_limit=1, resolve_iterations=args.iterations)
  root = TestTimeSearch(true_game, config, leaf_values).initial_solution()
  root_layer, leaf_layer = root.layers[0], root.layers[1]
  p1_strategy = root.cfr.average_strategies()[0][0][root_layer.history_iset[0][0]][:3]
  full_layer = evaluator.layers[1]  # the decision of player 2 in the full game tree, histories by P1 action
  leaf_cf_values = root.cfr.iset_cf_values()[1][0]
  p1_cf_values = np.array([leaf_cf_values[leaf_layer.find_iset(0, full_layer.infosets[0, h], LEAF)]
                           for h in range(full_layer.num_histories)])
  print(f"Root solve (true game): P1 strategy {_round(p1_strategy)}, P1 counterfactual values at the leaves "
        f"(R, P, S) {_round(p1_cf_values)}, the decision of P2 is a depth-limit leaf: "
        f"{bool(np.all(leaf_layer.kind == LEAF))}")

  # 2. Resolve of player 2 in the game with the approximation error.
  resolver = TestTimeSearch(perturbed_game, config, leaf_values)
  resolved = resolver.next_solution(root, STEP, leaf_layer.public_states[0])
  p2_infoset = leaf_layer.infosets[RESOLVING_PLAYER, 0]
  safe = resolved.cfr.get_strategy(resolved.offset, RESOLVING_PLAYER, p2_infoset, DECISION)[:3]
  gadget = resolved.cfr.average_strategies()[0][0]
  follow = np.array([gadget[resolved.layers[0].find_iset(0, full_layer.infosets[0, h]), 1]
                     for h in range(full_layer.num_histories)])

  reaches = root.cfr.history_reaches()[1]
  init_reaches = np.stack([reaches[0], np.ones_like(reaches[1]), reaches[2]])
  subgame = build_tree(perturbed_game, leaf_layer.states, leaf_layer.game_legal, 1, leaf_values)
  unsafe_cfr = DepthLimitedCFR(subgame, init_reaches)
  unsafe_cfr.multiple_steps(args.iterations)
  unsafe = unsafe_cfr.get_strategy(0, RESOLVING_PLAYER, p2_infoset, DECISION)[:3]

  perturbed_payoff = build_tree(perturbed_game, leaf_layer.states, leaf_layer.game_legal)[0].utility[:, 0, :3]
  p2_values = -(p1_strategy @ perturbed_payoff)
  print(f"Resolve with error epsilon={args.epsilon}: P2 values (R, P, S) against the P1 range {_round(p2_values)}")
  print(f"  Unsafe resolve (no gadget): P2 strategy (R, P, S) {_round(unsafe)}")
  print(f"  Gadget resolve: P2 strategy (R, P, S) {_round(safe)}, P1 follow probabilities (R, P, S) {_round(follow)}")
  p1_values = perturbed_payoff @ np.asarray(safe, dtype=np.float64)
  print(f"  P1 values (R, P, S) against the gadget resolve in the resolved game {_round(p1_values)} "
        f"vs terminate values {_round(p1_cf_values)}")

  # Exploitability of player 2 in the true game: the best response value of player 1 (game value 0).
  p1_table = {infoset_key(root_layer.infosets[0, 0]): np.asarray(p1_strategy)}
  exploitability = {}
  for name, strategy in [("blueprint", np.ones(3) / 3), ("unsafe", unsafe), ("gadget", safe)]:
    tables = (p1_table, {infoset_key(p2_infoset): np.asarray(strategy, dtype=np.float64)})
    exploitability[name] = evaluator.evaluate_tables(tables)["br_p1"]
  print("P2 exploitability in the true game: " + ", ".join(f"{k} {v:.5f}" for k, v in exploitability.items()))

  bound = exploitability["blueprint"] + 3 * args.epsilon
  sound = exploitability["gadget"] <= bound + args.tolerance
  discriminative = exploitability["unsafe"] > bound + args.tolerance
  print(f"{'PASSED' if sound else 'FAILED'}: gadget resolve exploitability {exploitability['gadget']:.5f} "
        f"<= blueprint + 3 * epsilon = {bound:.5f}")
  print(f"{'OK' if discriminative else 'WARNING'}: the unsafe resolve "
        f"{'exceeds' if discriminative else 'does not exceed'} the bound ({exploitability['unsafe']:.5f}), "
        f"{'so the test discriminates' if discriminative else 'the test does not discriminate'}")
  raise SystemExit(0 if sound else 1)


if __name__ == "__main__":
  main()
