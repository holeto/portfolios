"""Validation of the full-game CFR against a reference Nash equilibrium.

The reference is a pickle (value_p1, value_p2, infoset_maps, behaviorals) where
infoset_maps[depth][player] is an [I, infoset_size] array of infoset tensors and
behaviorals[depth][player][i] is the behavioral strategy in infoset_maps[depth][player][i].
The values are in the units of the reference (e.g. chips for Leduc, where the
environment divides the rewards by `max_bet_amount`).

Runs CFR on the full game and reports at the given iteration checkpoints the
NashConv and value against the reference, and the per infoset differences of
the average strategy from the reference behaviorals. Note that equilibria need
not be unique, so per infoset differences may remain in some infosets (mostly
the ones with low reach) even for an exact equilibrium.

Usage:
  uv run python evaluation/cfr_validation.py --iterations 100 1000 10000
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

from games import make_game
from iig_algorithms.depth_limited_cfr import DepthLimitedCFR
from iig_algorithms.exploitability import (_finalize_policies, _reaches, best_response_values,
                                           history_policies_from_tables, infoset_key)
from iig_algorithms.tree_builder import DECISION, TreeLayer, build_full_tree

DEFAULT_REFERENCE = os.path.join(os.path.dirname(__file__), "leduc_nash.pkl")
LEDUC_CARDS = "JJQQKK"
LEDUC_ACTIONS = "fcr"  # fold, call, raise


def load_reference(path: str) -> dict:
  with open(path, "rb") as f:
    value_p1, value_p2, infoset_maps, behaviorals = pickle.load(f)
  tables = ({}, {})
  for depth_maps, depth_behaviorals in zip(infoset_maps, behaviorals):
    for player in range(2):
      maps, probs = np.asarray(depth_maps[player]), np.asarray(depth_behaviorals[player])
      for tensor, prob in zip(maps.reshape(-1, maps.shape[-1]) if maps.size else [], probs):
        tables[player][infoset_key(tensor)] = prob
  return {"value_p1": float(value_p1), "value_p2": float(value_p2), "tables": tables}


def describe_infoset(game_name: str, tensor: np.ndarray) -> str:
  """Human readable infoset (Leduc only, otherwise the nonzero entries)."""
  if game_name == "leduc" and tensor.shape[-1] == 39:
    player = int(np.argmax(tensor[:2])) + 1
    # Cards are indexed by rank and suit, e.g. J0 and J1 are the two jacks.
    private_idx = int(np.argmax(tensor[2:8]))
    private = f"{LEDUC_CARDS[private_idx]}{private_idx % 2}"
    public_idx = int(np.argmax(tensor[8:15]))
    public = f"{LEDUC_CARDS[public_idx - 1]}{(public_idx - 1) % 2}" if public_idx > 0 else "-"
    history = tensor[15:].reshape(-1, 3)
    actions = "".join(LEDUC_ACTIONS[int(np.argmax(row))] for row in history if row.sum() > 0)
    return f"P{player} card={private} public={public} history='{actions}'"
  return f"nonzero={np.flatnonzero(tensor).tolist()}"


def decision_tables(layers: list[TreeLayer], strategies, num_actions: int) -> tuple[dict, dict]:
  """Tabular policies (infoset key -> probs) from per layer infoset strategies
  (D x Pl x [I, W]) in the decision infosets with more than one legal action."""
  tables = ({}, {})
  for d, layer in enumerate(layers):
    for player in range(2):
      for iset_id in range(layer.iset_legal[player].shape[0]):
        if layer.iset_kind[player][iset_id] == DECISION and layer.iset_legal[player][iset_id].sum() > 1:
          probs = np.asarray(strategies[d][player][iset_id][:num_actions], dtype=np.float64)
          tables[player][infoset_key(layer.iset_tensors[player][iset_id])] = probs
  return tables


def infoset_reach(layers: list[TreeLayer], tables) -> tuple[dict, dict]:
  """Probability of reaching each infoset (key -> reach) under the tabular profile, including chance."""
  policies = _finalize_policies(layers, history_policies_from_tables(layers, tables))
  reaches = _reaches(layers, policies)
  result = ({}, {})
  for d, layer in enumerate(layers):
    history_reach = np.prod(reaches[d], axis=0)
    for player in range(2):
      iset_reach = np.zeros(layer.iset_legal[player].shape[0])
      np.add.at(iset_reach, layer.history_iset[player], history_reach)
      for iset_id, reach in enumerate(iset_reach):
        result[player][infoset_key(layer.iset_tensors[player][iset_id])] = reach
  return result


def compare_tables(tables, reference_tables, reference_reach, game_name: str, top_k: int,
                   tv_threshold: float) -> dict:
  """Total variation distances between tabular policies and the reference in
  all infosets of `tables` (both players)."""
  rows = []
  for player in range(2):
    for key, ours in tables[player].items():
      ours = np.asarray(ours, dtype=np.float64)
      reference = reference_tables[player].get(key)
      if reference is None:
        continue
      reference = np.asarray(reference, dtype=np.float64)[:len(ours)]
      tv = 0.5 * float(np.abs(ours - reference).sum())
      rows.append((tv, reference_reach[player].get(key, 0.0), key, ours, reference))
  tvs = np.array([r[0] for r in rows])
  reach = np.array([r[1] for r in rows])
  summary = {
      "infosets_compared": len(rows),
      "mean_tv": float(tvs.mean()),
      "max_tv": float(tvs.max()),
      "reach_weighted_mean_tv": float((tvs * reach).sum() / max(reach.sum(), 1e-30)),
      f"infosets_tv_above_{tv_threshold}": int((tvs > tv_threshold).sum()),
      "max_tv_reached_infosets": float(tvs[reach > 1e-3].max()) if np.any(reach > 1e-3) else 0.0,
  }
  if top_k > 0:
    print(f"  Top {top_k} infosets by reach weighted TV distance:")
    for tv, r, key, ours, reference in sorted(rows, key=lambda x: -x[0] * x[1])[:top_k]:
      tensor = np.frombuffer(key, dtype=np.float32)
      print(f"    tv={tv:.4f} reach={r:.4f} {describe_infoset(game_name, tensor)}"
            f" ours={(np.round(ours, 3) + 0.0).tolist()} ref={(np.round(reference, 3) + 0.0).tolist()}")
  return summary


def format_summary(summary: dict) -> str:
  return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items())


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--game", default="leduc")
  parser.add_argument("--reference", default=DEFAULT_REFERENCE)
  parser.add_argument("--iterations", type=int, nargs="+", default=[100, 1000, 10000],
                      help="Iteration checkpoints at which the CFR average strategy is evaluated.")
  parser.add_argument("--value_scale", type=float, default=None,
                      help="Multiplier from environment rewards to the reference units "
                           "(default: the game's max_bet_amount if it has one, else 1).")
  parser.add_argument("--top_k", type=int, default=5, help="Print the top K differing infosets at each checkpoint.")
  parser.add_argument("--tv_threshold", type=float, default=0.05)
  parser.add_argument("--nash_conv_tolerance", type=float, default=1e-3, help="In the reference units.")
  parser.add_argument("--value_tolerance", type=float, default=1e-3, help="In the reference units.")
  args = parser.parse_args()

  game = make_game(args.game)
  scale = args.value_scale if args.value_scale is not None else float(getattr(game, "max_bet_amount", 1.0))
  num_actions = game.num_distinct_actions()
  reference = load_reference(args.reference)

  start = time.time()
  layers = build_full_tree(game)
  print(f"Built the {args.game} tree in {time.time() - start:.1f}s, histories per layer: "
        f"{[l.num_histories for l in layers]}")
  reference_result = best_response_values(layers, history_policies_from_tables(layers, reference["tables"]))
  print(f"Reference: stored value {reference['value_p1']:.6f}, value under our environment "
        f"{reference_result['value'] * scale:.6f}, NashConv {reference_result['nash_conv'] * scale:.2e} (scale {scale})")
  reference_reach = infoset_reach(layers, reference["tables"])

  cfr = DepthLimitedCFR(layers, np.ones((3, 1)))
  done, solve_time, summary = 0, 0.0, {}
  for checkpoint in sorted(args.iterations):
    start = time.time()
    cfr.multiple_steps(checkpoint - done)
    cfr.regrets[0][0].block_until_ready()
    solve_time += time.time() - start
    done = checkpoint
    strategies, _, _ = cfr.analyze()
    result = best_response_values(layers, strategies)
    nash_conv, value = result["nash_conv"] * scale, result["value"] * scale
    print(f"Iterations {done}: NashConv {nash_conv:.3e}, value {value:.6f} "
          f"(reference {reference['value_p1']:.6f}, diff {value - reference['value_p1']:+.2e}), "
          f"CFR time {solve_time:.1f}s")
    tables = decision_tables(layers, cfr.average_strategies(), num_actions)
    summary = compare_tables(tables, reference["tables"], reference_reach, args.game, args.top_k, args.tv_threshold)
    print("  Behavioral differences: " + format_summary(summary))

  passed = nash_conv <= args.nash_conv_tolerance and abs(value - reference["value_p1"]) <= args.value_tolerance
  print(f"{'PASSED' if passed else 'FAILED'}: final NashConv {nash_conv:.3e} (tolerance {args.nash_conv_tolerance}), "
        f"value difference {abs(value - reference['value_p1']):.3e} (tolerance {args.value_tolerance})")
  raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
  main()
