"""Compares the NashConv with resolving of several runs (e.g. a run and its forks).

Plots the blueprint NashConv of the first run and the NashConv with resolving of every
run, from the evaluation results saved by `run.py --mode eval --restore_step all`
(of the portfolio size eval.hullcover.k for HullCover runs, or --k).
Saves the figure (.pdf) and a CSV table of the plotted values.

Usage:
  uv run python plotting/compare_runs.py --config configs/a.yaml configs/b.yaml --labels "Run" "Fork" \\
      --output plots/comparison.pdf
  uv run python plotting/compare_runs.py --config configs/leduc_hullcover.yaml configs/leduc_sepot_mvs.yaml \\
      --labels "HullCover" "SePoT" --max_step 10000 --output plots/hullcover_vs_sepot.pdf
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402

from plot_nashconv import GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, load_results  # noqa: E402
from train.run_config import load_config, model_dir  # noqa: E402

# Validated categorical slots in fixed order (light mode).
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def _portfolio_size(config, k=None):
  """The evaluated portfolio size of a HullCover run, None for other runs."""
  if config.rnad.portfolio_method != "hullcover":
    return None
  return k if k is not None else (config.eval.get("hullcover") or {}).get("k")


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--config", nargs="+", required=True, help="Run configs, the first one gives the blueprint.")
  parser.add_argument("--labels", nargs="+", default=None, help="A label per config.")
  parser.add_argument("--output", required=True, help="Figure path (.pdf).")
  parser.add_argument("--title", default="NashConv with resolving")
  parser.add_argument("--k", type=int, default=None,
                      help="Portfolio size of the HullCover runs (default: eval.hullcover.k of each config).")
  parser.add_argument("--max_step", type=int, default=None, help="Plot only the checkpoints up to this step.")
  parser.add_argument("--scale", type=float, default=None,
                      help="Multiplier of the NashConv (default: the game's max_bet_amount if it has one, else 1).")
  args = parser.parse_args()
  labels = args.labels or [os.path.splitext(os.path.basename(c))[0] for c in args.config]
  if len(labels) != len(args.config):
    raise ValueError("Give a label per config.")
  if len(args.config) + 1 > len(COLORS):
    raise ValueError(f"At most {len(COLORS) - 1} runs.")

  configs = [load_config(path) for path in args.config]
  game = configs[0].game.make()
  max_bet = getattr(game, "max_bet_amount", None)
  scale = args.scale if args.scale is not None else float(max_bet or 1.0)
  unit = "chips" if args.scale is None and max_bet else ""
  results = [load_results(model_dir(config), _portfolio_size(config, args.k)) for config in configs]
  if args.max_step is not None:
    results = [{test: {s: v for s, v in values.items() if s <= args.max_step} for test, values in r.items()}
               for r in results]

  series = [("Blueprint", COLORS[0], results[0]["blueprint_exploitability"])]
  series += [(f"{label} + resolving", COLORS[i + 1], r["search_exploitability"])
             for i, (label, r) in enumerate(zip(labels, results))]

  fig, ax = plt.subplots(figsize=(9, 5))
  fig.patch.set_facecolor(SURFACE)
  ax.set_facecolor(SURFACE)
  for label, color, values in series:
    steps = sorted(values)
    if not steps:
      continue
    ys = [values[s] * scale for s in steps]
    ax.plot(steps, ys, color=color, linewidth=2, marker="o", markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
    ax.annotate(f"{ys[-1]:.3g}", xy=(steps[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=TEXT_PRIMARY)
  ax.set_yscale("log")
  plain = ticker.FuncFormatter(lambda value, _: f"{value:g}")
  ax.yaxis.set_major_formatter(plain)
  ax.yaxis.set_minor_formatter(plain)
  ax.set_xlabel("Training steps", color=TEXT_SECONDARY)
  ax.set_ylabel(f"NashConv{f' ({unit})' if unit else ''}, log scale", color=TEXT_SECONDARY)
  fig.suptitle(args.title, color=TEXT_PRIMARY, x=0.02, ha="left", fontsize=12)
  ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8)
  ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4)
  ax.tick_params(which="both", colors=TEXT_SECONDARY, labelsize=9)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(TEXT_SECONDARY)
  ax.legend(frameon=False, loc="upper right", labelcolor=TEXT_PRIMARY, fontsize=9)
  fig.tight_layout()
  os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
  fig.savefig(args.output, facecolor=SURFACE)
  plt.close(fig)

  table = os.path.splitext(args.output)[0] + ".csv"
  steps = sorted(set().union(*(values for _, _, values in series)))
  with open(table, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step"] + [label for label, _, _ in series])
    for step in steps:
      writer.writerow([step] + [values[step] * scale if step in values else "" for _, _, values in series])
  print(f"Saved {args.output} and {table}")


if __name__ == "__main__":
  main()
