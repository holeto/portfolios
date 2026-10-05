"""Plots the NashConv of the blueprint and of the blueprint with resolving over training.

Reads the evaluation results saved by
  uv run python run.py --config CONFIG --mode eval --restore_step all
(the `blueprint_exploitability` and `search_exploitability` tests) from the model
directory given by the config, and saves the figure and a CSV table of the plotted
values next to them (or to --output).

HullCover runs are evaluated per portfolio size k (results `step_{N}_k{k}_{test}.json`), the
plotted k is --k or the eval.hullcover.k of the config.

Usage:
  uv run python plotting/plot_nashconv.py --config configs/leduc_sepot.yaml
  uv run python plotting/plot_nashconv.py --config configs/leduc_hullcover.yaml --k 4
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402

from train.run_config import load_config, model_dir  # noqa: E402

# Validated categorical slots 1 and 2 (light mode), text and grid inks.
SERIES = {
    "blueprint_exploitability": ("Blueprint", "#2a78d6"),
    "search_exploitability": ("Blueprint + resolving", "#eb6834"),
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def load_results(directory: str, k=None) -> dict:
  """{test: {step: NashConv}} from the saved evaluation results (of portfolio size k for HullCover runs)."""
  results = {test: {} for test in SERIES}
  for path in glob.glob(os.path.join(directory, "eval", "step_*_*.json")):
    match = re.fullmatch(r"step_(\d+)_(?:k(\d+)_)?(.+)\.json", os.path.basename(path))
    if not match or match.group(3) not in SERIES or match.group(2) != (None if k is None else str(k)):
      continue
    with open(path) as f:
      results[match.group(3)][int(match.group(1))] = json.load(f)["results"]["nash_conv"]
  return results


def plot(results: dict, scale: float, unit: str, title: str, subtitle: str, output: str):
  fig, ax = plt.subplots(figsize=(9, 5))
  fig.patch.set_facecolor(SURFACE)
  ax.set_facecolor(SURFACE)
  for test, (label, color) in SERIES.items():
    steps = sorted(results[test])
    if not steps:
      continue
    values = [results[test][s] * scale for s in steps]
    ax.plot(steps, values, color=color, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    # Direct label at the end of the line, in text ink.
    ax.annotate(f"{label}  {values[-1]:.3g}", xy=(steps[-1], values[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=TEXT_PRIMARY)
  ax.set_yscale("log")
  plain = ticker.FuncFormatter(lambda value, _: f"{value:g}")
  ax.yaxis.set_major_formatter(plain)
  ax.yaxis.set_minor_formatter(plain)
  ax.set_xlabel("Training steps", color=TEXT_SECONDARY)
  ax.set_ylabel(f"NashConv{f' ({unit})' if unit else ''}, log scale", color=TEXT_SECONDARY)
  fig.suptitle(title, color=TEXT_PRIMARY, x=0.02, ha="left", fontsize=12)
  ax.set_title(subtitle, color=TEXT_SECONDARY, loc="left", fontsize=9)
  ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8)
  ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4)
  ax.tick_params(which="both", colors=TEXT_SECONDARY, labelsize=9)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(TEXT_SECONDARY)
  ax.legend(frameon=False, loc="upper right", labelcolor=TEXT_PRIMARY, fontsize=9)
  ax.margins(x=0.02)
  fig.tight_layout()
  fig.subplots_adjust(right=0.74)  # room for the direct labels
  fig.savefig(output, facecolor=SURFACE)
  plt.close(fig)


def write_table(results: dict, scale: float, output: str):
  steps = sorted(set().union(*(results[test] for test in SERIES)))
  with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step"] + [SERIES[test][0] for test in SERIES])
    for step in steps:
      writer.writerow([step] + [results[test][step] * scale if step in results[test] else "" for test in SERIES])


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--config", required=True, help="The run config, used to locate the model directory.")
  parser.add_argument("--output", default=None, help="Figure path (.pdf), <model dir>/eval/nashconv.pdf by default.")
  parser.add_argument("--scale", type=float, default=None,
                      help="Multiplier of the NashConv (default: the game's max_bet_amount if it has one, e.g. "
                           "Leduc chips, else 1).")
  parser.add_argument("--k", type=int, default=None,
                      help="Portfolio size of a HullCover run (default: eval.hullcover.k of the config).")
  args = parser.parse_args()

  config = load_config(args.config)
  hullcover = config.rnad.portfolio_method == "hullcover"
  k = (args.k if args.k is not None else (config.eval.get("hullcover") or {}).get("k")) if hullcover else None
  directory = model_dir(config)
  game = config.game.make()
  max_bet = getattr(game, "max_bet_amount", None)
  scale = args.scale if args.scale is not None else float(max_bet or 1.0)
  unit = "chips" if args.scale is None and max_bet else ""
  results = load_results(directory, k)
  if not any(results.values()):
    raise FileNotFoundError(f"No evaluation results in {directory}eval/, run run.py --mode eval first.")
  output = args.output or os.path.join(directory, "eval", f"nashconv{f'_k{k}' if hullcover else ''}.pdf")
  search = config.eval.get("search", {})
  title = f"{game.to_compact_str()}: NashConv of the blueprint and with resolving during training"
  limit = (f"leaves after {search['leaf_after_chances']} chance events" if search.get("leaf_after_chances") is not None
           else f"depth limit {search.get('depth_limit')}")
  portfolio = (f"HullCover k={k} from pools of {config.rnad.hullcover.cap_c}" if hullcover
               else f"K={config.rnad.num_transformations} transformations")
  subtitle = (f"{portfolio}, {config.rnad.value_type} states, {limit}, "
              f"{search.get('resolve_iterations')} CFR iterations per solve")
  plot(results, scale, unit, title, subtitle, output)
  table = os.path.splitext(output)[0] + ".csv"
  write_table(results, scale, table)
  print(f"Saved {output} and {table}")


if __name__ == "__main__":
  main()
