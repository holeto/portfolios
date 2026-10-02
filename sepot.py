"""SePoT entrypoint: training of the blueprint with portfolios and matrix valued
states, and evaluation of the depth-limited test-time search.

Modes:
  train       Trains a new model, or continues training (checkpoint.resume).
  eval        Evaluates a trained model, found from the config. The game and the
              training hyperparameters must match the config saved with the model.
  train_eval  Trains and then evaluates the final checkpoint.

Examples:
  uv run python sepot.py --config configs/goofspiel4.yaml --mode train
  uv run python sepot.py --config configs/goofspiel4.yaml --mode eval --restore_step 5000 \\
      --set eval.search.depth_limit=2 --set eval.tests.search_exploitability.leaf_values=exact
  uv run python sepot.py --config configs/leduc_sepot.yaml --mode eval --restore_step all --skip_existing
"""
import argparse

from evaluation.runner import run_evaluation
from train.run_config import load_config
from train.trainer import train


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--config", required=True, help="Path to the YAML config.")
  parser.add_argument("--mode", required=True, choices=["train", "eval", "train_eval"])
  parser.add_argument("--restore_step", default=None,
                      help="Checkpoint step to evaluate, 'all' for all the checkpoints, the latest one by default.")
  parser.add_argument("--skip_existing", action="store_true",
                      help="Do not rerun the tests with results already saved for a checkpoint.")
  parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="eval.KEY=VALUE",
                      help="Override a value of the eval part of the config (repeatable), e.g. eval.search.depth_limit=2.")
  args = parser.parse_args()

  config = load_config(args.config, args.overrides)
  if args.mode in ("train", "train_eval"):
    solver = train(config)
  if args.restore_step not in (None, "all"):
    args.restore_step = int(args.restore_step)
  if args.mode == "eval":
    run_evaluation(config, args.restore_step, args.skip_existing)
  elif args.mode == "train_eval":
    run_evaluation(config, args.restore_step if args.restore_step is not None else solver.learner_steps,
                   args.skip_existing)


if __name__ == "__main__":
  main()
