# AGENTS.md

## Maintaining this file

Focus on clarity, precision and brevity. Remove session anecdotes, commit references,
measured numbers and change history. Raise any comment that describes buggy or
non-conforming behavior with the user instead of documenting it here.

## Purpose

Depth-limited solving of two-player zero-sum imperfect-information games with
portfolios (SePoT). An RNaD blueprint is trained together with per-player policy
transformations (the portfolio) and a value network over portfolio options. At test
time, depth-limited CFR solves from the current public state with these values at the
leaves, and resolves later public states behind a gadget.

## Setup and commands

The repository is a uv project, installed into `.venv` in editable mode, so the
packages below are importable from any script. Run everything from the repository
root: models are saved relative to the working directory.

```
uv sync
uv run python sepot.py --config configs/<name>.yaml --mode train|eval|train_eval [--restore_step N|all] [--skip_existing] [--set eval.KEY=VALUE ...]
uv run python evaluation/cfr_validation.py      # full-game CFR vs the Leduc reference equilibrium
uv run python evaluation/dl_validation.py       # depth-limited CFR (+ resolving) with exact leaves vs the reference
uv run python evaluation/gadget_validation.py   # gadget safety under an approximation error
uv run python plotting/plot_nashconv.py --config configs/<name>.yaml
```

`sepot.py` is the only script in the root. Further entry points belong to
`evaluation/`, `plotting/` or `debug/`.

## Layout

- `sepot.py`: CLI entry point. Loads the YAML config and runs training and/or evaluation.
- `configs/`: run configs (sections `game`, `train`, `checkpoint`, `eval`).
- `games/`: JAX games implementing `JaxGame` (`jax_game.py`), registered in `games/__init__.py` (`make_game`).
  Goofspiel (fixed or random point cards), Leduc (full and first round only),
  RPS (simultaneous, stochastic, turn-based with an optional value error).
- `iig_algorithms/`: game-generic algorithms.
  - `tree_builder.py`: layered (BFS) tree over real game states, with chance, depth-limit
    leaves and the resolving gadget layer. Defines `LeafValues` and the value types.
  - `depth_limited_cfr.py`: jitted CFR+ (alternating updates, linear averaging) over the
    layers, with chance and both leaf value types.
  - `exploitability.py`: best responses, NashConv, exact portfolio values, fixed-continuation leaves.
  - `normal_form_leaves.py`: exact matrix valued leaves with the whole continuation in
    normal form, and conversion of the leaf mixtures back to behavioral strategies.
- `train/`: neural network training.
  - `blueprint_and_mvs.py`: `RNaDSolver`, simultaneous-move RNaD (on-policy) with the
    transformation and value networks.
  - `run_config.py`: config schema, `--set` overrides, model directories, checkpoints, config matching.
  - `trainer.py`: training loop with checkpointing and blueprint evaluation.
- `gameplay/test_time_search.py`: `TestTimeSearch` (depth-limited solving and resolving
  for one player), baseline agents and `play_match`.
- `evaluation/`: evaluation tests and validation scripts.
  - `__init__.py`: registry of the tests run by `sepot.py` (`EVALUATIONS`).
  - `runner.py`: resolves the `eval` config and runs the tests on checkpoints.
  - `blueprint_exploitability.py`, `search_exploitability.py`: registered tests.
  - `cfr_validation.py`, `dl_validation.py`, `gadget_validation.py`: standalone validations.
  - `leduc_nash.pkl`: reference Leduc equilibrium `(value_p1, value_p2, infoset_maps, behaviorals)`,
    per depth and player, values in chips.
- `plotting/`: plotting scripts. Figures are saved as `.pdf`, with a `.csv` of the plotted values.
- `debug/`: diagnostic scripts, run by path (not a package; `debug_utils.py` is imported as a sibling module).
  - `leaf_value_error.py`: resolving with learned vs exact leaf values of the same portfolio, and the error of the learned leaf values.
  - `leaf_option_selection.py`: bias and wrong option rate of the opponent's choice in multi valued leaves.
- `trained_networks_{mvs,mavs}/`, `plots/`: generated outputs.

## Conventions

- Utilities and values are from the perspective of player 1.
- Games are simultaneous-move. Turn-based games give the non-acting player the single
  legal dummy action 0. In chance nodes player 2 "plays" the outcomes (outcome k is
  action k) and player 1 has the dummy action.
- Infoset and public state tensors are exact; infosets are identified by their tensor
  (`infoset_key`), never by similarity. Tabular policies map infoset keys to action probabilities.
- Leduc rewards are divided by `max_bet_amount`; validation scripts report chips.
- A chance node is never followed by another chance node (design choice): consecutive
  chance events are absorbed into a single chance node.
- Depth limit: `depth_limit` counts decision layers only (chance layers are always
  expanded); `leaf_after_chances` alternatively places the leaves at the first decisions
  after a number of chance events since the root of each solve.
- Value types (`train.value_type`), with K transformations and option 0 the blueprint:
  - `multi_valued`: player i stays on its blueprint; in each infoset s_i the opponent
    picks the option minimizing the counterfactual value of s_i. Leaves have no actions.
    The network outputs `1 + 2K` values: [both blueprints, P2 transformations, P1 transformations].
  - `matrix_valued`: both players choose an option at the leaf by CFR. The network outputs
    `(K+1)^2` values, index `i * (K+1) + j` for P1 option i and P2 option j.
- The V-trace clipping is decoupled: `rho_vtrace` clips the RNaD targets (including `1/mu` in the
  NeuRD Q-values), `mvs_rho_vtrace` (default 1.0) the importance ratios of the portfolio policies in
  the value network targets. Runs saved before `mvs_rho_vtrace` existed were trained without clipping
  (`LEGACY_DEFAULTS` in `train/blueprint_and_mvs.py`).
- Leaf value functions take `(states, state_tensors, game_legal)` and return `[H, Pl, K]`
  (multi valued) or `[H, K, K]` (matrix valued), optionally with legal options `[H, Pl, K]`.
- Resolving uses the gadget of the opponent: the resolving player's and chance reaches
  come from the previous average strategy, the opponent may terminate with its
  counterfactual values from the previous solve.

## Configs and checkpoints

- Models are saved to `./trained_networks_{mvs|mavs}/{checkpoint.save}/{game.to_compact_str()}/seed_{seed}/`
  as `step_{N}.pkl`, with `config.yaml` written once and evaluation results in `eval/step_{N}_{test}.json`.
- `train.steps` is the total target of learner steps; `checkpoint.resume` (`true` for the latest checkpoint,
  or a step) continues up to it. `checkpoint.save_first` also saves the initial model as `step_0.pkl`.
- Resuming never overwrites later checkpoints. To continue from an earlier step, `checkpoint.fork_from: S`
  forks the run into `<model dir>/fork_from_S/`, starting from the state (weights, optimizer states,
  learner step, random key) of its `step_S.pkl`; the fork is then trained, resumed, evaluated and plotted
  with its own config. A fork may change the training hyperparameters except `FORK_FIXED_FIELDS`
  (`train/run_config.py`: the network shapes and the seed). With unchanged hyperparameters, a fork
  continues the trajectory of the run deterministically, so finer `save_every`/`eval_every` inspects a window of it.
  `checkpoint.fork_name` distinguishes forks from the same step (`fork_from_S_<name>/`).
- Numbers in configs may use scientific notation with or without a decimal point (`5e-5`, `5.0e-5`).
- Evaluation and resuming fail if the game or the training hyperparameters (all of
  `train` except `steps`) differ from the saved `config.yaml`. Only `eval.*` may be
  overridden with `--set`; eval fields left at their defaults are reported as a warning.
- Adding an evaluation test: a module in `evaluation/` with a `Config` dataclass and
  `run(solver, game, search_config, test_config, cache) -> dict`, registered in `evaluation/__init__.py`.
- Adding a top-level package: list it under `[tool.hatch.build.targets.wheel]` in `pyproject.toml` and run `uv sync`.
