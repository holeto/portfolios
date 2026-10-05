"""YAML run configuration, model directories and checkpoints.

The config has four sections:
  game:        name and constructor parameters of the game.
  train:       `steps` (total learner steps) and the RNaDConfig hyperparameters.
  checkpoint:  save (the model root directory), save_every, save_first, print_every, eval_every, resume,
               fork_from.
  eval:        the evaluation part, resolved by `evaluation.runner`.
"""
from __future__ import annotations

import copy
import dataclasses
import glob
import os
import pickle
import re
from typing import Any, Optional, Union

import yaml

from games import make_game
from games.jax_game import JaxGame
from iig_algorithms.hull_cover import HullCoverConfig
from iig_algorithms.tree_builder import MATRIX_VALUED, MULTI_VALUED
from train.blueprint_and_mvs import LEGACY_DEFAULTS, AdamConfig, RNaDConfig, RNaDSolver

CONFIG_FILENAME = "config.yaml"


class _ConfigLoader(yaml.SafeLoader):
  """YAML loader also reading scientific notation without a decimal point (e.g. 5e-5)
  as floats, which YAML 1.1 reads as strings."""


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                |[-+]?\.(?:inf|Inf|INF)
                |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."))


def _load_yaml(stream):
  return yaml.load(stream, Loader=_ConfigLoader)
SECTIONS = ("game", "train", "checkpoint", "eval")


@dataclasses.dataclass(frozen=True)
class GameConfig:
  name: str
  params: dict = dataclasses.field(default_factory=dict)

  def make(self) -> JaxGame:
    return make_game(self.name, **self.params)


@dataclasses.dataclass(frozen=True)
class CheckpointConfig:
  # The model root directory, models are saved into
  # ./trained_networks_{mvs|mavs}/{save}/{game.to_compact_str()}/seed_{seed}/ for multi
  # and matrix valued states respectively.
  save: str = "default"
  # Save every N learner steps, 0 saves only at the end of training.
  save_every: int = 0
  # Also save (and evaluate with eval_every) the initial model at step 0 of a new training.
  save_first: bool = False
  print_every: int = 500
  # Blueprint exploitability every N learner steps, 0 disables.
  eval_every: int = 0
  # false: train from scratch, true: continue from the latest checkpoint, int: continue from that
  # step (only the latest one, use fork_from to continue from an earlier step).
  resume: Union[bool, int] = False
  # Fork of the run from the checkpoint at this step, living in the subfolder fork_from_{step}/ of
  # the run. Its first training starts from the weights (and optimizer states) of the run at
  # that step, then it behaves as any run. The training hyperparameters may differ from the run,
  # except the ones defining the networks (FORK_FIXED_FIELDS).
  fork_from: Optional[int] = None
  # Name distinguishing forks from the same step, the fork then lives in fork_from_{step}_{name}/.
  fork_name: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class RunConfig:
  game: GameConfig
  steps: int
  rnad: RNaDConfig
  checkpoint: CheckpointConfig
  eval: dict


def _from_dict(cls, values: dict, section: str):
  """Constructs a dataclass, failing on unknown fields."""
  values = dict(values or {})
  fields = {f.name: f for f in dataclasses.fields(cls)}
  unknown = set(values) - set(fields)
  if unknown:
    raise ValueError(f"Unknown fields {sorted(unknown)} in config section '{section}'. Valid fields: {sorted(fields)}.")
  for name, value in values.items():
    if isinstance(value, list):
      values[name] = tuple(value)
  return cls(**values)


def parse_rnad_config(train_section: dict) -> RNaDConfig:
  values = dict(train_section)
  values.pop("steps", None)
  if "adam" in values:
    values["adam"] = _from_dict(AdamConfig, values["adam"], "train.adam")
  if values.get("hullcover") is not None:
    values["hullcover"] = _from_dict(HullCoverConfig, values["hullcover"], "train.hullcover")
  return _from_dict(RNaDConfig, values, "train")


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
  """Applies `key.subkey=value` overrides (values parsed as YAML). Only the eval part may be overridden."""
  raw = copy.deepcopy(raw)
  for override in overrides:
    if "=" not in override:
      raise ValueError(f"Override '{override}' is not of the form key=value.")
    key, value = override.split("=", 1)
    path = key.strip().split(".")
    if path[0] != "eval" or len(path) < 2:
      raise ValueError(f"Only the eval part of the config can be overridden, got '{key}'.")
    node = raw
    for part in path[:-1]:
      if node.get(part) is None:
        node[part] = {}
      node = node[part]
    node[path[-1]] = _load_yaml(value)
  return raw


def parse_config(raw: dict) -> RunConfig:
  unknown = set(raw) - set(SECTIONS)
  if unknown:
    raise ValueError(f"Unknown config sections {sorted(unknown)}, expected {SECTIONS}.")
  if "game" not in raw or "name" not in raw["game"]:
    raise ValueError("The config needs a game section with the game name.")
  game = GameConfig(name=raw["game"]["name"], params=dict(raw["game"].get("params") or {}))
  train_section = raw.get("train") or {}
  if "steps" not in train_section:
    raise ValueError("The train section needs the total amount of learner `steps`.")
  return RunConfig(game=game, steps=int(train_section["steps"]), rnad=parse_rnad_config(train_section),
                   checkpoint=_from_dict(CheckpointConfig, raw.get("checkpoint"), "checkpoint"),
                   eval=copy.deepcopy(raw.get("eval") or {}))


def load_config(path: str, overrides: list[str] = ()) -> RunConfig:
  with open(path) as f:
    raw = _load_yaml(f) or {}
  return parse_config(apply_overrides(raw, list(overrides)))


def _plain(value: Any) -> Any:
  """Converts dataclasses and tuples into YAML friendly dicts and lists."""
  if dataclasses.is_dataclass(value):
    return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
  if isinstance(value, (list, tuple)):
    return [_plain(v) for v in value]
  if isinstance(value, dict):
    return {k: _plain(v) for k, v in value.items()}
  return value


def canonical_train(config: RunConfig) -> dict:
  """The train section with all the defaults filled in."""
  return {"steps": config.steps, **_plain(config.rnad)}


def canonical_dict(config: RunConfig) -> dict:
  return {"game": _plain(config.game), "train": canonical_train(config),
          "checkpoint": _plain(config.checkpoint), "eval": _plain(config.eval)}


def parent_model_dir(config: RunConfig) -> str:
  """The directory of the run, without the fork subfolder."""
  game = config.game.make()
  value_suffix = {MULTI_VALUED: "mvs", MATRIX_VALUED: "mavs"}[config.rnad.value_type]
  model_save_dir = (f"/trained_networks_{value_suffix}/{config.checkpoint.save}/{game.to_compact_str()}"
                    f"/seed_{config.rnad.seed}/")
  return os.getcwd() + model_save_dir


def model_dir(config: RunConfig) -> str:
  """The directory of the run, or of its fork with checkpoint.fork_from."""
  directory = parent_model_dir(config)
  if config.checkpoint.fork_from is not None:
    name = f"fork_from_{config.checkpoint.fork_from}"
    if config.checkpoint.fork_name:
      name += f"_{config.checkpoint.fork_name}"
    directory = os.path.join(directory, name) + os.sep
  return directory


def checkpoint_path(directory: str, step: int) -> str:
  return os.path.join(directory, f"step_{step}.pkl")


def list_checkpoints(directory: str) -> list[int]:
  steps = []
  for path in glob.glob(os.path.join(directory, "step_*.pkl")):
    match = re.fullmatch(r"step_(\d+)\.pkl", os.path.basename(path))
    if match:
      steps.append(int(match.group(1)))
  return sorted(steps)


def save_checkpoint(solver: RNaDSolver, directory: str) -> str:
  path = checkpoint_path(directory, solver.learner_steps)
  tmp_path = path + ".tmp"
  with open(tmp_path, "wb") as f:
    pickle.dump(solver, f)
  os.replace(tmp_path, path)
  return path


def load_checkpoint(directory: str, step: int) -> RNaDSolver:
  path = checkpoint_path(directory, step)
  if not os.path.exists(path):
    available = list_checkpoints(directory)
    raise FileNotFoundError(f"Checkpoint {path} does not exist. Available steps: {available or 'none'}.")
  with open(path, "rb") as f:
    return pickle.load(f)


def save_config_once(config: RunConfig, directory: str):
  """Writes the config into the model directory, unless it is already there."""
  path = os.path.join(directory, CONFIG_FILENAME)
  if os.path.exists(path):
    return
  os.makedirs(directory, exist_ok=True)
  with open(path, "w") as f:
    yaml.safe_dump(canonical_dict(config), f, sort_keys=False)


def load_saved_config(directory: str) -> RunConfig:
  path = os.path.join(directory, CONFIG_FILENAME)
  if not os.path.exists(path):
    raise FileNotFoundError(f"No saved config at {path}.")
  with open(path) as f:
    raw = _load_yaml(f)
  # Hyperparameters added after the model was saved take the values it was trained with.
  train_section = raw.setdefault("train", {})
  for key, value in LEGACY_DEFAULTS.items():
    train_section.setdefault(key, value)
  return parse_config(raw)


# Hyperparameters a fork has to share with its run: they define the shapes of the
# networks, or (the seed) locate the run, or (the portfolio method) the state continued by the fork.
FORK_FIXED_FIELDS = ("actor_network_layers", "critic_network_layers", "mvs_network_layers",
                     "transformation_network_layers", "num_transformations", "value_type", "seed",
                     "portfolio_method", "hullcover")


def config_differences(config: RunConfig, directory: str) -> list[tuple[str, Any, Any]]:
  """(field, saved, provided) of the game and the training hyperparameters (all of the
  train section except `steps`) that differ from the config saved in the model directory."""
  saved = load_saved_config(directory)
  differences = []
  if _plain(saved.game) != _plain(config.game):
    differences.append(("game", _plain(saved.game), _plain(config.game)))
  saved_train, provided_train = canonical_train(saved), canonical_train(config)
  for key in provided_train:
    if key != "steps" and saved_train.get(key) != provided_train[key]:
      differences.append((f"train.{key}", saved_train.get(key), provided_train[key]))
  return differences


def _format(differences) -> str:
  return "\n".join(f"  {field}: saved {saved}, provided {provided}" for field, saved, provided in differences)


def check_matches_saved(config: RunConfig, directory: str):
  """Fails if the game or the training hyperparameters differ from the config saved in the model directory."""
  differences = config_differences(config, directory)
  if differences:
    raise ValueError(f"The provided config does not match the one saved in {directory}:\n" + _format(differences))


def check_fork_compatible(config: RunConfig, directory: str) -> list[tuple[str, Any, Any]]:
  """Fails if the fork differs from its run in the game or in the hyperparameters defining the
  networks (FORK_FIXED_FIELDS). Returns the other, allowed, differences of the hyperparameters."""
  differences = config_differences(config, directory)
  fixed = {"game"} | {f"train.{field}" for field in FORK_FIXED_FIELDS}
  incompatible = [d for d in differences if d[0] in fixed]
  if incompatible:
    raise ValueError(f"A fork must keep the game and the hyperparameters {FORK_FIXED_FIELDS} of its run "
                     f"in {directory}:\n" + _format(incompatible))
  return [d for d in differences if d[0] not in fixed]
