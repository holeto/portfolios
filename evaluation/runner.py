"""Runs the evaluation part of the config on a trained checkpoint."""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Optional, Union

import numpy as np

from evaluation import DEFAULT_TESTS, EVALUATIONS
from gameplay.test_time_search import TestTimeSearchConfig
from train.run_config import RunConfig, check_matches_saved, list_checkpoints, load_checkpoint, model_dir

# Set per player by the tests, not configurable.
_SEARCH_INTERNAL_FIELDS = ("player",)


@dataclasses.dataclass
class ResolvedEval:
  search: TestTimeSearchConfig
  tests: list  # of (name, module, test config)
  defaulted: list  # of (field path, default value)


def _build(cls, values: Optional[dict], section: str, excluded=()):
  values = dict(values or {})
  fields = [f for f in dataclasses.fields(cls) if f.name not in excluded]
  names = {f.name for f in fields}
  unknown = set(values) - names
  if unknown:
    raise ValueError(f"Unknown fields {sorted(unknown)} in '{section}'. Valid fields: {sorted(names)}.")
  config = cls(**values)
  defaulted = [(f"{section}.{f.name}", getattr(config, f.name)) for f in fields if f.name not in values]
  return config, defaulted


def resolve_eval_config(eval_raw: dict) -> ResolvedEval:
  """Dataclass defaults overridden by the eval part of the config (YAML and --set)."""
  unknown = set(eval_raw) - {"search", "tests"}
  if unknown:
    raise ValueError(f"Unknown fields {sorted(unknown)} in 'eval', expected 'search' and 'tests'.")
  search, defaulted = _build(TestTimeSearchConfig, eval_raw.get("search"), "eval.search", _SEARCH_INTERNAL_FIELDS)
  tests_raw = eval_raw.get("tests")
  if tests_raw is None:
    tests_raw = DEFAULT_TESTS
    defaulted.append(("eval.tests", list(DEFAULT_TESTS)))
  tests = []
  for name, params in tests_raw.items():
    if name not in EVALUATIONS:
      raise ValueError(f"Unknown evaluation test '{name}', available: {sorted(EVALUATIONS)}.")
    module = EVALUATIONS[name]
    test_config, test_defaulted = _build(module.Config, params, f"eval.tests.{name}")
    tests.append((name, module, test_config))
    defaulted.extend(test_defaulted)
  return ResolvedEval(search=search, tests=tests, defaulted=defaulted)


def _report_eval_config(eval_raw: dict, resolved: ResolvedEval):
  if not eval_raw:
    print("WARNING: No eval parameters were provided, all the defaults are used:")
  elif resolved.defaulted:
    print("WARNING: The following eval parameters were not provided, their defaults are used:")
  for path, value in resolved.defaulted:
    print(f"  {path} = {value}")
  search = {f.name: getattr(resolved.search, f.name) for f in dataclasses.fields(resolved.search)
            if f.name not in _SEARCH_INTERNAL_FIELDS}
  print(f"Eval search config: {search}")
  for name, _, test_config in resolved.tests:
    print(f"Eval test {name}: {dataclasses.asdict(test_config)}")


def _jsonable(value):
  if isinstance(value, dict):
    return {k: _jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(v) for v in value]
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  return value


def _result_path(directory: str, step: int, name: str) -> str:
  return os.path.join(directory, "eval", f"step_{step}_{name}.json")


def run_evaluation(config: RunConfig, restore_step: Union[None, int, str] = None, skip_existing: bool = False) -> dict:
  """Evaluates the checkpoint at `restore_step` of the model given by the config:
  the latest one if None, all the checkpoints if "all". With `skip_existing`, tests
  with results already saved for a checkpoint are not run again.
  Returns {step: {test: results}}."""
  directory = model_dir(config)
  checkpoints = list_checkpoints(directory)
  if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found in {directory}.")
  check_matches_saved(config, directory)
  if restore_step is None:
    steps = [checkpoints[-1]]
  elif restore_step == "all":
    steps = checkpoints
  else:
    steps = [int(restore_step)]

  resolved = resolve_eval_config(config.eval)
  _report_eval_config(config.eval, resolved)
  search = {f.name: getattr(resolved.search, f.name) for f in dataclasses.fields(resolved.search)
            if f.name not in _SEARCH_INTERNAL_FIELDS}

  # The cache (e.g. the full game tree) is shared by all the tests and checkpoints.
  all_results, cache = {}, {}
  os.makedirs(os.path.join(directory, "eval"), exist_ok=True)
  for step in steps:
    tests = [(name, module, test_config) for name, module, test_config in resolved.tests
             if not (skip_existing and os.path.exists(_result_path(directory, step, name)))]
    if not tests:
      print(f"Skipping step {step}, all the results exist.", flush=True)
      continue
    solver = load_checkpoint(directory, step)
    print(f"Evaluating {directory} at step {step}", flush=True)
    results = {}
    for name, module, test_config in tests:
      print(f"Running {name}...", flush=True)
      result = _jsonable(module.run(solver, solver.game, resolved.search, test_config, cache))
      results[name] = result
      print(f"{name}: {result}", flush=True)
      with open(_result_path(directory, step, name), "w") as f:
        json.dump({"step": step, "test": name, "test_config": dataclasses.asdict(test_config),
                   "search_config": search, "results": result}, f, indent=2)
    all_results[step] = results
  return all_results
