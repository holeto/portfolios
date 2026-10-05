"""Training of the blueprint, portfolios (transformations or HullCover pools) and matrix valued
states with checkpointing."""
from __future__ import annotations

import resource
import time

import jax
import numpy as np

from evaluation import blueprint_exploitability
from train.blueprint_and_mvs import RNaDSolver
from train.run_config import (RunConfig, check_fork_compatible, check_matches_saved, list_checkpoints,
                              load_checkpoint, model_dir, parent_model_dir, save_checkpoint, save_config_once)


def _peak_memory() -> str:
  """Peak resident memory of the process and, on accelerators, the peak memory in use on each
  device (not the memory preallocated by JAX), in GiB."""
  # ru_maxrss is in KiB on Linux.
  usage = [f"peak_rss_gib: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.3g}"]
  for device in jax.local_devices():
    stats = device.memory_stats()  # None on CPU.
    if stats and "peak_bytes_in_use" in stats:
      usage.append(f"peak_{device.platform}{device.id}_gib: {stats['peak_bytes_in_use'] / 2**30:.3g}")
  return ", ".join(usage)


def _fork(config: RunConfig, directory: str, fork_from: int) -> RNaDSolver:
  """A solver with the hyperparameters of the fork and the state (weights, optimizer
  states, learner step and random key) of the run at the given step."""
  parent = parent_model_dir(config)
  changes = check_fork_compatible(config, parent)
  parent_solver = load_checkpoint(parent, fork_from)
  solver = RNaDSolver(config.rnad, config.game.name, config.game.params)
  expected = jax.tree.map(lambda x: np.shape(x), solver.state)
  loaded = jax.tree.map(lambda x: np.shape(x), parent_solver.state)
  if jax.tree.structure(expected) != jax.tree.structure(loaded) or jax.tree.leaves(expected) != jax.tree.leaves(loaded):
    raise ValueError(f"The checkpoint {fork_from} of {parent} does not fit the networks of the fork.")
  solver.state = parent_solver.state
  solver.hullcover_state = parent_solver.hullcover_state
  save_config_once(config, directory)
  save_checkpoint(solver, directory)
  print(f"Forking {parent} at step {fork_from} into {directory}", flush=True)
  for field, saved, provided in changes:
    print(f"  {field}: {saved} -> {provided}", flush=True)
  return solver


def _restore_or_create(config: RunConfig, directory: str) -> RNaDSolver:
  checkpoints = list_checkpoints(directory)
  resume, fork_from = config.checkpoint.resume, config.checkpoint.fork_from

  if fork_from is not None and not checkpoints:
    return _fork(config, directory, fork_from)

  if resume is False:
    if checkpoints:
      raise FileExistsError(f"{directory} already contains checkpoints {checkpoints}. "
                            "Set checkpoint.resume to true (latest) or a step to continue training.")
    save_config_once(config, directory)
    check_matches_saved(config, directory)
    print(f"Training a new model in {directory}", flush=True)
    return RNaDSolver(config.rnad, config.game.name, config.game.params)

  if not checkpoints:
    raise FileNotFoundError(f"Nothing to resume, {directory} has no checkpoints.")
  check_matches_saved(config, directory)
  step = checkpoints[-1] if resume is True else int(resume)
  later = [c for c in checkpoints if c > step]
  if later:
    raise ValueError(f"Resuming {directory} from step {step} would overwrite the later checkpoints {later}. "
                     f"Set checkpoint.fork_from: {step} to continue from it in a fork of the run instead.")
  print(f"Resuming training from {directory} at step {step}", flush=True)
  return load_checkpoint(directory, step)


def train(config: RunConfig) -> RNaDSolver:
  """Trains until `config.steps` learner steps, saving into the model directory."""
  directory = model_dir(config)
  solver = _restore_or_create(config, directory)
  checkpoint = config.checkpoint
  if solver.learner_steps >= config.steps:
    print(f"The model already has {solver.learner_steps} >= {config.steps} learner steps.", flush=True)
    return solver

  if solver.is_hullcover and solver.hullcover_state is None:
    if solver.learner_steps > 0:
      raise ValueError(f"The checkpoint at step {solver.learner_steps} has no HullCover state.")
    solver.hullcover_round()  # The snapshot of the initial policy.

  cache = {}

  def evaluate_blueprint():
    results = blueprint_exploitability.run(solver, solver.game, None, blueprint_exploitability.Config(), cache)
    print(f"Step {solver.learner_steps} blueprint exploitability: {results['exploitability']:.5f}", flush=True)

  if checkpoint.save_first and solver.learner_steps == 0:
    if checkpoint.eval_every > 0:
      evaluate_blueprint()
    print(f"Saved {save_checkpoint(solver, directory)}", flush=True)

  start = time.time()
  while solver.learner_steps < config.steps:
    logs = solver.step()
    step = solver.learner_steps
    if solver.is_hullcover and step == solver.hullcover_state.next_round_step():
      solver.hullcover_round()
    if checkpoint.print_every > 0 and step % checkpoint.print_every == 0:
      print(f"[{time.time() - start:7.1f}s] " + ", ".join(f"{k}: {v:.4g}" for k, v in logs.items())
            + f", {_peak_memory()}", flush=True)
      if solver.is_hullcover:
        print(f"  {solver.hullcover_state.summary()}", flush=True)
    if checkpoint.eval_every > 0 and step % checkpoint.eval_every == 0:
      evaluate_blueprint()
    if checkpoint.save_every > 0 and step % checkpoint.save_every == 0:
      save_checkpoint(solver, directory)
  path = save_checkpoint(solver, directory)
  print(f"Saved {path}", flush=True)
  print(f"Peak memory: {_peak_memory()}", flush=True)
  return solver
