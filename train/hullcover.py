"""HullCover portfolios of the simultaneous move R-NaD.

Two `iig_algorithms.hull_cover.HullCover` instances run on one R-NaD stream, one per covered
player: the P2 portfolio is built against P1 test strategies and the P1 portfolio against P2 test
strategies. Every `every` learner steps a round harvests from the shared actor network:
  policy(theta_t)          the current policy,
  vertex(theta_s, theta_t) per infoset the argmax of the change of the logits (or of the policy)
                           over the last learner step (s = t - 1, the gradient step, by default)
                           or since the previous round (s = t - every, `vertex_window: round`),
and offers both to the candidates of each instance, then the vertex to its tests. The first round
initialises C = {policy(theta_0), first vertex} and X = {first vertex}, after the first learner
step (or the first `every` steps with the round window). The same base strategy is used by both
players (on their own infosets), so the bank of snapshots is shared.

Values of the value matrices M are Monte Carlo estimates of the expected P1 return of pairs of base
strategies, all under one fixed key (common random numbers over the whole run), cached per pair.
A member (a realization-plan mixture of bases) gets the weighted sum of its components' values.

The value network learns the values of the pools during training ("pool values"): the outputs
are the pairs of [blueprint + C slots] of both players in the layouts of the value types, and the
importance ratios of a slot member come from its realization-plan mixture (`pool_ratios`). At
evaluation, the eps-Dom-Mixed-MILP selects k mixtures lam over each pool (`select_portfolio`)
and the leaf values are the lam-linear combinations of the pool values (`combine`), i.e. the
values of drawing the pool member with lam at the leaf.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from iig_algorithms.exploitability import history_policies_from_fn, infoset_key, make_exact_leaf_values, mixture_tables
from iig_algorithms.hull_cover import HullCover, HullCoverConfig
from iig_algorithms.tree_builder import MATRIX_VALUED, LeafValues, build_full_tree

BASE_POLICY = 0
BASE_VERTEX = 1


def make_base_policy(actor, vertex_space: str):
  """base_policy(params_a, params_b, kind, obs, legal) -> policies: the softmax policy of params_b,
  or for vertices the one-hot argmax over the legal actions of the change from params_a to params_b."""
  def base_policy(params_a, params_b, kind, obs, legal):
    pi_a, _, logit_a = actor.apply(params_a, obs, legal)
    pi_b, _, logit_b = actor.apply(params_b, obs, legal)
    return _base_from_outputs(pi_a, logit_a, pi_b, logit_b, kind, legal, vertex_space)
  return base_policy


def _base_from_outputs(pi_a, logit_a, pi_b, logit_b, kind, legal, vertex_space):
  change = logit_b - logit_a if vertex_space == "logit" else pi_b - pi_a
  change = jnp.where(legal > 0, change, -jnp.inf)
  vertex = jax.nn.one_hot(jnp.argmax(change, axis=-1), legal.shape[-1], dtype=pi_b.dtype)
  kind = jnp.reshape(kind, jnp.shape(kind) + (1,) * (pi_b.ndim - jnp.ndim(kind)))
  return jnp.where(kind == BASE_VERTEX, vertex, pi_b)


def _stack(trees):
  return jax.tree.map(lambda *xs: np.stack(xs), *trees)


# Folder in the model directory of a run with the params of the snapshots, each written once.
SNAPSHOTS_DIR = "snapshots"


def snapshot_path(directory: str, sid: int) -> str:
  return os.path.join(directory, SNAPSHOTS_DIR, f"{sid}.pkl")


class StrategyBank:
  """Frozen actor snapshots and the base strategies defined on them.

  The pickled bank refers to its snapshots by id: `persist` writes the params of each snapshot once
  into the snapshots/ folder of the run, `restore` reads them back."""

  def __init__(self):
    self.snapshots: dict[int, dict] = {}       # snapshot id -> actor params (numpy)
    self.snapshot_steps: dict[int, int] = {}   # snapshot id -> learner step
    self.bases: dict[int, tuple[int, int, int]] = {}  # base id -> (kind, snapshot a, snapshot b)
    self._next_snapshot = 0
    self._next_base = 0
    # Snapshot ids written into the snapshots/ folder of the run in `_directory`.
    self.persisted: set[int] = set()
    self._directory: Optional[str] = None

  def __getstate__(self):
    state = {k: v for k, v in self.__dict__.items() if k != "_directory"}
    state["snapshots"] = sorted(self.snapshots)
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self.persisted = set(state.get("persisted", ()))
    self._directory = None
    # Checkpoints from before the snapshots/ folder hold the params themselves.
    if not isinstance(self.snapshots, dict):
      self.snapshots = dict.fromkeys(self.snapshots)

  def persist(self, directory: str):
    """Writes the snapshots missing in the snapshots/ folder of the run in `directory`. A bank
    restored from another run (a fork) writes all its snapshots into its own folder. Files of
    snapshots not in `persisted` (left by an interrupted save) are overwritten."""
    directory = os.path.abspath(directory)
    if directory != self._directory:
      self.persisted, self._directory = set(), directory
    for sid, params in self.snapshots.items():
      if sid not in self.persisted:
        path = snapshot_path(directory, sid)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as f:
          pickle.dump(params, f)
        os.replace(path + ".tmp", path)
        self.persisted.add(sid)

  def restore(self, directory: str):
    """Reads the params of the snapshots from the snapshots/ folder of the run in `directory`."""
    self._directory = os.path.abspath(directory)
    for sid, params in self.snapshots.items():
      if params is None:
        with open(snapshot_path(directory, sid), "rb") as f:
          self.snapshots[sid] = pickle.load(f)

  def add_snapshot(self, params, step: int) -> int:
    sid = self._next_snapshot
    self._next_snapshot += 1
    self.snapshots[sid] = jax.device_get(params)
    self.snapshot_steps[sid] = step
    return sid

  def add_base(self, kind: int, snapshot_a: int, snapshot_b: int) -> int:
    bid = self._next_base
    self._next_base += 1
    self.bases[bid] = (kind, snapshot_a, snapshot_b)
    return bid

  def describe(self, base: int) -> str:
    kind, a, b = self.bases[base]
    if kind == BASE_POLICY:
      return f"policy@{self.snapshot_steps[b]}"
    return f"vertex@{self.snapshot_steps[a]}->{self.snapshot_steps[b]}"

  def gc(self, bases: set, snapshots: set):
    """Keeps the given bases, their snapshots and the given snapshots."""
    self.bases = {b: v for b, v in self.bases.items() if b in bases}
    keep = set(snapshots) | {s for _, a, b in self.bases.values() for s in (a, b)}
    self.snapshots = {s: p for s, p in self.snapshots.items() if s in keep}
    self.snapshot_steps = {s: t for s, t in self.snapshot_steps.items() if s in keep}


class PairEvaluator:
  """Monte Carlo P1 values of pairs of base strategies, `games` games per pair under one key."""

  def __init__(self, solver, vertex_space: str, games: int, chunk: int = 16):
    self.chunk = chunk
    base_policy = make_base_policy(solver.actor, vertex_space)

    def pair_value(p1_a, p1_b, p1_kind, p2_a, p2_b, p2_kind, key):
      def policy_fn(obs, legal):
        return jnp.stack([base_policy(p1_a, p1_b, p1_kind, obs[:, 0], legal[:, 0]),
                          base_policy(p2_a, p2_b, p2_kind, obs[:, 1], legal[:, 1])], axis=1)
      ts = solver.rollout_with_policy(policy_fn, key, games)
      return jnp.mean(jnp.sum(ts.reward, axis=0))

    self._fn = jax.jit(jax.vmap(pair_value, in_axes=(0, 0, 0, 0, 0, 0, None)))

  def __call__(self, bank: StrategyBank, pairs: list[tuple[int, int]], key: np.ndarray) -> np.ndarray:
    values = []
    for start in range(0, len(pairs), self.chunk):
      chunk = pairs[start:start + self.chunk]
      padded = chunk + [chunk[-1]] * (self.chunk - len(chunk))
      args = []
      for side in range(2):
        kinds, snaps_a, snaps_b = zip(*(bank.bases[pair[side]] for pair in padded))
        args += [_stack([bank.snapshots[s] for s in snaps_a]), _stack([bank.snapshots[s] for s in snaps_b]),
                 np.asarray(kinds, dtype=np.int32)]
      values.append(np.asarray(self._fn(*args, jnp.asarray(key)))[:len(chunk)])
    return np.concatenate(values).astype(np.float64)


class HullCoverState:
  """The online part of HullCover: bank, cached pair values and both instances."""

  def __init__(self, config: HullCoverConfig, seed: int):
    self.config = config
    self.bank = StrategyBank()
    self.pair_values: dict[tuple[int, int], float] = {}  # (P1 base, P2 base) -> P1 value
    # Indexed by the covered player: [P1 portfolio vs P2 tests, P2 portfolio vs P1 tests].
    self.instances: list[Optional[HullCover]] = [None, None]
    self.scenario_key = np.asarray(jax.random.fold_in(jax.random.PRNGKey(seed), 0x48C))
    self.prev_snapshot: Optional[int] = None
    self.last_round_step = -1
    self.rounds = 0
    self.pair_evaluations = 0
    self.seconds = 0.0
    self._evaluator = None

  def __getstate__(self):
    return {k: v for k, v in self.__dict__.items() if k != "_evaluator"}

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._evaluator = None

  def bind(self, solver):
    """Rebinds the Monte Carlo evaluator and the value oracles (after creation or unpickling)."""
    if self._evaluator is None:
      self._evaluator = PairEvaluator(solver, self.config.vertex_space, self.config.eval_games)
    for covered, instance in enumerate(self.instances):
      if instance is not None:
        instance.value_fn = self._oracle(covered)

  def _ensure_pairs(self, pairs: set):
    missing = sorted(p for p in pairs if p not in self.pair_values)
    if missing:
      values = self._evaluator(self.bank, missing, self.scenario_key)
      self.pair_values.update(zip(missing, values.tolist()))
      self.pair_evaluations += len(missing)

  def _oracle(self, covered: int):
    """Loss of the covered player: the P1 value of (test, candidate) when covering P2, the
    negated P1 value of (candidate, test) when covering P1."""
    def key(test_base, candidate_base):
      return (test_base, candidate_base) if covered == 1 else (candidate_base, test_base)
    sign = 1.0 if covered == 1 else -1.0

    def value_fn(tests, candidates):
      self._ensure_pairs({key(a, b) for t in tests for c in candidates for a in t for b in c})
      return sign * np.array([[sum(wa * wb * self.pair_values[key(a, b)] for a, wa in t.items() for b, wb in c.items())
                               for c in candidates] for t in tests])
    return value_fn

  def next_round_step(self) -> int:
    """The learner step of the next round: 0 stores the initial policy, the initialisation follows
    after the first vertex window, then every `every` steps."""
    if self.prev_snapshot is None:
      return 0
    if self.instances[0] is None:
      return 1 if self.config.vertex_window == "step" else self.config.every
    return (self.last_round_step // self.config.every + 1) * self.config.every

  def run_round(self, solver, reference_params=None):
    """Harvests the stream at the solver's current learner step. With the step window,
    `reference_params` are the actor params before the last learner step."""
    start = time.time()
    self.bind(solver)
    step = solver.learner_steps
    snapshot = self.bank.add_snapshot(solver.state.params["actor"], step)
    if self.prev_snapshot is not None:
      reference = self.prev_snapshot
      # (With the step window, the previous snapshot may already be garbage collected.)
      if self.config.vertex_window == "step" and self.bank.snapshot_steps.get(self.prev_snapshot) != step - 1:
        if reference_params is None:
          raise ValueError("The step vertex needs the actor params before the last learner step.")
        reference = self.bank.add_snapshot(reference_params, step - 1)
      vertex = {self.bank.add_base(BASE_VERTEX, reference, snapshot): 1.0}
      if self.instances[0] is None:
        # Step 1: C = {first policy, first vertex}, X = {first vertex}.
        first = {self.bank.add_base(BASE_POLICY, self.prev_snapshot, self.prev_snapshot): 1.0}
        for covered in range(2):
          self.instances[covered] = HullCover(self.config, self._oracle(covered))
          self.instances[covered].init([first, vertex], vertex)
      else:
        policy = {self.bank.add_base(BASE_POLICY, snapshot, snapshot): 1.0}
        for covered in (1, 0):
          instance = self.instances[covered]
          instance.offer_candidate(policy)
          instance.offer_candidate(vertex)
          instance.offer_test(vertex)
    self.prev_snapshot = snapshot
    referenced = set().union(*(i.referenced_bases() for i in self.instances if i is not None))
    # The current snapshot starts the next round window (and is the initial policy before the first round).
    keep = {snapshot} if self.config.vertex_window == "round" or self.instances[0] is None else set()
    self.bank.gc(referenced, keep)
    self.pair_values = {p: v for p, v in self.pair_values.items() if p[0] in referenced and p[1] in referenced}
    self.last_round_step = step
    self.rounds += 1
    self.seconds += time.time() - start

  def pools(self) -> list[list[dict]]:
    """The candidate pools [C of P1, C of P2]."""
    return [instance.C if instance is not None else [] for instance in self.instances]

  def summary(self) -> str:
    parts = []
    for covered, name in ((0, "P1"), (1, "P2")):
      instance = self.instances[covered]
      if instance is None:
        continue
      c = instance.counters
      parts.append(f"{name} |C| {len(instance.C)} |X| {len(instance.X)} "
                   f"C+{c['col_append']}/m{c['col_merge']}/r{c['col_replace']}/-{c['col_reject']} "
                   f"X+{c['row_append']}/m{c['row_mix']}/r{c['row_replace']}/-{c['row_reject']}")
    return (f"hullcover rounds {self.rounds}, " + ", ".join(parts)
            + f", bases {len(self.bank.bases)}, snapshots {len(self.bank.snapshots)}, "
              f"pair evaluations {self.pair_evaluations}, {self.seconds:.0f}s")

  def pool_arrays(self, fallback_params) -> dict:
    """Arrays of the pools for the value network loss, per player: snapshot params [Pl, S, ...],
    base snapshots [Pl, B, 2] and kinds [Pl, B], slot weights [Pl, C_max, B] and slot mask
    [Pl, C_max]."""
    return pool_arrays(self.bank, self.pools(), self.config.cap_c, fallback_params)


def pool_arrays(bank: StrategyBank, pools: list[list[dict]], cap: int, fallback_params) -> dict:
  """Per player only the snapshots and bases of its own pool, padded to a common bucket."""
  per_player = []
  for pool in pools:
    bases = sorted({b for member in pool for b in member})
    per_player.append((bases, sorted({s for b in bases for s in bank.bases[b][1:]})))
  num_bases = _pool_bucket(max(len(bases) for bases, _ in per_player))
  num_snapshots = _pool_bucket(max(len(snapshots) for _, snapshots in per_player))
  snap_params = []
  base_snaps = np.zeros((2, num_bases, 2), dtype=np.int32)
  base_kind = np.zeros((2, num_bases), dtype=np.int32)
  weights = np.zeros((2, cap, num_bases), dtype=np.float32)
  slot_mask = np.zeros((2, cap), dtype=np.float32)
  for player, ((bases, snapshots), pool) in enumerate(zip(per_player, pools)):
    params = [bank.snapshots[s] for s in snapshots] or [jax.device_get(fallback_params)]
    snap_params.append(_stack(params + [params[0]] * (num_snapshots - len(params))))
    snapshot_index = {s: i for i, s in enumerate(snapshots)}
    base_index = {b: i for i, b in enumerate(bases)}
    for b, i in base_index.items():
      kind, a, s = bank.bases[b]
      base_kind[player, i] = kind
      base_snaps[player, i] = snapshot_index[a], snapshot_index[s]
    for slot, member in enumerate(pool):
      slot_mask[player, slot] = 1.0
      for b, w in member.items():
        weights[player, slot, base_index[b]] = w
  return {"snap_params": _stack(snap_params), "base_snaps": base_snaps, "base_kind": base_kind,
          "weights": weights, "slot_mask": slot_mask}


def _pool_bucket(n: int, multiple: int = 8) -> int:
  """Padded size of the pool arrays (padding is evaluated too): powers of two up to `multiple`,
  then multiples of it, limiting both the wasted evaluations and the recompilations."""
  if n <= multiple:
    return 1 << max(n - 1, 0).bit_length()
  return -(-n // multiple) * multiple


def pool_ratios(actor, vertex_space: str, pool: dict, pi, ts, player: int):
  """Importance ratios [T, B, 1 + C_max] of the taken actions of a player for its blueprint
  (the current policy pi) and its pool slots, against the sampling policy.

  A slot member is a realization-plan mixture of bases with weights w_b. Its probability of the
  action a_t in s_t is the mixture conditioned on the player's own past actions,
    sum_b w_b R_b(t) pi_b(a_t | s_t) / sum_b w_b R_b(t),   R_b(t) = prod_{t' < t} pi_b(a_t' | s_t'),
  exact since the trajectories start at the initial state (non-acting steps have the single legal
  dummy action, i.e. probability 1). If no component reaches s_t, the prior weights are used.
  Empty slots get ratio 1 (and are masked out of the loss)."""
  obs, legal, action_oh = ts.obs[:, :, player], ts.legal[:, :, player], ts.action_oh[:, :, player]
  valid = ts.valid > 0
  sampling = jnp.sum(action_oh * ts.policy[:, :, player], axis=-1)
  blueprint = jnp.sum(action_oh * pi[:, :, player], axis=-1)

  def apply(params):
    pi_s, _, logit_s = actor.apply(params, obs, legal)
    return pi_s, logit_s
  # Only the snapshots of this player's pool, on this player's infosets.
  snap_pi, snap_logit = jax.vmap(apply)(jax.tree.map(lambda x: x[player], pool["snap_params"]))  # [S, T, B, A]
  a_idx, b_idx = pool["base_snaps"][player, :, 0], pool["base_snaps"][player, :, 1]
  base_pi = _base_from_outputs(snap_pi[a_idx], snap_logit[a_idx], snap_pi[b_idx], snap_logit[b_idx],
                               pool["base_kind"][player], legal[None], vertex_space)  # [Bb, T, B, A]
  base_prob = jnp.where(valid[None], jnp.sum(action_oh[None] * base_pi, axis=-1), 1.0)  # [Bb, T, B]
  log_prob = jnp.log(base_prob)
  log_reach = jnp.concatenate([jnp.zeros_like(log_prob[:, :1]), jnp.cumsum(log_prob, axis=1)[:, :-1]], axis=1)

  weights = pool["weights"][player]  # [C, Bb]
  log_w = jnp.where(weights > 0, jnp.log(jnp.where(weights > 0, weights, 1.0)), -jnp.inf)[:, :, None, None]
  logits = log_w + log_reach[None]  # [C, Bb, T, B]
  reached = jnp.any(jnp.isfinite(logits), axis=1, keepdims=True)
  logits = jnp.where(reached, logits, jnp.broadcast_to(log_w, logits.shape))
  occupied = pool["slot_mask"][player] > 0
  logits = jnp.where(occupied[:, None, None, None], logits, 0.0)
  posterior = jax.nn.softmax(logits, axis=1)
  member = jnp.sum(posterior * base_prob[None], axis=1)  # [C, T, B]
  member = jnp.where(occupied[:, None, None], member, sampling[None])
  probs = jnp.concatenate([blueprint[..., None], jnp.moveaxis(member, 0, -1)], axis=-1)
  return probs / sampling[..., None]


def pool_output_mask(pool: dict, value_type: str):
  """[num outputs] mask of the pool value outputs of occupied slots (blueprints always)."""
  masks = [jnp.concatenate([jnp.ones(1), pool["slot_mask"][p]]) for p in range(2)]
  if value_type == MATRIX_VALUED:
    return (masks[0][:, None] * masks[1][None, :]).reshape(-1)
  return jnp.concatenate([jnp.ones(1), masks[1][1:], masks[0][1:]])


# ------------------------------------------------------------------------------
# Selection at evaluation
# ------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SelectionConfig:
  """The `eval.hullcover` config section: the portfolio size and the MILP settings."""
  k: Optional[int] = None
  milp_time_limit: float = 120.0
  milp_gap: float = 0.01


@dataclasses.dataclass
class Selection:
  """k mixtures over the pool of each player, lam[p] [k, |C_p|]."""
  config: SelectionConfig
  step: int
  cap: int
  lam: list
  info: list

  @property
  def k(self) -> int:
    return self.config.k

  def combination(self, player: int) -> np.ndarray:
    """[k + 1, C_max + 1]: the blueprint and the k mixtures over the slots."""
    lam = self.lam[player]
    combination = np.zeros((self.k + 1, self.cap + 1))
    combination[0, 0] = 1.0
    combination[1:, 1:lam.shape[1] + 1] = lam
    return combination


def select_portfolio(state: HullCoverState, config: SelectionConfig, step: int) -> Selection:
  if config.k is None or config.k < 1:
    raise ValueError("HullCover runs need the portfolio size eval.hullcover.k >= 1.")
  if any(instance is None for instance in state.instances):
    raise ValueError(f"The HullCover pools are not initialised yet at step {step} (the first round is at step "
                     f"{state.next_round_step()}).")
  lam, info = [], []
  for player, instance in enumerate(state.instances):
    start = time.time()
    player_lam, player_info = instance.select(config.k, config.milp_time_limit, config.milp_gap)
    lam.append(player_lam)
    info.append({**player_info, "seconds": time.time() - start, "pool_size": len(instance.C),
                 "tests": len(instance.X), "counters": dict(instance.counters),
                 "members": [{state.bank.describe(b): w for b, w in m.items()} for m in instance.C]})
  return Selection(config=config, step=step, cap=state.config.cap_c, lam=lam, info=info)


def load_or_select(solver, directory: str, step: int, config: SelectionConfig) -> Selection:
  """The selection of the checkpoint, cached in <directory>/hullcover/step_{step}_k{k}.pkl."""
  path = os.path.join(directory, "hullcover", f"step_{step}_k{config.k}.pkl")
  if os.path.exists(path):
    with open(path, "rb") as f:
      selection = pickle.load(f)
    if selection.config == config:
      return selection
  selection = select_portfolio(solver.hullcover_state, config, step)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    pickle.dump(selection, f)
  with open(os.path.splitext(path)[0] + ".json", "w") as f:
    json.dump({"step": step, "config": dataclasses.asdict(config),
               "players": [{**i, "lam": l.tolist()} for i, l in zip(selection.info, selection.lam)]}, f, indent=2)
  print(f"HullCover selection k={config.k} at step {step}: "
        + ", ".join(f"P{p + 1} eps {i['eps']:.4f} ({i['status']}, |C| {i['pool_size']})"
                    for p, i in enumerate(selection.info)), flush=True)
  return selection


def combine(values: np.ndarray, selection: Selection, value_type: str) -> np.ndarray:
  """Leaf values of the selected portfolio from the pool values [H, outputs]: [H, k+1, k+1] for
  matrix valued and [H, Pl, k+1] for multi valued states."""
  cap = selection.cap
  first, second = selection.combination(0), selection.combination(1)
  if value_type == MATRIX_VALUED:
    matrix = values.reshape(values.shape[0], cap + 1, cap + 1)
    return np.einsum("ki,hij,lj->hkl", first, matrix, second)
  p1_view = np.concatenate([values[:, :1], values[:, 1:cap + 1]], axis=-1) @ second.T
  p2_view = np.concatenate([values[:, :1], values[:, cap + 1:]], axis=-1) @ first.T
  return np.stack([p1_view, p2_view], axis=1)


def matrix_to_outputs(matrix: np.ndarray, value_type: str) -> np.ndarray:
  """[H, C+1, C+1] values of all pairs -> [H, outputs] in the layout of the value type."""
  if value_type == MATRIX_VALUED:
    return matrix.reshape(matrix.shape[0], -1)
  return np.concatenate([matrix[:, 0, :1], matrix[:, 0, 1:], matrix[:, 1:, 0]], axis=-1)


# ------------------------------------------------------------------------------
# Exact values (small games)
# ------------------------------------------------------------------------------

class ExactPool:
  """Exact behavioral policies of the pool members, from the full game tree."""

  def __init__(self, solver):
    state = solver.hullcover_state
    self.solver = solver
    self.cap = state.config.cap_c
    base_policy = jax.jit(make_base_policy(solver.actor, state.config.vertex_space))
    layers = build_full_tree(solver.game)
    self.tables = []
    for player, pool in enumerate(state.pools()):
      bases = sorted({b for member in pool for b in member})
      component_policies = []
      for b in bases:
        kind, a, s = state.bank.bases[b]
        params_a, params_b = state.bank.snapshots[a], state.bank.snapshots[s]

        def policy_fn(infosets, legal, params_a=params_a, params_b=params_b, kind=kind):
          return np.asarray(base_policy(params_a, params_b, kind, jnp.asarray(infosets, jnp.float32),
                                        jnp.asarray(legal, jnp.float32)))
        component_policies.append([p[player] for p in history_policies_from_fn(layers, policy_fn)])
      per_layer = [np.stack([c[d] for c in component_policies]) for d in range(len(layers))]
      weights = np.array([[member.get(b, 0.0) for b in bases] for member in pool])
      self.tables.append(mixture_tables(layers, per_layer, weights, player))

  def portfolio_policies(self, player: int, infosets: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """[N, A, C_max + 1]: the blueprint and the pool slots (empty slots repeat the blueprint)."""
    blueprint = self.solver.policy(infosets, legal)
    policies = np.repeat(blueprint[..., None], self.cap + 1, axis=-1)
    for slot, table in enumerate(self.tables[player]):
      for n, infoset in enumerate(infosets):
        probs = table.get(infoset_key(infoset))
        if probs is not None:
          policies[n, :, slot + 1] = probs
    return policies

  def leaf_values(self, selection: Selection, value_type: str) -> LeafValues:
    exact = make_exact_leaf_values(self.solver.game, self.portfolio_policies, self.cap + 1, MATRIX_VALUED)

    def leaf_value_fn(states, state_tensors, game_legal):
      return combine(matrix_to_outputs(exact.fn(states, state_tensors, game_legal), value_type), selection, value_type)
    return LeafValues(fn=leaf_value_fn, value_type=value_type, num_options=selection.k + 1)
