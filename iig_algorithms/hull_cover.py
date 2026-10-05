"""HullCover: online construction of a portfolio candidate pool with a capped value matrix.

Port of `find_portfolio_hullcover_efg` (HullCover_orig, portfolia-master notes/18). One instance
builds the portfolio of a single player, the "candidate" side, against test strategies of the
other player:
  C  candidates of the covered player (at most `cap_c`), the pool the portfolio is selected from,
  X  test strategies of the other player (at most `cap_x`),
  M  [|X|, |C|] value matrix in the LOSS of the covered player (the P1 value when covering P2,
     its negation when covering P1), so the candidate side always minimises.
The dominance quasi-metric is d(a, b) = max_i (a_i - b_i)_+ over the rows of X: how far a fails
to dominate b.

Strategies are members: realization-plan mixtures given as sparse weights {base id: weight} over
opaque base strategies. Mixing two members mixes their weights, and the value matrix is bilinear in
them, so merged columns and rows are exact without new value queries. The only game access is the
batched value oracle `value_fn(tests, candidates) -> [len(tests), len(candidates)]`.

Each round of the online phase offers candidates (`offer_candidate`) and tests (`offer_test`):
  - A candidate is kept iff no mixture of C eps-dominates it over X (`excess` > tol). It is
    appended while C has room, otherwise folded into one of the `partners` incumbents nearest in
    the symmetrised d_X, with the merge weight alpha from a grid (alpha = 1 replaces), choosing the
    pair minimising max(excess(C', v), excess(C', C_j)).
  - A test is kept iff it widens some pair gap (row_a - row_b)_+ beyond d_X(a, b). It is appended
    while X has room, otherwise mixed into an incumbent (`row_op` "mix", scored like the candidate
    merge) or replaces one (`row_op` "replace" with `evict_rule` "outclassed" or "redundant").
At the end, `select` picks k mixtures over C with the eps-Dom-Mixed-MILP on M.
"""
from __future__ import annotations

import collections
import dataclasses
from typing import Callable, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

Member = dict  # {base id: weight}, weights sum to 1
# (tests, candidates) -> [len(tests), len(candidates)] loss of the covered player
ValueFn = Callable[[Sequence[Member], Sequence[Member]], np.ndarray]

ROW_OPS = ("mix", "replace")
EVICT_RULES = ("outclassed", "redundant")
VERTEX_SPACES = ("logit", "policy")
VERTEX_WINDOWS = ("step", "round")


@dataclasses.dataclass(frozen=True)
class HullCoverConfig:
  """Hyperparameters of HullCover (the `train.hullcover` config section)."""
  # Learner steps between two harvests of the RNaD stream (T_step).
  every: int = 100
  cap_c: int = 30
  cap_x: int = 30
  # Merge weights of a newcomer folded into an incumbent of a full set, 1 replaces it.
  alphas: Sequence[float] = (0.25, 0.5, 0.75, 1.0)
  # Incumbents scored for a merge, the nearest ones to the newcomer.
  partners: int = 4
  tol: float = 1e-9
  # A newcomer of a full X is mixed into an incumbent or replaces one (by `evict_rule`).
  row_op: str = "mix"
  evict_rule: str = "outclassed"
  # Vertex of a player: per infoset the argmax of the change of the logits (or of the policy)
  # over the last learner step, i.e. of the gradient step ("step"), or since the previous
  # harvest ("round").
  vertex_space: str = "logit"
  vertex_window: str = "step"
  # Games per pair of base strategies in the Monte Carlo estimates of M.
  eval_games: int = 512
  # Mixture components below this weight are dropped after a merge.
  min_component_weight: float = 1e-3

  def __post_init__(self):
    if self.row_op not in ROW_OPS:
      raise ValueError(f"Unknown hullcover.row_op {self.row_op}, expected one of {ROW_OPS}.")
    if self.evict_rule not in EVICT_RULES:
      raise ValueError(f"Unknown hullcover.evict_rule {self.evict_rule}, expected one of {EVICT_RULES}.")
    if self.vertex_space not in VERTEX_SPACES:
      raise ValueError(f"Unknown hullcover.vertex_space {self.vertex_space}, expected one of {VERTEX_SPACES}.")
    if self.vertex_window not in VERTEX_WINDOWS:
      raise ValueError(f"Unknown hullcover.vertex_window {self.vertex_window}, expected one of {VERTEX_WINDOWS}.")
    if not self.alphas or any(not 0 < a <= 1 for a in self.alphas):
      raise ValueError(f"hullcover.alphas must be in (0, 1], got {self.alphas}.")


def mix_members(a: Member, b: Member, alpha: float) -> Member:
  """The realization-plan mixture alpha * a + (1 - alpha) * b."""
  mixed = collections.defaultdict(float)
  for base, weight in a.items():
    mixed[base] += alpha * weight
  for base, weight in b.items():
    mixed[base] += (1 - alpha) * weight
  return {base: weight for base, weight in mixed.items() if weight > 0}


def prune_member(member: Member, min_weight: float) -> Member:
  """Drops the components below `min_weight` (keeping the largest one) and renormalises."""
  kept = {base: w for base, w in member.items() if w >= min_weight}
  if not kept:
    base = max(member, key=member.get)
    kept = {base: member[base]}
  total = sum(kept.values())
  return {base: w / total for base, w in kept.items()}


def excess(M: np.ndarray, w: np.ndarray) -> float:
  """min_{lam in simplex} max_i ((M lam)_i - w_i)_+: the smallest eps with which a mixture of the
  columns of M eps-dominates w, over the rows. One LP with |C| + 1 variables and |X| rows."""
  n_x, n_c = M.shape
  if n_c == 0:
    return np.inf
  c_obj = np.zeros(n_c + 1)
  c_obj[-1] = 1.0
  result = linprog(c_obj, A_ub=np.hstack([M, -np.ones((n_x, 1))]), b_ub=np.asarray(w, float),
                   A_eq=np.concatenate([np.ones(n_c), [0.0]])[None, :], b_eq=[1.0],
                   bounds=[(0, None)] * n_c + [(None, None)], method="highs")
  return max(float(result.fun), 0.0) if result.success else np.inf


def pair_gaps(row: np.ndarray) -> np.ndarray:
  """d_x(a, b) = (row_a - row_b)_+ for one test strategy's row of values."""
  return np.maximum(row[:, None] - row[None, :], 0.0)


def coverage_radius(V: np.ndarray, lam: np.ndarray) -> float:
  """The eps with which the k mixtures `lam` [k, |C|] dominate every column of V [|X|, |C|]."""
  centers = V @ lam.T
  per_target = np.maximum(centers[:, :, None] - V[:, None, :], 0.0).max(axis=0)
  return float(per_target.min(axis=0).max())


def kcenter_select(points: np.ndarray, k: int) -> list[int]:
  """Farthest-first traversal in the dominance quasi-metric over the points [n, |X|]."""
  n = len(points)
  gaps = np.maximum(points[:, None, :] - points[None, :, :], 0.0).max(axis=-1)  # [from, to]
  chosen = [int(np.argmin(gaps.max(axis=1)))]
  cover = gaps[chosen[0]].copy()
  while len(chosen) < min(k, n):
    nxt = int(np.argmax(cover))
    if nxt in chosen:
      break
    chosen.append(nxt)
    cover = np.minimum(cover, gaps[nxt])
  for j in range(n):
    if len(chosen) >= k:
      break
    if j not in chosen:
      chosen.append(j)
  return chosen


def eps_dom_milp(V: np.ndarray, k: int, time_limit: float = 120.0, gap: float = 0.01) -> tuple[np.ndarray, dict]:
  """eps-Dom-Mixed-MILP restricted to the pool: k mixtures lam [k, |C|] over the columns of
  V [|X|, |C|] minimising the eps by which every column is dominated by its assigned mixture.
  The model of `_milp_select_on_pool` (HullCover_orig) with the symmetry breaking d[0, 0] = 1,
  solved by HiGHS. Falls back to farthest-first pool members if no solution is found."""
  m, n = V.shape
  big_m = float(V.max() - V.min())
  num_lam = k * n
  num_vars = 2 * num_lam + 1
  lam_idx = np.arange(num_lam).reshape(k, n)
  d_idx = num_lam + lam_idx
  eps_idx = 2 * num_lam

  # Every column assigned to exactly one mixture, every mixture a distribution.
  assign = sparse.csr_matrix((np.ones(num_lam), (np.tile(np.arange(n), k), d_idx.ravel())), shape=(n, num_vars))
  simplex = sparse.csr_matrix((np.ones(num_lam), (np.repeat(np.arange(k), n), lam_idx.ravel())), shape=(k, num_vars))
  # lam_z @ V[i] + big_m d[z, j] - eps <= V[i, j] + big_m for all z, j, i.
  z, j, i = np.meshgrid(np.arange(k), np.arange(n), np.arange(m), indexing="ij")
  z, j, i = z.ravel(), j.ravel(), i.ravel()
  rows = np.arange(z.size)
  lam_rows = np.repeat(rows, n)
  lam_cols = lam_idx[z].ravel()
  lam_vals = V[i].ravel()
  dominate = sparse.csr_matrix(
      (np.concatenate([lam_vals, np.full(rows.size, big_m), -np.ones(rows.size)]),
       (np.concatenate([lam_rows, rows, rows]), np.concatenate([lam_cols, d_idx[z, j], np.full(rows.size, eps_idx)]))),
      shape=(rows.size, num_vars))
  constraints = [LinearConstraint(assign, 1, 1), LinearConstraint(simplex, 1, 1),
                 LinearConstraint(dominate, -np.inf, V[i, j] + big_m)]

  lower = np.zeros(num_vars)
  upper = np.concatenate([np.ones(2 * num_lam), [np.inf]])
  lower[d_idx[0, 0]] = 1.0
  integrality = np.zeros(num_vars)
  integrality[num_lam:2 * num_lam] = 1
  objective = np.zeros(num_vars)
  objective[eps_idx] = 1.0
  result = milp(objective, constraints=constraints, integrality=integrality, bounds=Bounds(lower, upper),
                options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False})

  if result.x is not None:
    lam = np.clip(result.x[:num_lam].reshape(k, n), 0, None)
    lam /= lam.sum(axis=1, keepdims=True)
    status = "optimal" if result.status == 0 else result.message
  else:
    lam = np.zeros((k, n))
    lam[np.arange(k), kcenter_select(V.T, k)] = 1.0
    status = f"fallback farthest-first ({result.message})"
  return lam, {"eps": coverage_radius(V, lam), "status": status}


class HullCover:
  """One HullCover instance: candidates C, tests X and the value matrix M between them."""

  def __init__(self, config: HullCoverConfig, value_fn: ValueFn):
    self.config = config
    self.value_fn = value_fn
    self.C: list[Member] = []
    self.X: list[Member] = []
    self.M = np.zeros((0, 0))
    self.counters = collections.Counter()

  def __getstate__(self):
    # The value oracle is rebound by the owner after unpickling.
    return {k: v for k, v in self.__dict__.items() if k != "value_fn"}

  def __setstate__(self, state):
    self.__dict__.update(state)
    self.value_fn = None

  def _values(self, tests: Sequence[Member], candidates: Sequence[Member]) -> np.ndarray:
    self.counters["value_queries"] += len(tests) * len(candidates)
    return np.asarray(self.value_fn(tests, candidates), dtype=np.float64).reshape(len(tests), len(candidates))

  def _excess(self, M: np.ndarray, w: np.ndarray) -> float:
    self.counters["lp_solves"] += 1
    return excess(M, w)

  def referenced_bases(self) -> set:
    return {base for member in self.C + self.X for base in member}

  def init(self, candidates: Sequence[Member], test: Member):
    """Step 1: C = candidates, X = {test}."""
    self.C = [dict(c) for c in candidates]
    self.X = [dict(test)]
    self.M = self._values(self.X, self.C)

  # -- steps 2a-2c: candidates ------------------------------------------------

  def offer_candidate(self, v: Member) -> str:
    config = self.config
    col = self._values(self.X, [v])[:, 0]
    if self._excess(self.M, col) <= config.tol:  # already dominated by conv(C)
      self.counters["col_reject"] += 1
      return "reject"
    if len(self.C) < config.cap_c:
      self.C.append(dict(v))
      self.M = np.hstack([self.M, col[:, None]])
      self.counters["col_append"] += 1
      return "append"
    # Fold into the incumbent that keeps the frontier furthest out, scored by what the merge
    # fails to keep; alpha = 1 is pure replacement and competes under the same score.
    M = self.M
    d_to = np.maximum(M - col[:, None], 0.0).max(axis=0)
    d_from = np.maximum(col[:, None] - M, 0.0).max(axis=0)
    order = np.argsort(d_to + d_from)[:max(1, config.partners)]
    best_j, best_a, best_s = int(order[0]), 1.0, np.inf
    for j in order:
      j = int(j)
      for alpha in config.alphas:
        Mp = M.copy()
        Mp[:, j] = alpha * col + (1 - alpha) * M[:, j]
        s = max(self._excess(Mp, col), self._excess(Mp, M[:, j]))
        if s < best_s:
          best_j, best_a, best_s = j, alpha, s
    merged = mix_members(v, self.C[best_j], best_a)
    pruned = prune_member(merged, config.min_component_weight)
    self.C[best_j] = pruned
    if pruned == merged:  # bilinear in the plans, so the merged column needs no value query
      self.M[:, best_j] = best_a * col + (1 - best_a) * M[:, best_j]
    else:
      self.M[:, best_j] = self._values(self.X, [pruned])[:, 0]
    outcome = "replace" if best_a == 1.0 else "merge"
    self.counters[f"col_{outcome}"] += 1
    return outcome

  # -- steps 2d-2e: tests -------------------------------------------------------

  def offer_test(self, x: Member) -> str:
    config = self.config
    row = self._values([x], self.C)[0]
    d_new = pair_gaps(row)
    gaps = np.stack([pair_gaps(r) for r in self.M]) if len(self.X) else np.zeros((1,) + d_new.shape)
    d_cur = gaps.max(axis=0)
    if (d_new - d_cur).max() <= config.tol:  # reveals no new gap
      self.counters["row_reject"] += 1
      return "reject"
    if len(self.X) < config.cap_x:
      self.X.append(dict(x))
      self.M = np.vstack([self.M, row[None, :]])
      self.counters["row_append"] += 1
      return "append"
    if config.row_op == "mix":
      # Fold into the incumbent that loses the least pair gap reach, exact by bilinearity.
      d_full = np.maximum(d_cur, d_new)
      order = np.argsort(np.linalg.norm(self.M - row[None, :], axis=1))[:max(1, config.partners)]
      best_i, best_a, best_s = int(order[0]), 1.0, np.inf
      for i in order:
        i = int(i)
        d_wo = np.delete(gaps, i, axis=0).max(axis=0) if len(gaps) > 1 else np.zeros_like(d_cur)
        for alpha in config.alphas:
          r_mix = alpha * row + (1 - alpha) * self.M[i]
          loss = float((d_full - np.maximum(d_wo, pair_gaps(r_mix))).max())
          if loss < best_s:
            best_i, best_a, best_s = i, alpha, loss
      merged = mix_members(x, self.X[best_i], best_a)
      pruned = prune_member(merged, config.min_component_weight)
      self.X[best_i] = pruned
      if pruned == merged:
        self.M[best_i, :] = best_a * row + (1 - best_a) * self.M[best_i, :]
      else:
        self.M[best_i, :] = self._values([pruned], self.C)[0]
      outcome = "replace" if best_a == 1.0 else "mix"
      self.counters[f"row_{outcome}"] += 1
      return outcome
    if config.evict_rule == "outclassed":
      # The incumbent the newcomer most outclasses.
      i = int(np.argmax([float((d_new - g).max()) for g in gaps]))
    else:
      # The incumbent whose removal costs the least.
      worth = [float((d_cur - (np.delete(gaps, t, axis=0).max(axis=0) if len(gaps) > 1 else 0.0)).max())
               for t in range(len(self.X))]
      i = int(np.argmin(worth))
    self.X[i] = dict(x)
    self.M[i, :] = row
    self.counters["row_replace"] += 1
    return "replace"

  # -- step 3 -----------------------------------------------------------------------

  def select(self, k: int, time_limit: float = 120.0, gap: float = 0.01) -> tuple[np.ndarray, dict]:
    """k mixtures lam [k, |C|] over C by the eps-Dom-Mixed-MILP on M. With |C| <= k every member
    is taken, the last one repeated."""
    n = len(self.C)
    if n <= k:
      lam = np.eye(n)[np.minimum(np.arange(k), n - 1)]
      return lam, {"eps": coverage_radius(self.M, lam), "status": "pool not larger than k"}
    return eps_dom_milp(self.M, k, time_limit, gap)
