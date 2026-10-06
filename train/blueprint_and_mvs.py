"""Simultaneous move R-NaD with portfolio transformations and matrix valued states.

The policy learning is the simultaneous move variant of R-NaD
(https://arxiv.org/pdf/2510.05048, ported from `sim_rnad.py`) trained
on-policy on batched rollouts of a JAX game. On top of it, the portfolio of each
player is built by one of the methods (`portfolio_method`):
  - "gct", transformations: for each player, a network producing K policy
    deviation directions, fitted to the directions of the R-NaD policy updates.
    The portfolio of a player is the policy itself and its K transformations.
  - "hullcover": candidate pools maintained online from the R-NaD stream
    (`train/hullcover.py`), the portfolio is selected from them at evaluation.
  - Matrix valued states (MVS): a network on the state tensor estimating
    the P1 value of each pair of portfolio policies, (K + 1)^2 outputs with
    index i * (K + 1) + j for P1 portfolio policy i and P2 portfolio policy j
    (for hullcover the pairs of [blueprint + pool slots], K = cap_c).
These are used as the blueprint and the depth-limit leaf values in resolving.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Optional, Sequence, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

from games import make_game
from iig_algorithms.exploitability import make_exact_leaf_values
from iig_algorithms.hull_cover import HullCoverConfig
from iig_algorithms.tree_builder import MATRIX_VALUED, MULTI_VALUED, VALUE_TYPES, LeafValues
from train import hullcover

Params = chex.ArrayTree


"""BEGINNING OF CODE FROM OpenSpiel RNaD"""
class EntropySchedule:
  """An increasing list of steps where the regularisation network is updated.

  Example
    EntropySchedule([3, 5, 10], [2, 4, 1])
    =>   [0, 3, 6, 11, 16, 21, 26, 36]
          | 3 x2 |      5 x4     | 10 x1
  """

  def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
    try:
      if len(repeats) != len(sizes):
        raise ValueError("`repeats` must be parallel to `sizes`.")
      if not sizes:
        raise ValueError("`sizes` and `repeats` must not be empty.")
      if any([(repeat <= 0) for repeat in repeats]):
        raise ValueError("All repeat values must be strictly positive")
      if repeats[-1] != 1:
        raise ValueError("The last value in `repeats` must be equal to 1, "
                         "ince the last iteration size is repeated forever.")
    except ValueError as e:
      raise ValueError(
          f"Entropy iteration schedule: repeats ({repeats}) and sizes"
          f" ({sizes})."
      ) from e

    schedule = [0]
    for size, repeat in zip(sizes, repeats):
      schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

    self.schedule = np.array(schedule, dtype=np.int32)

  def __call__(self, learner_step: int) -> Tuple[float, bool]:
    """Returns the mixing weight alpha of the previous regularisation policies
    and whether the regularisation policies should be updated."""
    last_size = self.schedule[-1] - self.schedule[-2]
    last_start = self.schedule[-1] + (
        learner_step - self.schedule[-1]) // last_size * last_size
    start = jnp.amax(self.schedule * (self.schedule <= learner_step))
    finish = jnp.amin(
        self.schedule * (learner_step < self.schedule),
        initial=self.schedule[-1],
        where=(learner_step < self.schedule))
    size = finish - start

    beyond = (self.schedule[-1] <= learner_step)
    iteration_start = (last_start * beyond + start * (1 - beyond))
    iteration_size = (last_size * beyond + size * (1 - beyond))

    update_target_net = jnp.logical_and(
        learner_step > 0,
        jnp.sum(learner_step == iteration_start + iteration_size - 1),
    )
    alpha = jnp.minimum(
        (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0)
    return alpha, update_target_net


def _legal_policy(logits: chex.Array, legal_actions: chex.Array) -> chex.Array:
  """A soft-max policy that respects legal_actions."""
  chex.assert_equal_shape((logits, legal_actions))
  l_min = logits.min(axis=-1, keepdims=True)
  logits = jnp.where(legal_actions, logits, l_min)
  logits -= logits.max(axis=-1, keepdims=True)
  logits *= legal_actions
  exp_logits = jnp.where(legal_actions, jnp.exp(logits), 0)
  exp_logits_sum = jnp.sum(exp_logits, axis=-1, keepdims=True)
  return exp_logits / exp_logits_sum


def legal_log_policy(logits: chex.Array, legal_actions: chex.Array) -> chex.Array:
  """Return the log of the policy on legal action, 0 on illegal action."""
  chex.assert_equal_shape((logits, legal_actions))
  logits_masked = logits + jnp.log(legal_actions)
  max_legal_logit = logits_masked.max(axis=-1, keepdims=True)
  logits_masked = logits_masked - max_legal_logit
  exp_logits_masked = jnp.exp(logits_masked)
  baseline = jnp.log(jnp.sum(exp_logits_masked, axis=-1, keepdims=True))
  log_policy = jnp.multiply(legal_actions, (logits - max_legal_logit - baseline))
  return log_policy


def apply_force_with_threshold(decision_outputs: chex.Array, force: chex.Array,
                               threshold: float,
                               threshold_center: chex.Array) -> chex.Array:
  """Apply the force with below a given threshold."""
  chex.assert_equal_shape((decision_outputs, force, threshold_center))
  can_decrease = decision_outputs - threshold_center > -threshold
  can_increase = decision_outputs - threshold_center < threshold
  force_negative = jnp.minimum(force, 0.0)
  force_positive = jnp.maximum(force, 0.0)
  clipped_force = can_decrease * force_negative + can_increase * force_positive
  return decision_outputs * lax.stop_gradient(clipped_force)
"""END OF CODE FROM OpenSpiel RNaD"""


def tree_where(pred: chex.Array, true_data: chex.ArrayTree, false_data: chex.ArrayTree) -> chex.ArrayTree:
  """jnp.where with `pred` being a broadcastable prefix of the leaves."""
  def _where_one(t, f):
    p = jnp.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape)))
    return jnp.where(p, t, f)
  return jax.tree.map(_where_one, true_data, false_data)


def masked_mean(x: chex.Array, mask: chex.Array, normalization_mult: float = 1.0) -> chex.Array:
  """Mean of x over the entries where the (broadcastable) mask is set."""
  normalization = jnp.sum(mask) * normalization_mult
  return jnp.sum(x * mask) / (normalization + (normalization == 0.0))


def policy_ratio(pi: chex.Array, mu: chex.Array, action_oh: chex.Array, valid: chex.Array) -> chex.Array:
  """pi(a) / mu(a) of the played action, 1 on non valid states. Keeps the action dimension."""
  def _select(policy):
    return jnp.sum(action_oh * policy, axis=-1, keepdims=True) * valid + (1 - valid)
  return _select(pi) / _select(mu)


def normalize_direction_with_mask(x: chex.Array, mask: chex.Array) -> chex.Array:
  """Normalizes directions [..., A, K] over the masked action dimension."""
  chex.assert_shape((mask,), x.shape[:-1])
  x = mask[..., jnp.newaxis] * x
  norm = jnp.linalg.norm(x, 2, -2, keepdims=True)
  return jnp.where(norm < 1e-15, x, x / jnp.where(norm < 1e-15, 1.0, norm))


def transform_policies(pi: chex.Array, transformations: chex.Array, legal: chex.Array) -> chex.Array:
  """Portfolio policies [..., A, K] given policy [..., A] and directions [..., A, K]."""
  transformed = jnp.maximum(pi[..., jnp.newaxis] + transformations, 1e-8) * legal[..., jnp.newaxis]
  return transformed / jnp.sum(transformed, axis=-2, keepdims=True)


def neurd_loss(logits: chex.Array, policy: chex.Array, q_values: chex.Array, legal: chex.Array,
               importance_sampling: chex.Array, clip: float = 10_000, threshold: float = 2.0):
  advantage = q_values - jnp.sum(policy * q_values, axis=-1, keepdims=True)
  advantage = advantage * importance_sampling
  advantage = lax.stop_gradient(jnp.clip(advantage, -clip, clip))
  mean_logit = jnp.sum(logits * legal, axis=-1, keepdims=True) / jnp.sum(legal, axis=-1, keepdims=True)
  logits_shifted = logits - mean_logit
  threshold_center = jnp.zeros_like(logits_shifted)
  return jnp.sum(legal * apply_force_with_threshold(logits_shifted, advantage, threshold, threshold_center),
                 axis=-1, keepdims=True)


def v_trace(
    v: chex.Array,                     # [T, B, 1] critic values (P1)
    valid: chex.Array,                 # [T, B, 1, 1]
    sampling_policy: chex.Array,       # [T, B, Pl, A]
    network_policy: chex.Array,        # [T, B, Pl, A]
    regularization_term: chex.Array,   # [T, B, Pl, A] log(pi / pi_reg)
    action_oh: chex.Array,             # [T, B, Pl, A]
    reward: chex.Array,                # [T, B] P1 reward, not regularized
    lambda_: float = 1.0,
    c: float = jnp.inf,
    rho: float = jnp.inf,
    eta: float = 0.2,
    gamma: float = 1.0):
  """Simultaneous move V-trace. Returns value targets [T, B, 1] and
  per player Q-value estimates [T, B, Pl, A] (each from its own perspective)."""
  importance_sampling = policy_ratio(network_policy, sampling_policy, action_oh, valid)
  inverted_sampling = policy_ratio(jnp.ones_like(sampling_policy), sampling_policy, action_oh, valid)

  # KL divergence from the regularisation policy, [T, B, Pl].
  regularization_entropy = eta * jnp.sum(network_policy * regularization_term, axis=-1)
  weighted_regularization_term = -eta * regularization_term
  # The value is higher if the opponent deviates from its regularisation
  # policy and lower if we do so.
  both_player_entropy = regularization_entropy[..., 1] - regularization_entropy[..., 0]
  entropy_reward = jnp.expand_dims(reward + both_player_entropy, -1)
  q_reward = jnp.stack((reward, -reward), axis=-1) + regularization_entropy[..., (1, 0)]
  q_reward = jnp.expand_dims(q_reward, -1)

  @chex.dataclass(frozen=True)
  class VTraceCarry:
    next_value: chex.Array
    delta_v: chex.Array

  init_carry = VTraceCarry(next_value=jnp.zeros_like(v[-1]), delta_v=jnp.zeros_like(v[-1]))

  def _v_trace(carry: VTraceCarry, x) -> tuple[VTraceCarry, Any]:
    (importance_sampling, v, q_reward, entropy_reward, weighted_regularization_term, valid,
     inverted_sampling, action_oh) = x
    # Both players act in each step, the joint importance sampling is used.
    rho_is = jnp.minimum(rho, importance_sampling)
    rho_joint_is = jnp.prod(rho_is, axis=-2)
    c_joint_is = jnp.prod(jnp.minimum(c, importance_sampling), axis=-2)
    rho_inv_is = jnp.minimum(rho, inverted_sampling)

    delta_v = rho_joint_is * (entropy_reward + gamma * carry.next_value - v)
    carry_delta_v = delta_v + lambda_ * c_joint_is * gamma * carry.delta_v
    v_target = v + carry_delta_v

    per_player_v = jnp.stack([v, -v], axis=-2)
    q_term = jnp.stack([(carry.next_value + carry.delta_v) - v, v - (carry.next_value + carry.delta_v)], axis=-2)
    opponent_sampling = jnp.flip(rho_is, -2)
    q_value = (per_player_v + weighted_regularization_term
               + action_oh * opponent_sampling * rho_inv_is * (q_reward + gamma * q_term))

    next_carry = VTraceCarry(next_value=v, delta_v=carry_delta_v)
    next_carry, v_target = tree_where(jnp.squeeze(valid, axis=-1), (next_carry, v_target),
                                      (init_carry, jnp.zeros_like(v_target)))
    q_value = jnp.where(valid, q_value, jnp.zeros_like(q_value))
    return next_carry, (v_target, q_value)

  _, (v_target, q_value) = lax.scan(
      f=_v_trace,
      init=init_carry,
      xs=(importance_sampling, v, q_reward, entropy_reward, weighted_regularization_term, valid,
          inverted_sampling, action_oh),
      reverse=True)
  return v_target, q_value


def mvs_v_trace(state_v: chex.Array,       # [T, B, M] target network values
                valid: chex.Array,         # [T, B]
                joint_ratio: chex.Array,   # [T, B, M] joint portfolio importance ratios
                reward: chex.Array,        # [T, B] P1 reward
                lambda_: float, c: float, rho: float, gamma: float = 1.0) -> chex.Array:
  """V-trace targets of all the portfolio pairs at once."""

  @chex.dataclass(frozen=True)
  class LoopStateVTraceCarry:
    next_state_value: chex.Array
    next_state_v_target: chex.Array

  init_carry = LoopStateVTraceCarry(next_state_value=jnp.zeros_like(state_v[-1]),
                                    next_state_v_target=jnp.zeros_like(state_v[-1]))

  def _loop(carry: LoopStateVTraceCarry, x):
    cs, valid, state_v, reward = x
    state_target = (state_v + jnp.minimum(rho, cs) * (reward[..., jnp.newaxis] + gamma * carry.next_state_value - state_v)
                    + lambda_ * jnp.minimum(c, cs) * gamma * (carry.next_state_v_target - carry.next_state_value))
    new_carry = LoopStateVTraceCarry(next_state_value=state_v, next_state_v_target=state_target)
    return tree_where(valid, (new_carry, state_target), (init_carry, jnp.zeros_like(state_target)))

  _, state_v_target = lax.scan(f=_loop, init=init_carry, xs=(joint_ratio, valid, state_v, reward), reverse=True)
  return state_v_target


class MLP(nn.Module):
  layers: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for size in self.layers:
      x = nn.relu(nn.Dense(size)(x))
    return x


class ActorNetwork(nn.Module):
  """Policy on the infoset tensor of a single player, shared by both players."""
  actions: int
  network_layers: Sequence[int]

  @nn.compact
  def __call__(self, obs: chex.Array, legal: chex.Array):
    logit = nn.Dense(self.actions)(MLP(self.network_layers)(obs))
    pi = _legal_policy(logit, legal)
    log_pi = legal_log_policy(logit, legal)
    return pi, log_pi, logit


class CriticNetwork(nn.Module):
  """Centralized scalar P1 value on the concatenated infosets of both players."""
  network_layers: Sequence[int]

  @nn.compact
  def __call__(self, joint_obs: chex.Array):
    return nn.Dense(1)(MLP(self.network_layers)(joint_obs))


class MultiValuedStatesNetwork(nn.Module):
  out_dims: int
  network_layers: Sequence[int]

  @nn.compact
  def __call__(self, state: chex.Array):
    return nn.Dense(self.out_dims)(MLP(self.network_layers)(state))


class TransformationsNetwork(nn.Module):
  actions: int
  transformations: int
  network_layers: Sequence[int]

  @nn.compact
  def __call__(self, obs: chex.Array):
    pi_deviation = nn.Dense(self.actions * self.transformations)(MLP(self.network_layers)(obs))
    return pi_deviation.reshape((*pi_deviation.shape[:-1], self.actions, self.transformations))


@dataclasses.dataclass(frozen=True)
class AdamConfig:
  b1: float = 0.0
  b2: float = 0.999
  eps: float = 10e-8


# Hyperparameters added after models were saved, with the values those models were trained with
# (their pickled configs and saved config.yaml files do not contain them).
LEGACY_DEFAULTS = {"mvs_rho_vtrace": None, "portfolio_method": "gct", "hullcover": None}

GCT = "gct"
HULLCOVER = "hullcover"
PORTFOLIO_METHODS = (GCT, HULLCOVER)


@dataclasses.dataclass(frozen=True)
class RNaDConfig:
  """Hyperparameters of the simultaneous move R-NaD with portfolios."""
  actor_network_layers: Sequence[int] = (256, 256)
  critic_network_layers: Sequence[int] = (256, 256)
  mvs_network_layers: Sequence[int] = (256, 256)
  transformation_network_layers: Sequence[int] = (256, 256)

  batch_size: int = 256
  learning_rate: float = 0.00005
  adam: AdamConfig = AdamConfig()
  clip_gradient: float = 10_000
  # The "speed" at which the target networks follow the online networks.
  target_network_avg: float = 0.001

  entropy_schedule_repeats: Sequence[int] = (1,)
  # Swapping the regularisation policy often enough keeps the policy from getting
  # stuck at the equilibrium of the regularized game.
  entropy_schedule_size: Sequence[int] = (1_000,)
  eta_reward_transform: float = 0.2
  neurd_beta: float = 2.0
  neurd_clip: float = 10_000
  lambda_vtrace: float = 1.0
  c_vtrace: float = 1.0
  rho_vtrace: float = np.inf
  gamma: float = 1.0
  # Clipping of the importance ratios of the portfolio policies in the value network targets,
  # rho_vtrace if None. Independent of rho_vtrace, which also clips 1 / mu in the NeuRD Q-values.
  mvs_rho_vtrace: Optional[float] = 1.0

  # K, the portfolio of each player has K + 1 policies (the identity first).
  num_transformations: int = 10
  # "multi_valued": a player plays its blueprint and the opponent any of its
  # K + 1 options, 1 + 2K outputs ordered as [both blueprints, P2 transformations,
  # P1 transformations]. "matrix_valued": all (K + 1)^2 pairs of options, index
  # i * (K + 1) + j for P1 option i and P2 option j.
  value_type: str = MULTI_VALUED
  seed: int = 42
  # "gct" (the transformations) or "hullcover" (the `hullcover` section is then required).
  portfolio_method: str = GCT
  hullcover: Optional[HullCoverConfig] = None


@chex.dataclass(frozen=True)
class TimeStep:
  """Batched on-policy trajectories, [T, B, ...]."""
  obs: chex.Array = ()       # [T, B, Pl, I] infoset tensors
  legal: chex.Array = ()     # [T, B, Pl, A]
  policy: chex.Array = ()    # [T, B, Pl, A] sampling policy
  action_oh: chex.Array = ()  # [T, B, Pl, A]
  reward: chex.Array = ()    # [T, B] P1 reward of the transition
  valid: chex.Array = ()     # [T, B] the state is not terminal
  state: chex.Array = ()     # [T, B, S] state tensors


@chex.dataclass
class TrainState:
  params: Params = ()               # {"actor", "critic"}
  critic_target: Params = ()
  prev_actor: Params = ()
  prev_actor_: Params = ()
  opt_state: Any = ()
  transformation_params: Any = ()   # list over players
  transformation_opt_state: Any = ()
  mvs_params: Params = ()
  mvs_target: Params = ()
  mvs_opt_state: Any = ()
  learner_step: chex.Array = ()
  key: chex.Array = ()


def _canonicalize(tree: chex.ArrayTree) -> chex.ArrayTree:
  """Strong typed jax arrays, so that the states can be carried in a scan."""
  return jax.tree.map(lambda x: jnp.array(x, dtype=jnp.asarray(x).dtype), tree)


# Inference functions with the networks as static arguments: networks are equal by their fields, so
# the solvers of all the checkpoints of a run share the compilations, and the cache of jit does not
# keep the solvers alive.

@functools.partial(jax.jit, static_argnums=(0,))
def _apply_policy(actor: ActorNetwork, actor_params, obs, legal):
  return actor.apply(actor_params, obs, legal)[0]


@functools.partial(jax.jit, static_argnums=(0,))
def _apply_mvs(mvs_network: MultiValuedStatesNetwork, mvs_params, states):
  return mvs_network.apply(mvs_params, states)


@functools.partial(jax.jit, static_argnums=(0, 1))
def _apply_portfolio(actor: ActorNetwork, transformation_network: TransformationsNetwork, actor_params,
                     transformation_params, obs, legal):
  pi, _, _ = actor.apply(actor_params, obs, legal)
  directions = transformation_network.apply(transformation_params, obs)
  directions = normalize_direction_with_mask(directions, legal)
  directions = jnp.concatenate([jnp.zeros_like(directions[..., :1]), directions], axis=-1)
  return transform_policies(pi, directions, legal)


class RNaDSolver:
  """Simultaneous move R-NaD with portfolio transformations and MVS."""

  def __init__(self, config: RNaDConfig, game_name: str, game_params: Optional[dict] = None):
    self.config = config
    self.game_name = game_name
    self.game_params = dict(game_params or {})
    self.init()

  def init(self):
    config = self.config
    self.game = make_game(self.game_name, **self.game_params)
    self.num_actions = self.game.num_distinct_actions()
    self.num_players = self.game.num_players()
    # The HullCover portfolio size is known once its selection is attached.
    self.num_portfolio = None if config.portfolio_method == HULLCOVER else config.num_transformations + 1
    # The terminal node is not trained on.
    self.trajectory_max = self.game.max_trajectory_length_no_chance() - 1

    self.actor = ActorNetwork(self.num_actions, config.actor_network_layers)
    self.critic = CriticNetwork(config.critic_network_layers)
    if config.value_type not in VALUE_TYPES:
      raise ValueError(f"Unknown value type {config.value_type}, expected one of {VALUE_TYPES}.")
    if config.portfolio_method not in PORTFOLIO_METHODS:
      raise ValueError(f"Unknown portfolio method {config.portfolio_method}, expected one of {PORTFOLIO_METHODS}.")
    if (config.portfolio_method == HULLCOVER) != (config.hullcover is not None):
      raise ValueError("The hullcover section is required by, and only allowed with, portfolio_method hullcover.")
    self.hullcover_state: Optional[hullcover.HullCoverState] = None
    self.selection: Optional[hullcover.Selection] = None
    self._pool = None
    self._vertex_reference = None
    # Options of the value network outputs: the portfolio, or [blueprint + pool slots] for hullcover.
    num_options = config.hullcover.cap_c + 1 if self.is_hullcover else self.num_portfolio
    num_values = (num_options ** 2 if config.value_type == MATRIX_VALUED else 1 + 2 * (num_options - 1))
    self.mvs_network = MultiValuedStatesNetwork(num_values, config.mvs_network_layers)
    self.transformation_network = TransformationsNetwork(self.num_actions, config.num_transformations,
                                                         config.transformation_network_layers)
    self._entropy_schedule = EntropySchedule(sizes=config.entropy_schedule_size,
                                             repeats=config.entropy_schedule_repeats)
    self.optimizer = optax.chain(
        optax.scale_by_adam(eps_root=0.0, **dataclasses.asdict(config.adam)),
        optax.scale(-config.learning_rate),
        optax.clip(config.clip_gradient))

    obs = jnp.zeros((self.game.information_state_tensor_shape(),))
    legal = jnp.ones((self.num_actions,))
    state = jnp.zeros((self.game.state_tensor_shape(),))
    key = jax.random.PRNGKey(config.seed)
    key, actor_key, critic_key, mvs_key, *t_keys = jax.random.split(key, 4 + self.num_players)
    params = {"actor": self.actor.init(actor_key, obs, legal),
              "critic": self.critic.init(critic_key, jnp.concatenate([obs] * self.num_players))}
    transformation_params = [self.transformation_network.init(k, obs) for k in t_keys]
    mvs_params = self.mvs_network.init(mvs_key, state)
    self.state = TrainState(
        params=params, critic_target=params["critic"],
        prev_actor=params["actor"], prev_actor_=params["actor"],
        opt_state=self.optimizer.init(params),
        transformation_params=transformation_params,
        transformation_opt_state=[self.optimizer.init(p) for p in transformation_params],
        mvs_params=mvs_params, mvs_target=mvs_params,
        mvs_opt_state=self.optimizer.init(mvs_params),
        learner_step=jnp.zeros((), dtype=jnp.int32), key=key)

  @property
  def learner_steps(self) -> int:
    return int(self.state.learner_step)

  @property
  def is_hullcover(self) -> bool:
    return self.config.portfolio_method == HULLCOVER

  # ----------------------------------------------------------------------------
  # Trajectory collection
  # ----------------------------------------------------------------------------

  def _play_chance(self, states, legal, key):
    """Samples a chance outcome in all the chance states of the batch."""
    game = self.game
    is_chance = jax.vmap(lambda s: jnp.asarray(game.is_chance(s), dtype=bool))(states)
    outcomes, probs = jax.vmap(game.get_outcomes_and_probs)(states)
    probs = jnp.where(is_chance[:, None], probs, jax.nn.one_hot(0, probs.shape[-1]))
    outcome_idx = jax.random.categorical(key, jnp.log(probs), axis=-1)
    actions = jnp.take_along_axis(outcomes, outcome_idx[:, None, None], axis=1)[:, 0].astype(jnp.int32)
    chance_states, _, _, chance_legal = jax.vmap(game.apply_action)(states, actions)
    states = tree_where(is_chance, chance_states, states)
    legal = jnp.where(is_chance[:, None, None], chance_legal.astype(jnp.float32), legal)
    return states, legal

  def _rollout(self, actor_params: Params, key: chex.PRNGKey) -> TimeStep:
    return self.rollout_with_policy(lambda obs, legal: self.actor.apply(actor_params, obs, legal)[0], key,
                                    self.config.batch_size)

  def rollout_with_policy(self, policy_fn, key: chex.PRNGKey, batch: int) -> TimeStep:
    """Batched trajectories from the initial state, both players acting by
    policy_fn(obs [B, Pl, I], legal [B, Pl, A]) -> policies [B, Pl, A]."""
    game = self.game
    init_state, init_legal = game.initialize_structures()
    states = jax.tree.map(lambda x: jnp.broadcast_to(jnp.asarray(x), (batch,) + jnp.shape(x)), init_state)
    legal = jnp.broadcast_to(jnp.asarray(init_legal, dtype=jnp.float32), (batch,) + jnp.shape(init_legal))
    key, chance_key = jax.random.split(key)
    states, legal = self._play_chance(states, legal, chance_key)
    states = _canonicalize(states)

    def step(carry, step_key):
      states, legal, done = carry
      action_key, chance_key = jax.random.split(step_key)
      state_tensor, p1_iset, p2_iset, _ = jax.vmap(game.get_info)(states)
      obs = jnp.stack([p1_iset, p2_iset], axis=1).astype(jnp.float32)
      # Guard against terminal states without legal actions.
      legal = jnp.where(jnp.sum(legal, -1, keepdims=True) > 0, legal, jax.nn.one_hot(0, self.num_actions))
      pi = policy_fn(obs, legal)
      actions = jax.random.categorical(action_key, jnp.log(pi), axis=-1).astype(jnp.int32)
      new_states, terminal, reward, new_legal = jax.vmap(game.apply_action)(states, actions)
      new_states, new_legal = self._play_chance(new_states, new_legal.astype(jnp.float32), chance_key)

      valid = ~done
      new_done = done | terminal
      new_states = jax.tree.map(lambda n, o: n.astype(o.dtype), new_states, states)
      new_states = tree_where(done, states, new_states)
      new_legal = jnp.where(done[:, None, None], legal, new_legal)
      timestep = TimeStep(obs=obs, legal=legal, policy=pi,
                          action_oh=jax.nn.one_hot(actions, self.num_actions),
                          reward=jnp.where(valid, reward, 0.0).astype(jnp.float32),
                          valid=valid.astype(jnp.float32), state=state_tensor.astype(jnp.float32))
      return (new_states, new_legal, new_done), timestep

    keys = jax.random.split(key, self.trajectory_max)
    done = jnp.zeros((batch,), dtype=bool)
    _, timesteps = lax.scan(step, (states, legal, done), keys)
    return timesteps

  # ----------------------------------------------------------------------------
  # Losses
  # ----------------------------------------------------------------------------

  def rnad_loss(self, params, critic_target, prev_actor, prev_actor_, ts: TimeStep, alpha):
    config = self.config
    pi, log_pi, logit = self.actor.apply(params["actor"], ts.obs, ts.legal)
    joint_obs = jnp.reshape(ts.obs, (*ts.obs.shape[:-2], -1))
    v = self.critic.apply(params["critic"], joint_obs)
    v_target = self.critic.apply(critic_target, joint_obs)
    _, log_pi_prev, _ = self.actor.apply(prev_actor, ts.obs, ts.legal)
    _, log_pi_prev_, _ = self.actor.apply(prev_actor_, ts.obs, ts.legal)
    regularization_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

    expanded_valid = ts.valid[..., None, None]
    v_train_target, q_value = v_trace(
        v_target, expanded_valid, ts.policy, pi, regularization_term, ts.action_oh, ts.reward,
        config.lambda_vtrace, config.c_vtrace, config.rho_vtrace, config.eta_reward_transform, config.gamma)

    v_loss = masked_mean((v - lax.stop_gradient(v_train_target)) ** 2, ts.valid[..., None])
    # On-policy, so no counterfactual importance sampling correction.
    loss_neurd = neurd_loss(logit, pi, q_value, ts.legal, 1.0, config.neurd_clip, config.neurd_beta)
    # Both players act in each step, hence the normalization multiplier.
    neurd_loss_value = -masked_mean(loss_neurd, expanded_valid, normalization_mult=self.num_players)
    return v_loss + neurd_loss_value, {"value_loss": v_loss, "policy_loss": neurd_loss_value}

  def transformation_loss(self, transformation_params, policy_before, policy_after, player: int, ts: TimeStep):
    """Fits the closest transformation direction (per trajectory) to the
    direction of the policy update of the given player."""
    obs, legal = ts.obs[:, :, player], ts.legal[:, :, player]
    direction = self.transformation_network.apply(transformation_params, obs)  # [T, B, A, K]
    update = (policy_after - policy_before)[:, :, player, :, None]
    normalized_direction = normalize_direction_with_mask(direction, legal)
    update = normalize_direction_with_mask(update, legal)

    # Only steps with an actual decision and policy change are relevant.
    step_mask = ts.valid * (jnp.sum(legal, -1) > 1) * (jnp.linalg.norm(update[..., 0], axis=-1) > 1e-12)
    distances = jnp.linalg.norm(normalized_direction - update, 2, -2)  # [T, B, K]
    distances = masked_mean_over_time(distances, step_mask)
    closest = jnp.argmin(distances, -1)
    train_mask = (jax.nn.one_hot(closest, self.config.num_transformations)[None, :, None, :]
                  * step_mask[..., None, None] * legal[..., None])
    loss = (direction - lax.stop_gradient(update)) ** 2
    return masked_mean(loss, train_mask)

  def mvs_loss(self, mvs_params, mvs_target, actor_params, transformation_params, ts: TimeStep, pool=None):
    pi, _, _ = self.actor.apply(actor_params, ts.obs, ts.legal)
    ratios = []
    for player in range(self.num_players):
      if self.is_hullcover:
        # The blueprint and the realization-plan mixtures of the pool slots.
        ratios.append(hullcover.pool_ratios(self.actor, self.config.hullcover.vertex_space, pool, pi, ts, player))
        continue
      legal = ts.legal[:, :, player]
      directions = self.transformation_network.apply(transformation_params[player], ts.obs[:, :, player])
      directions = normalize_direction_with_mask(directions, legal)
      directions = jnp.concatenate([jnp.zeros_like(directions[..., :1]), directions], axis=-1)
      portfolio = transform_policies(pi[:, :, player], directions, legal)  # [T, B, A, K + 1]
      action_oh = ts.action_oh[:, :, player]
      portfolio_prob = jnp.sum(action_oh[..., None] * portfolio, axis=-2)
      sampling_prob = jnp.sum(action_oh * ts.policy[:, :, player], axis=-1, keepdims=True)
      ratios.append(portfolio_prob / sampling_prob)
    if self.config.value_type == MATRIX_VALUED:
      joint_ratio = ratios[0][..., :, None] * ratios[1][..., None, :]
      joint_ratio = jnp.reshape(joint_ratio, (*joint_ratio.shape[:-2], -1))
    else:
      # [both blueprints, P1 blueprint x P2 transformations, P1 transformations x P2 blueprint]
      joint_ratio = jnp.concatenate([ratios[0][..., :1] * ratios[1], ratios[0][..., 1:] * ratios[1][..., :1]], axis=-1)
    joint_ratio = jnp.where(ts.valid[..., None] > 0, joint_ratio, 1.0)
    joint_ratio = lax.stop_gradient(joint_ratio)

    mvs_v = self.mvs_network.apply(mvs_params, ts.state)
    mvs_v_target = self.mvs_network.apply(mvs_target, ts.state)
    rho = self.config.rho_vtrace if self.config.mvs_rho_vtrace is None else self.config.mvs_rho_vtrace
    target = mvs_v_trace(mvs_v_target, ts.valid > 0, joint_ratio, ts.reward,
                         self.config.lambda_vtrace, self.config.c_vtrace, rho, self.config.gamma)
    squared_error = (mvs_v - lax.stop_gradient(target)) ** 2
    if self.is_hullcover:
      # Outputs of empty pool slots are not trained (summed over the outputs, as without the mask).
      squared_error = squared_error * hullcover.pool_output_mask(pool, self.config.value_type)
    return masked_mean(squared_error, ts.valid[..., None])

  # ----------------------------------------------------------------------------
  # Learner step
  # ----------------------------------------------------------------------------

  @functools.partial(jax.jit, static_argnums=(0,))
  def _learner_step(self, state: TrainState, pool=None):
    config = self.config
    key, rollout_key = jax.random.split(state.key)
    ts = self._rollout(state.params["actor"], rollout_key)
    alpha, update_regularization = self._entropy_schedule(state.learner_step)

    policy_before, _, _ = self.actor.apply(state.params["actor"], ts.obs, ts.legal)
    (_, logs), grads = jax.value_and_grad(self.rnad_loss, has_aux=True)(
        state.params, state.critic_target, state.prev_actor, state.prev_actor_, ts, alpha)
    updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    # Exponential moving average of the critic, i.e. the TD target network.
    critic_target = optax.incremental_update(params["critic"], state.critic_target, config.target_network_avg)
    prev_actor, prev_actor_ = lax.cond(
        update_regularization,
        lambda: (params["actor"], state.prev_actor),
        lambda: (state.prev_actor, state.prev_actor_))
    policy_after, _, _ = self.actor.apply(params["actor"], ts.obs, ts.legal)

    transformation_params, transformation_opt_state = state.transformation_params, state.transformation_opt_state
    if not self.is_hullcover:
      transformation_params, transformation_opt_state = [], []
      for player in range(self.num_players):
        t_loss, t_grad = jax.value_and_grad(self.transformation_loss)(
            state.transformation_params[player], policy_before, policy_after, player, ts)
        t_updates, t_opt = self.optimizer.update(t_grad, state.transformation_opt_state[player])
        transformation_params.append(optax.apply_updates(state.transformation_params[player], t_updates))
        transformation_opt_state.append(t_opt)
        logs[f"transformation_loss_p{player}"] = t_loss

    mvs_loss, mvs_grad = jax.value_and_grad(self.mvs_loss)(
        state.mvs_params, state.mvs_target, params["actor"], transformation_params, ts, pool)
    mvs_updates, mvs_opt_state = self.optimizer.update(mvs_grad, state.mvs_opt_state)
    mvs_params = optax.apply_updates(state.mvs_params, mvs_updates)
    mvs_target = optax.incremental_update(mvs_params, state.mvs_target, config.target_network_avg)
    logs["mvs_loss"] = mvs_loss
    logs["average_return"] = jnp.sum(ts.reward) / config.batch_size

    new_state = TrainState(
        params=params, critic_target=critic_target, prev_actor=prev_actor, prev_actor_=prev_actor_,
        opt_state=opt_state, transformation_params=transformation_params,
        transformation_opt_state=transformation_opt_state, mvs_params=mvs_params,
        mvs_target=mvs_target, mvs_opt_state=mvs_opt_state,
        learner_step=state.learner_step + 1, key=key)
    return new_state, logs

  def step(self) -> dict:
    """One on-policy learner step: collect a batch and update all networks."""
    if self.is_hullcover:
      if self._pool is None:
        self._pool = self._pool_arrays()
      state = self.hullcover_state
      if (state is not None and self.config.hullcover.vertex_window == "step"
          and self.learner_steps + 1 == state.next_round_step()):
        # The step vertex of the next round is the change over this learner step.
        self._vertex_reference = jax.device_get(self.state.params["actor"])
      self.state, logs = self._learner_step(self.state, self._pool)
    else:
      self.state, logs = self._learner_step(self.state)
    logs = {k: float(v) for k, v in logs.items()}
    logs["learner_steps"] = self.learner_steps
    return logs

  # ----------------------------------------------------------------------------
  # HullCover
  # ----------------------------------------------------------------------------

  def _pool_arrays(self) -> dict:
    if self.hullcover_state is None:
      return hullcover.pool_arrays(hullcover.StrategyBank(), [[], []], self.config.hullcover.cap_c,
                                   self.state.params["actor"])
    return self.hullcover_state.pool_arrays(self.state.params["actor"])

  def hullcover_round(self):
    """A round of HullCover at the current learner step (the first one, at step 0, only stores the
    initial policy). Rounds are due at `hullcover_state.next_round_step()`."""
    if self.hullcover_state is None:
      self.hullcover_state = hullcover.HullCoverState(self.config.hullcover, self.config.seed)
    self.hullcover_state.run_round(self, self._vertex_reference)
    self._vertex_reference = None
    self._pool = self._pool_arrays()

  def attach_selection(self, selection: "hullcover.Selection"):
    """The HullCover portfolio (k mixtures over each pool) used by the leaf values."""
    self.selection = selection
    self.num_portfolio = selection.k + 1

  # ----------------------------------------------------------------------------
  # Inference API
  # ----------------------------------------------------------------------------

  def policy(self, obs: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Blueprint policy [..., A] for infoset tensors [..., I] and legal actions [..., A]."""
    obs = jnp.asarray(obs, dtype=jnp.float32)
    legal = jnp.asarray(legal, dtype=jnp.float32)
    return np.asarray(_apply_policy(self.actor, self.state.params["actor"], obs, legal))

  def state_values(self, state_tensors: np.ndarray) -> np.ndarray:
    """P1 values of the portfolio policies in the given states:
    multi valued [H, Pl, K + 1], where [h, i, f] has player i on its blueprint and
    the opponent on its option f (0 being the blueprint), or
    matrix valued [H, K + 1, K + 1] with the P1 option in rows."""
    state_tensors = jnp.asarray(state_tensors, dtype=jnp.float32)
    values = np.asarray(_apply_mvs(self.mvs_network, self.state.mvs_target, state_tensors))
    if self.is_hullcover:
      return hullcover.combine(values, self._require_selection(), self.config.value_type)
    num_states, k = state_tensors.shape[0], self.config.num_transformations
    if self.config.value_type == MATRIX_VALUED:
      return values.reshape(num_states, self.num_portfolio, self.num_portfolio)
    p1_view = values[:, :k + 1]
    p2_view = np.concatenate([values[:, :1], values[:, k + 1:]], axis=-1)
    return np.stack([p1_view, p2_view], axis=1)

  def leaf_values(self) -> LeafValues:
    """Depth-limit leaf values for `tree_builder.build_tree`."""
    if self.is_hullcover:
      self._require_selection()
    return LeafValues(fn=lambda states, state_tensors, legal: self.state_values(state_tensors),
                      value_type=self.config.value_type, num_options=self.num_portfolio)

  def exact_leaf_values(self) -> LeafValues:
    """Leaf values of the portfolio evaluated exactly in the rest of the game below each leaf
    (only for short follow-ups)."""
    if self.is_hullcover:
      return hullcover.ExactPool(self).leaf_values(self._require_selection(), self.config.value_type)
    return make_exact_leaf_values(self.game, self.portfolio_policies, self.num_portfolio, self.config.value_type)

  def _require_selection(self) -> "hullcover.Selection":
    if self.selection is None:
      raise ValueError("The HullCover portfolio is selected at evaluation, attach it with attach_selection "
                       "(evaluation.runner.load_for_eval).")
    return self.selection

  def portfolio_policies(self, player: int, obs: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """The portfolio policies [..., A, K + 1] of a player, the first one being the blueprint."""
    if self.is_hullcover:
      raise ValueError("HullCover portfolio members draw a pool member at the leaf, use exact_leaf_values.")
    return np.asarray(_apply_portfolio(self.actor, self.transformation_network, self.state.params["actor"],
                                       self.state.transformation_params[player], jnp.asarray(obs, dtype=jnp.float32),
                                       jnp.asarray(legal, dtype=jnp.float32)))

  def __getstate__(self):
    return {"config": self.config, "game_name": self.game_name, "game_params": self.game_params,
            "state": jax.device_get(self.state), "hullcover_state": self.hullcover_state}

  def __setstate__(self, state):
    config = state["config"]
    missing = {k: v for k, v in LEGACY_DEFAULTS.items() if k not in vars(config)}
    self.config = dataclasses.replace(config, **missing) if missing else config
    self.game_name = state["game_name"]
    self.game_params = state["game_params"]
    self.init()
    self.state = jax.tree.map(jnp.asarray, state["state"])
    self.hullcover_state = state.get("hullcover_state")


def masked_mean_over_time(x: chex.Array, mask: chex.Array) -> chex.Array:
  """Mean of x [T, B, K] over the time dimension where mask [T, B] is set."""
  normalization = jnp.sum(mask, axis=0)[..., None]
  return jnp.sum(x * mask[..., None], axis=0) / (normalization + (normalization == 0.0))
