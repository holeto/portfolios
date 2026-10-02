import jax
import chex
import jax.numpy as jnp
import numpy as np

import functools
from games.jax_game import JaxGame, GameState, InformationType

u8 = jnp.uint8
i8 = jnp.int8
f32 = jnp.float32

@chex.dataclass(frozen=True)
class RPSSTate(GameState):
  p1_points: chex.Array
  terminal: chex.Array


class JaxRPS(JaxGame):
  def __init__(self) -> None:
    self.max_turns = 2 # Decision node, terminal state
    self.moves = {"r" : 0, "p": 1, "s": 2}
    # p1 wins: pr: 1, sp, 1, rs -2
    # p1 loses : rp: -1, ps: -1, sr: 2
    self.actions = 3
    
  
  def game_name(self):
    return "rps"
  
  def params_dict(self):
    return {}
  
  def information_type(self):
    return InformationType.IIG
  
  def num_players(self):
    return 2

  def state_tensor_shape(self):
    return self.information_state_tensor_shape() - 2
   
  def information_state_tensor_shape(self):
    return 8 # 2 for player encoding, 2 for the terminal flag, 4 for player 1 points
    # this is just the observation tensor of the current state, but because the game is only a single turn, it corresponds
    # to the perfect recall infoset 

  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def public_state_tensor_shape(self):
    return self.state_tensor_shape()
  
  def num_distinct_actions(self):
    return self.actions
  
  def max_trajectory_length(self):
    return self.max_turns
  
  
  @functools.partial(jax.jit, static_argnums=(0))
  def initialize_structures(self):
    game_state = RPSSTate(terminal = jnp.array(False, dtype=bool),
                          p1_points = jnp.array(-2))
    return game_state, jnp.ones((2, self.actions))

  
  # This just returns an observation tensor for the current 
  # state, however, since the game is just one turn 
  # this is actually a perfect recall representation, since the no information
  # of the initial state will not play a role
  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state:RPSSTate):
    terminal_oh = jax.nn.one_hot(game_state.terminal.astype(int), 2) 
    p1_points_oh = jax.nn.one_hot(game_state.p1_points + 2, 4)
    state_tensor = jnp.concatenate([terminal_oh.ravel(), p1_points_oh.ravel()], axis=0)
    p1_infoset_tensor = jnp.concatenate([jax.nn.one_hot(0, 2), state_tensor], axis=0)
    p2_infoset_tensor = jnp.concatenate([jax.nn.one_hot(1, 2), state_tensor], axis=0)
    #We use just p1 points for state_tensor, public state tensor and p2_infoset_tensor as well, since they uniquely define p2 points as well
    return state_tensor, p1_infoset_tensor, p2_infoset_tensor, state_tensor
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, game_state:RPSSTate, actions):
    #A single action. The state after applying will always be terminal
    terminal = jnp.array(True, dtype=bool)
    action_difference = actions[0]- actions[1]
    # R: 0, P: 1, S: 2, with these, p1 wins at values -2, 1, loses at -1, 2 and its a tie at 0
    # sign(x) * (3 - 2 * abs(x)) produces these values, assuming sign(0) = 0, which it is in jnp
    p1_points = jnp.sign(action_difference) * (3 - (2 * jnp.abs(action_difference)))

    legal_actions = jnp.ones((2, self.actions))
    new_game_state = RPSSTate(terminal = terminal,
                              p1_points = p1_points)
    
    return new_game_state, terminal, p1_points.astype(f32), legal_actions

@chex.dataclass(frozen=True)
class JaxStochasticRPSState(GameState):
  p1_points: chex.Array
  terminal: chex.Array
  game_type: chex.Array
  is_chance: chex.Array

class JaxStochasticRPS(JaxGame):
  """Made to have a simple 2p0s game with chance.
  The initial node is a chance node and it picks between three outcomes, 
  which determine the type of game. 0 means standard RPS,
  1 perturbed RPS, where all payoffs featuring the "paper" action from either player
  are multiplied by 2, and 2 RPS where player 1 loses the "paper" action.
  Importantly, both players know which game is being played"""
  def __init__(self) -> None:
    self.moves = {"r" : 0, "p": 1, "s": 2}
    self.games = {"standard": 0, "perturbed": 1, "lose_paper": 2}
    # p1 wins: pr: 1, sp, 1, rs -2
    # p1 loses : rp: -1, ps: -1, sr: 2
    self.actions = 3
    self.game_types = 3
    
  
  def game_name(self):
    return "stochastic_rps"
  
  def params_dict(self):
    return {}
  
  def num_players(self):
    return 2
  
  def information_type(self):
    return InformationType.IIG

  def state_tensor_shape(self):
    return self.information_state_tensor_shape() - 2
   
  def information_state_tensor_shape(self):
    return 12 # 2 for distinguishing player 2 for terminal flag, 5 for player 1 points
  # which differentiates the winner and 3 for encoding the type of game
  # being played

  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def public_state_tensor_shape(self):
    return self.state_tensor_shape()
  
  def num_distinct_actions(self):
    return self.actions
  
  def max_trajectory_length(self):
    return 3 # Chance node, decision node, terminal state
  
  def max_trajectory_length_no_chance(self):
    return 2 # decision node, terminal state
  
  
  @functools.partial(jax.jit, static_argnums=(0))
  def initialize_structures(self):
    game_state = JaxStochasticRPSState(terminal = jnp.array(False, dtype=bool),
                          is_chance = jnp.array(True, dtype=bool),
                          p1_points = jnp.array(-3, dtype=i8),
                          game_type = jnp.array(-1, dtype=i8))
    return game_state, jnp.ones((2, self.actions), dtype=u8)

  
  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state:JaxStochasticRPSState):
    terminal_oh = jax.nn.one_hot(game_state.terminal.astype(int), 2)
    #The p1 points span range [-2, ..., 2], so ve need to represent 5 values 
    p1_points_oh = jax.nn.one_hot(game_state.p1_points + 2, 5)
    game_played_oh = jax.nn.one_hot(game_state.game_type, self.game_types)
    state_tensor = jnp.concatenate([terminal_oh.ravel(), p1_points_oh.ravel(), game_played_oh], axis=0)
    state_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(state_tensor), state_tensor)
    p1_infoset_tensor = jnp.concatenate([jax.nn.one_hot(0, 2), state_tensor], axis=0)
    p2_infoset_tensor = jnp.concatenate([jax.nn.one_hot(1, 2), state_tensor], axis=0)
    p1_infoset_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(p1_infoset_tensor), p1_infoset_tensor)
    p2_infoset_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(p2_infoset_tensor), p2_infoset_tensor)
    #We use just p1 points for state_tensor, public state tensor and p2_infoset_tensor as well, since they uniquely define p2 points as well
    return state_tensor, p1_infoset_tensor, p2_infoset_tensor, state_tensor
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def get_outcomes_and_probs(self, game_state: JaxStochasticRPSState):
    outcomes = jnp.repeat(jnp.arange(self.game_types)[..., None], 2, axis=-1)
    probs = jnp.where(game_state.is_chance, jnp.ones(self.game_types) / self.game_types, jnp.zeros(self.game_types))
    return outcomes, probs
  
  def depth_chance_outcomes(self, depth):
    if depth == 0:
      return 3
    return 1
  
  def depth_chance_valid_outcomes(self, depth):
    return self.depth_chance_outcomes(depth)
  
  def is_chance(self, game_state: JaxStochasticRPSState):
    return game_state.is_chance
  
  def apply_action_chance(self, game_state:JaxStochasticRPSState, actions):
    # By convention, player 2 "plays" in the chance node
    game_type = actions[1].astype(i8)
    legals = jnp.ones((2, self.actions), dtype=u8)
    legals = jnp.where(game_type == self.games["lose_paper"], (legals - jax.nn.one_hot(self.moves["p"], self.actions)[None, ...]).astype(legals.dtype), legals )
    after_chance_state = JaxStochasticRPSState(p1_points = game_state.p1_points,
                                               terminal = jnp.array(False, dtype=bool),
                                               game_type = game_type,
                                               is_chance = jnp.array(False, dtype=bool))
    return after_chance_state, jnp.array(False, dtype=bool), jnp.array(0, dtype=f32), legals 

  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action_no_chance(self, game_state:JaxStochasticRPSState, actions):
    #A single action. The state after applying will always be terminal
    terminal = jnp.array(True, dtype=bool)
    action_difference = actions[0]- actions[1]
    # R: 0, P: 1, S: 2, with these, p1 wins at values -2, 1, loses at -1, 2 and its a tie at 0
    # sign(x) * (3 - 2 * abs(x)) produces these values, assuming sign(0) = 0, which it is in jnp
    p1_points = jnp.sign(action_difference) * (3 - (2 * jnp.abs(action_difference)))

    perturbed = game_state.game_type == self.games["perturbed"]
    played_paper = jnp.any(actions == self.moves["p"])

    #If playing the perturbed game, multiply all "paper"
    # payoffs by 2
    p1_points = p1_points * (1 + jnp.logical_and(perturbed, played_paper))

    legals = jnp.ones((2, self.actions), dtype=u8)
    legals = jnp.where(game_state.game_type == self.games["lose_paper"], (legals - jax.nn.one_hot(self.moves["p"], self.actions)[None, ...]).astype(legals.dtype), legals )
    new_game_state = JaxStochasticRPSState(terminal = terminal,
                              p1_points = p1_points.astype(i8),
                              game_type = game_state.game_type,
                              is_chance = jnp.array(False, dtype=bool))
    
    return new_game_state, terminal, p1_points.astype(f32), legals
  
  def apply_action(self, game_state: JaxStochasticRPSState, actions):
    return jax.lax.cond(game_state.is_chance, self.apply_action_chance, self.apply_action_no_chance, game_state, actions)
  
@chex.dataclass(frozen=True)
class JaxTurnBasedRPSState(GameState):
  turn: chex.Array
  p1_action: chex.Array
  p2_action: chex.Array


class JaxTurnBasedRPS(JaxGame):
  """Turn-based RPS: player 1 picks rock, paper or scissors, then player 2 picks
  without observing the choice of player 1, so all its histories are in a single
  infoset. The non-acting player has the single dummy action 0. Uniform
  strategies are the equilibrium, with value 0.

  `p2_rock_win_error` models an approximation error of the values: player 2
  winning with rock receives 1 + 3 * p2_rock_win_error instead of 1.
  """

  def __init__(self, p2_rock_win_error: float = 0.0) -> None:
    self.actions = 3
    self.turns = 3  # player 1 decision, player 2 decision, terminal
    self.p2_rock_win_error = p2_rock_win_error
    self.moves = {"r": 0, "p": 1, "s": 2}

  def game_name(self):
    return "turn_based_rps"

  def params_dict(self):
    return {"p2_rock_win_error": self.p2_rock_win_error} if self.p2_rock_win_error else {}

  def information_type(self):
    return InformationType.IIG

  def num_players(self):
    return 2

  # Turn [3], played actions of both players [2 * 3].
  def state_tensor_shape(self):
    return self.turns + 2 * self.actions

  # Player [2], turn [3], own played action [3].
  def information_state_tensor_shape(self):
    return 2 + self.turns + self.actions

  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()

  # Turn [3].
  def public_state_tensor_shape(self):
    return self.turns

  def num_distinct_actions(self):
    return self.actions

  def max_trajectory_length(self):
    return self.turns

  def _legal(self, turn):
    acting = jnp.ones(self.actions, dtype=f32)
    dummy = jax.nn.one_hot(0, self.actions, dtype=f32)
    return jnp.where(turn == 0, jnp.stack([acting, dummy]), jnp.stack([dummy, acting]))

  @functools.partial(jax.jit, static_argnums=(0,))
  def initialize_structures(self):
    game_state = JaxTurnBasedRPSState(turn=jnp.array(0, dtype=i8), p1_action=jnp.array(-1, dtype=i8),
                                      p2_action=jnp.array(-1, dtype=i8))
    return game_state, self._legal(0)

  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state: JaxTurnBasedRPSState):
    turn_oh = jax.nn.one_hot(game_state.turn, self.turns)
    p1_action_oh = jax.nn.one_hot(game_state.p1_action, self.actions)
    p2_action_oh = jax.nn.one_hot(game_state.p2_action, self.actions)
    state_tensor = jnp.concatenate([turn_oh, p1_action_oh, p2_action_oh])
    p1_infoset = jnp.concatenate([jax.nn.one_hot(0, 2), turn_oh, p1_action_oh])
    p2_infoset = jnp.concatenate([jax.nn.one_hot(1, 2), turn_oh, p2_action_oh])
    return state_tensor, p1_infoset, p2_infoset, turn_oh

  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, game_state: JaxTurnBasedRPSState, actions):
    first_turn = game_state.turn == 0
    p1_action = jnp.where(first_turn, actions[0], game_state.p1_action).astype(i8)
    p2_action = jnp.where(first_turn, game_state.p2_action, actions[1]).astype(i8)
    terminal = ~first_turn
    action_difference = p1_action.astype(jnp.int32) - p2_action.astype(jnp.int32)
    reward = jnp.where(terminal, jnp.sign(action_difference) * (3 - 2 * jnp.abs(action_difference)), 0).astype(f32)
    p2_rock_win = jnp.logical_and(p2_action == self.moves["r"], p1_action == self.moves["s"])
    reward = jnp.where(jnp.logical_and(terminal, p2_rock_win), -(1.0 + 3.0 * self.p2_rock_win_error), reward).astype(f32)
    turn = jnp.minimum(game_state.turn + 1, self.turns - 1).astype(i8)
    new_state = JaxTurnBasedRPSState(turn=turn, p1_action=p1_action, p2_action=p2_action)
    return new_state, terminal, reward, self._legal(turn)


def main():
  game = JaxStochasticRPS()
  def _tree_walk(state: JaxStochasticRPS, legals, terminal, reward, depth=0):
    legals = np.asarray(legals)
    print(f"In state: {state}")
    print(f"Legals: {legals}")
    print(f"Reward: {reward}")
    if terminal:
      return
    if game.is_chance(state):
      outcomes,  probs = game.get_outcomes_and_probs(state)
      for outcome, prob in zip(outcomes, probs):
        print(f"Appling outcome {outcome} with probability {prob}")
        if prob < 1e-5:
          continue
        new_state, new_terminal, reward, new_legals = game.apply_action(state, outcome)
        _tree_walk(new_state, new_legals, new_terminal, reward, depth=depth + 1)
      return
    for a1i, a1 in enumerate(legals[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals[1]):
        if a2 < 0.5:
          continue
        joint_action = jnp.array([a1i, a2i])
        print(f"Applying action {joint_action}")
        new_state, new_terminal, reward, new_legals = game.apply_action(state, joint_action)
        _tree_walk(new_state, new_legals, new_terminal, reward, depth = depth + 1)
  init_state, init_legals = game.initialize_structures()
  _tree_walk(init_state, init_legals, False, 0)

if __name__ == "__main__":
  main()
