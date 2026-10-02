from abc import ABC, abstractmethod
import chex
import jax.numpy as jnp

class InformationType():
  MDP = 0 # Markov decision process (num_players == 1)
  POMDP = 1 # Partially observable Markov decision process (num_players == 1)
  PIG = 2 # Perfect information game (num_players > 1)
  IIG = 3 # Imperfect information game (num_players > 1)


class GameState(ABC):
  pass


#Abstract parent class for any jax game
class JaxGame(ABC):
  """Abstract parent class for any JaxGame.
  Chance nodes are supported and represented explicitly. By design, a chance node
  is never followed by another chance node: consecutive chance events are absorbed
  into a single chance node, whose outcomes are their combinations."""
  
  def new_initial_state(self):
    """Returns the information tensors for the initial state"""
    init_state, legals = self.initialize_structures()
    return self.get_info(init_state)
  
  # Do not use this method if you plan to use any other tensor
  def state_tensor(self, game_state):
    return self.get_info(game_state)[0]
  
  
  # Do not use this method if you plan to use any other tensor
  def information_state_tensor(self, game_state, player: int):
    if player == 0:
      return self.get_info(game_state)[1]
    else:
      return self.get_info(game_state)[2]
  
  
  # Do not use this method if you plan to use any other tensor
  def public_state_tensor(self, game_state):
    return self.get_info(game_state)[3]
  
  def __str__(self):
    params_dict = self.params_dict()
    empty = ""
    params_str = empty.join(f"Param: {key} value: {value}" for key, value in params_dict.items())
    return str(self.game_name() + " " + params_str)
  
  def to_compact_str(self):
    """Returns a compact string representation of 
    the game, suitable for directory specification"""
    params_dict = self.params_dict()
    empty = ""
    params_str = empty.join(f"_{value}" for value in params_dict.values())
    return str(self.game_name() + params_str)
  
  def __repr__(self):
    return self.__str__()


  def is_chance(self, game_state: GameState)-> chex.Array:
    """Return whether the given game state is a chance node."""
    return jnp.array(False)
  

  def get_outcomes_and_probs(self, game_state:GameState) ->tuple[chex.Array, chex.Array]:
    """Gets the chance outcomes and probabilities of them for a game state at given turn.
    Outcomes here are just actions, that will be passed to apply_action to get the actual
    post chance node states.
    Returns invalid output if the state is not a chance node."""
    if self.num_players() == 1:
      return jnp.zeros(1, dtype=jnp.int32), jnp.zeros(1)
    return jnp.zeros((1, self.num_players()), dtype=jnp.int32), jnp.zeros(1)

  def get_outcomes_and_probs_with_check(self, game_state:GameState) ->tuple[chex.Array, chex.Array]:
    """Gets all chance outcomes and their respective probabilities
    for a chance node. This version checks whether the state is 
    actually a chance node first, and will not allow being called on none chance nodes
    (where the behavior is undefined). Uses standard Python assert and as such will not work under vmap"""
    assert self.is_chance(game_state), "Tried getting the chance outcomes and their probabilities of a non chance node!"
    return self.get_outcomes_and_probs(game_state)
  
  def max_chance_outcomes(self) ->int:
    """Returns the maximum chance outcomes of any chance
    node in the game."""
    return 1
  
  def depth_chance_outcomes(self, depth:int) ->int:
    """Return how many of the first indices of probs 
    gotten by get_outcomes_and_probs can contain valid chance outcomes 
    at the given tree depth. This does not necessarily mean
    that all of the indices have a nonzero prob. This is merely meant
    to filter outcomes that are certainly invalid."""
    return 1
  
  def depth_chance_valid_outcomes(self, depth:int) ->int:
    """Return how many possible chance outcomes are there
    at given depth. Unlike depth_chance_outcomes, this returns
    the actual amount of the outcomes, but not their index range (eg.
    using this for indexing can filter out valid outcomes)"""
    return 1
  
  def max_trajectory_length_no_chance(self) ->int:
    """Returns the maximum length of the game trajectory
    excluding chance nodes. Useful when we want to filter out
    the chance nodes under jit."""
    return self.max_trajectory_length()

  @abstractmethod
  def game_name(self) ->str:
    pass
  
  @abstractmethod
  def params_dict(self)->dict:
    pass
  @abstractmethod
  def information_type(self)->int:
    """Returns what kind of information
    this game provides. The types are detailed in
    the InformationType class."""
    pass

  @abstractmethod
  def num_players(self)->int:
    pass

  @abstractmethod
  def state_tensor_shape(self) ->int:
    pass
  
  @abstractmethod
  def observation_tensor_shape(self) ->int:
    pass
  
  @abstractmethod
  def information_state_tensor_shape(self) ->int:
    pass
    
  @abstractmethod
  def public_state_tensor_shape(self) ->int:
    pass
  
  @abstractmethod
  def num_distinct_actions(self) ->int:
    pass
  
  @abstractmethod
  def max_trajectory_length(self) ->int:
    pass
  
  #returns game_state, legal_actions
  @abstractmethod
  def initialize_structures(self) -> tuple[GameState, chex.Array]:
    """Returns a tuple of initial game state
    and its legal actions"""
    pass

  #returns state_tensor, p1_infoset_tensor, p2_infoset_tensor, public_state_tensor
  # for multi agent and state_tensor, observation_tensor for single agent
  @abstractmethod
  def get_info(self, game_state):
    """Get information tensor about the state. For 
    multi-agent setting, this is and ordered 4-tuple state_tensor, p1_infoset_tensor,
    p2_infoset_tensor, public_state_tensor. Some environments may instead return observ
    ation tensors instead of information state tensors, but the order should be the same, with the second and third tensor being the player specific ones."""
    pass

  #returns new_game_state, terminal, rewards, new_legals
  @abstractmethod
  def apply_action(self, game_state: GameState, actions: chex.Array) ->tuple[GameState, chex.Array, chex.Array, chex.Array]:
    """Apply the joint action to the game state. The actions should be
    of shape (num_players), specifying an action of each player.
    Returns a tuple of new game state, terminal flag of the new game state,
    reward for applying the action and legal actions of the new game state"""
    pass