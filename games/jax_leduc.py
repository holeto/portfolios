import jax
import jax.numpy as jnp
import numpy as np
import chex

import functools
from games.jax_game import JaxGame, GameState, InformationType

INVALID_ID = 0
FOLD_ID = 1
CALL_ID = 2
RAISE_ID = 3

@chex.dataclass(frozen=True)
class LeducGameState(GameState):
    action_history: chex.Array
    public_card: chex.Array
    private_cards: chex.Array
    current_chips: chex.Array
    turns_this_round: chex.Array
    terminal: chex.Array #Remember this to make sure that the state is correctly marked as terminal, 
    #when playing additional actions in terminal state
    turn: int
    is_chance: chex.Array


class JaxLeduc(JaxGame):
  def __init__(self):
    #Invalid action, fold, call, raise
    self.num_actions = 4
    #Two rounds and in each the maximum length trajectory consists of
    # actions Call, Raise, Raise, [Call or Fold],
    self.max_turns = 8
    #Cards in two suits, three cards from each suit
    self.total_cards = 6
    self.players = 2
    #Max 4 raises. Starting at 1, raises in the first
    # round are 2 + 2 and in the second round 4 + 4
    self.max_bet_amount = 13
    self.max_raises_per_round = 2
    self.raise_amount = 2
    #assuming that cards of different suits are counted as 
    #distinct cards
    self.private_chance_outcomes = 30
    #there are 4 cards left in the deck
    #self.public_chance_outcomes = 4
    #self.chance_outcomes = 120
    self.invalid_action_mask = jax.nn.one_hot(INVALID_ID, self.num_actions)

  
  def num_distinct_actions(self):
    return self.num_actions
  
  def max_chance_outcomes(self):
    return self.private_chance_outcomes
  
  def max_trajectory_length(self):
    # Includes the terminal state as well and the 2 chance nodes
    return self.max_turns + 3
  
  def max_trajectory_length_no_chance(self):
    # Just the terminal added now and not the chance nodes.
    return self.max_turns + 1
  
  def depth_chance_outcomes(self, depth:int):
    if depth == 0:
      return self.private_chance_outcomes
    elif depth >= 3 and depth <= 5:
      return self.total_cards
    return 1
  
  def depth_chance_valid_outcomes(self, depth:int):
    if depth == 0:
      return self.private_chance_outcomes
    elif depth >= 3 and depth <= 5:
      return self.total_cards - 2
    return 1
  
  def game_name(self):
    return "leduc"
  
  def params_dict(self):
    return {}
  
  def information_type(self):
    return InformationType.IIG
  
  def num_players(self):
    return 2
  
  def information_state_tensor_shape(self):
    # One hot encoded receiving player
    # One hot encoded private card of player
    # One hot encoded public card (1 bit added to recognize not yet revealed)
    # One hot encoded actions leading up to the terminal turn (not the invalid added actions)
    return self.players + self.total_cards + self.public_state_tensor_shape()
  
  def public_state_tensor_shape(self):
    # One hot encoded public card (1 bit added to recognize not yet revealed)
    # One hot encoded actions leading up to the terminal turn (not the invalid added actions)
    return self.total_cards + 1 + (self.max_turns) * (self.num_actions - 1)
  
  def state_tensor_shape(self):
    #One hot encoded private cards of both players
    #One hot encoded public card (1 bit added to recognize not yet revealed)
    #One hot encoded actions leading up to the terminal turn (not the invalid added actions) 
    return 2 * self.total_cards + self.public_state_tensor_shape()
  
  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  @functools.partial(jax.jit, static_argnums=(0))
  def generate_all_private_card_nodes(self, game_state: GameState) -> tuple[LeducGameState, chex.Array, chex.Array]:
    """ Get an array of all game states corresponding
    to all the outcomes of the first chance node
    and their legal actions (all the root states have the
    same legal actions.)
    """
    fold_oh = jax.nn.one_hot(FOLD_ID, self.num_actions)
    starting_action_mask = jnp.ones(self.num_actions) - self.invalid_action_mask - fold_oh
    #[H, A - 1]
    cards = jnp.arange(self.total_cards)[..., None]
    p1_private_cards = jnp.repeat(cards, self.total_cards - 1, axis=0)
    #TODO: This can probably be done better.
    p2_private_cards = jnp.concatenate([jnp.r_[0:i:1, i+1:self.total_cards:1] for i in range(self.total_cards)])[..., None]
    private_cards = jnp.concatenate([p1_private_cards, p2_private_cards], axis=1).astype(jnp.int16)

    #We assume that player 1 is the starting one      
    p1_legal_mask = starting_action_mask
    p2_legal_mask = self.invalid_action_mask
    legals = jnp.stack([p1_legal_mask, p2_legal_mask], axis=0)
    legals = jnp.tile(legals[None, ...], (self.private_chance_outcomes, 1, 1))
    stacked_game_state = jax.tree_util.tree_map(lambda x: jnp.tile(x[None, ...], (self.private_chance_outcomes,) + (1,) * len(x.shape)), game_state)
    game_states = LeducGameState(action_history=stacked_game_state.action_history,
                            public_card = stacked_game_state.public_card.astype(jnp.int16),
                            private_cards= private_cards,
                            current_chips = stacked_game_state.current_chips.astype(jnp.int16),
                            turns_this_round = stacked_game_state.turns_this_round.astype(jnp.int16),
                            terminal= stacked_game_state.terminal.astype(bool),
                            turn=stacked_game_state.turn,
                            is_chance = jnp.zeros_like(stacked_game_state.is_chance, dtype=bool))
    return game_states, legals, jnp.ones(self.private_chance_outcomes) / self.private_chance_outcomes
  
  @functools.partial(jax.jit, static_argnums=(0))
  def generate_all_public_card_nodes(self, state: LeducGameState) -> tuple[LeducGameState, chex.Array, chex.Array]:
    public_cards = jnp.arange(self.total_cards, dtype=jnp.int16) + 1
    public_cards = jnp.pad(public_cards, (0, self.private_chance_outcomes - self.total_cards), constant_values=-1)
    valid = 1 - jnp.sum(jax.nn.one_hot(state.private_cards, self.total_cards), axis=0)
    valid = jnp.pad(valid, (0, self.private_chance_outcomes - self.total_cards), constant_values=0)
    stacked_game_state = jax.tree_util.tree_map(lambda x: jnp.tile(x[None, ...], (self.private_chance_outcomes,) + (1,) * len(x.shape)), state)
    #We assume that player 1 is the starting one
    fold_oh = jax.nn.one_hot(FOLD_ID, self.num_actions)      
    p1_legal_mask = jnp.ones(self.num_actions) - self.invalid_action_mask - fold_oh
    p2_legal_mask = self.invalid_action_mask
    legals = jnp.stack([p1_legal_mask, p2_legal_mask], axis=0)
    legals = jnp.tile(legals[None, ...], (self.private_chance_outcomes, 1, 1))
    game_states = LeducGameState(action_history=stacked_game_state.action_history,
                            public_card = public_cards,
                            private_cards=stacked_game_state.private_cards.astype(jnp.int16),
                            current_chips = stacked_game_state.current_chips.astype(jnp.int16),
                            turns_this_round = stacked_game_state.turns_this_round.astype(jnp.int16),
                            terminal= stacked_game_state.terminal.astype(bool),
                            turn=stacked_game_state.turn,
                            is_chance = jnp.zeros_like(stacked_game_state.is_chance, dtype=bool))
    return game_states, legals, valid / jnp.sum(valid)
  
  def is_chance(self, game_state: LeducGameState) ->chex.Array:
    return game_state.is_chance
  
  @functools.partial(jax.jit, static_argnums=(0))
  def get_outcomes_and_probs(self, game_state:LeducGameState) -> tuple[LeducGameState, chex.Array, chex.Array]:
    outcomes = jnp.stack([jax.nn.one_hot(0, self.private_chance_outcomes), jnp.arange(self.private_chance_outcomes)], axis=-1)
    def invalid_probs(game_state):
      
      return jnp.zeros(self.private_chance_outcomes)
    
    def private_probs(game_state:LeducGameState):

      return jnp.ones(self.private_chance_outcomes) / self.private_chance_outcomes
    
    def public_probs(game_state: LeducGameState):
      valid = 1 - jnp.sum(jax.nn.one_hot(game_state.private_cards, self.total_cards), axis=0)
      valid = jnp.pad(valid, (0, self.private_chance_outcomes - self.total_cards), constant_values=0)
      return valid / jnp.sum(valid)
    
    probs = jax.lax.cond(game_state.is_chance,
                                           lambda s: jax.lax.cond(s.turn == 0, private_probs, public_probs, s) 
                                           , invalid_probs, game_state)
    return outcomes, probs
   
  
       
  
  @functools.partial(jax.jit, static_argnums=(0))
  def initialize_structures(self):  
    current_chips = jnp.ones(self.players, dtype=jnp.int16)
    #[H, A - 1]
    action_history = jnp.zeros([self.max_turns, self.num_actions - 1])
    public_card = jnp.array(0, dtype=jnp.int16)
    turns_this_round = jnp.zeros(1, dtype=jnp.int16)
    legals = jnp.ones((2, self.num_actions))
    game_state = LeducGameState(action_history=action_history,
                            public_card = public_card,
                            private_cards=jnp.full(2, -1, dtype=jnp.int16),
                            current_chips = current_chips,
                            turns_this_round = turns_this_round,
                            terminal= jnp.array(False),
                            turn=0,
                            is_chance = jnp.array(True))
    return game_state, legals
  
  

  @functools.partial(jax.jit, static_argnums=(0))
  def get_info(self, game_state:LeducGameState):
    #One additional bit for public card not dealt yet
    public_card_oh = jax.nn.one_hot(game_state.public_card, self.total_cards + 1)
    public_state_tensor = jnp.concatenate([public_card_oh.ravel(), game_state.action_history.ravel()], axis=0)
    public_state_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(public_state_tensor), public_state_tensor)
    private_cards_oh = jax.nn.one_hot(game_state.private_cards, self.total_cards)

    p1_player = jax.nn.one_hot(0, 2)
    
    p1_infoset_tensor = jnp.concatenate([p1_player.ravel(), private_cards_oh[0], public_state_tensor], axis=0)
    p1_infoset_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(p1_infoset_tensor), p1_infoset_tensor)
    p2_infoset_tensor = jnp.concatenate([1 - p1_player.ravel(), private_cards_oh[1], public_state_tensor], axis=0)
    
    p2_infoset_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(p2_infoset_tensor), p2_infoset_tensor)

    state_tensor = jnp.concatenate([private_cards_oh.ravel(), public_state_tensor], axis=0)
    state_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(state_tensor), state_tensor)

    return state_tensor, p1_infoset_tensor, p2_infoset_tensor, public_state_tensor

  
  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action(self, game_state:LeducGameState, actions: chex.Array):
   return jax.lax.cond(game_state.is_chance, self.apply_action_chance, self.apply_action_no_chance, game_state, actions)
  
  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action_chance(self, game_state: LeducGameState, actions: chex.Array):
    init_chance = game_state.turn == 0
    def apply_init(game_state: LeducGameState, actions: chex.Array):
      outcomes, legals, probs = self.generate_all_private_card_nodes(game_state)
      action_oh = jax.nn.one_hot(actions[1], self.private_chance_outcomes)
      outcome = jax.tree_util.tree_map(lambda x: jnp.sum(x * jnp.reshape(action_oh, (action_oh.shape[0], ) + (1,) * len(x.shape[1:])), axis=0),outcomes)
      legals = jnp.sum(action_oh[..., None, None] * legals, axis=0)
      #chance nodes do not produce lead to terminal state or produce any reward here
      return outcome, jnp.array(False), jnp.array(0, dtype=jnp.float32), legals
    def apply_public(game_state: LeducGameState, actions: chex.Array):
      outcomes, legals, probs = self.generate_all_public_card_nodes(game_state)
      action_oh = jax.nn.one_hot(actions[1], self.private_chance_outcomes)
      outcome = jax.tree_util.tree_map(lambda x: jnp.sum(x * jnp.reshape(action_oh, (action_oh.shape[0], ) + (1,) * len(x.shape[1:])), axis=0),outcomes)
      legals = jnp.sum(action_oh[..., None, None] * legals, axis=0)
      #chance nodes do not produce lead to terminal state or produce any reward here
      return outcome, jnp.array(False), jnp.array(0, dtype=jnp.float32), legals
    outcome, terminal, reward, legals = jax.lax.cond(init_chance, apply_init, apply_public, game_state, actions)
    new_game_state = LeducGameState(action_history=outcome.action_history,
                           public_card = outcome.public_card.astype(jnp.int16),
                           private_cards=outcome.private_cards.astype(jnp.int16),
                           current_chips = outcome.current_chips.astype(jnp.int16),
                           turns_this_round = outcome.turns_this_round.astype(jnp.int16),
                           terminal = outcome.terminal.astype(bool),
                           turn = outcome.turn.astype(int),
                           is_chance = outcome.is_chance.astype(bool))
    return new_game_state, terminal, reward, legals

  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action_no_chance(self, game_state : LeducGameState, actions: chex.Array):
    oh_actions = jax.nn.one_hot(actions, self.num_actions)
    oh_turn = jax.nn.one_hot(game_state.turn, self.max_turns)
    fold_oh = jax.nn.one_hot(FOLD_ID, self.num_actions)
    raise_oh = jax.nn.one_hot(RAISE_ID, self.num_actions)
    
    #We assume that player 1 is the starting one
    current_player = game_state.turns_this_round % 2
    max_chips = jnp.max(game_state.current_chips)
    oh_valid_action = jax.nn.one_hot(actions[current_player] - 1, self.num_actions - 1)

    #Integer division by 2 places cards into the [J1, J2], [Q1, Q2], [K1, K2] buckets
    player_card_types = jnp.floor_divide(game_state.private_cards, 2)
    round = game_state.public_card > 0

    
    folded = jnp.any(oh_actions[current_player] * fold_oh)
    raised = jnp.sum(oh_actions * raise_oh, axis=1)

    tie = jnp.all(jnp.isclose(player_card_types[0], player_card_types[1]))
    card_matched = jnp.any(jnp.floor_divide(game_state.public_card - 1, 2) == player_card_types)
    winner = jnp.where(card_matched, jnp.argmin(jnp.abs(jnp.floor_divide(game_state.public_card - 1, 2) - player_card_types)), jnp.argmax(player_card_types))
    winner = jnp.where(folded, 1 - current_player, winner)

    this_turn_played = oh_valid_action * oh_turn[..., None]
    action_history = game_state.action_history + this_turn_played

    #Taking advantage of the fact, that raises can only happen after each other
    num_raises = jnp.where(game_state.turn > 0, action_history[game_state.turn - 1, RAISE_ID - 1] + raised[current_player], 0)

    action_chips = jnp.concatenate([jnp.repeat(game_state.current_chips[..., None], 2, axis =1), jnp.array([max_chips, max_chips])[..., None] * jnp.ones(2)], axis=1)
    current_chips = jnp.sum(action_chips * oh_actions, axis=1)
    current_chips = current_chips + (raised * (round + 1) * self.raise_amount)
    
    bets_equal = jnp.all(jnp.isclose(current_chips[0], current_chips[1]))
    play_chance = jnp.logical_and(jnp.logical_and(game_state.turn >= 1, round == 0), bets_equal) 
    #make sure to properly reset to ready for the new round
    turns_this_round = jnp.where(play_chance, -1, game_state.turns_this_round)
    next_player = (turns_this_round + 1) % 2
    
    new_acting_legals = jnp.ones(self.num_actions) - self.invalid_action_mask
    #whether raise is still possible
    new_acting_legals = jnp.where(num_raises < self.max_raises_per_round, new_acting_legals, new_acting_legals - raise_oh)
    #whether fold is possible
    new_acting_legals = jnp.where(bets_equal, new_acting_legals - fold_oh, new_acting_legals)
    new_legals = jnp.where(next_player == 0, jnp.stack([new_acting_legals, self.invalid_action_mask], axis=0), jnp.stack([self.invalid_action_mask, new_acting_legals], axis=0))

   

    terminal = jnp.logical_or(folded, jnp.logical_and(jnp.logical_and(round > 0, turns_this_round >= 1), bets_equal))
    terminal = jnp.squeeze(terminal)
    
  
    #The division by self.max_bet_amount to make sure the rewards is normalized to [-1, 1] range
    reward = jnp.where(terminal, jnp.where(jnp.logical_and(tie, ~folded), 0, ((1 - 2 * winner) * current_chips[1-winner]) / self.max_bet_amount), 0)

    #If terminal state was already reached, mark this as terminal as well
    terminal = jnp.logical_or(game_state.terminal, terminal)

    new_game_state = LeducGameState(action_history=action_history,
                           public_card = game_state.public_card.astype(jnp.int16),
                           private_cards=game_state.private_cards.astype(jnp.int16),
                           current_chips = current_chips.astype(jnp.int16),
                           turns_this_round = (turns_this_round + 1).astype(jnp.int16),
                           terminal = terminal.astype(bool),
                           turn = game_state.turn + 1,
                           is_chance = play_chance)
    new_legals = jnp.where(play_chance, jnp.ones_like(new_legals), new_legals)

    return new_game_state, terminal, reward[0], new_legals
  
class JaxLeducRound1(JaxLeduc):
  """One-round Leduc poker: game ends after round-1 betting (no public card / round 2)."""

  def __init__(self, max_raises: int = 2):
    super().__init__()
    self.max_raises_per_round = max_raises
    # 0 raises: action_history still needs ≥1 row for JAX shape consistency
    self.max_turns = max(2, 2 + max_raises)
    self.max_bet_amount = 1 + max_raises * self.raise_amount  # 0→1, 1→3, 2→5

  def game_name(self):
    return "leduc_round1"

  def params_dict(self):
    return {'max_raises': self.max_raises_per_round}

  def max_trajectory_length(self):
    return self.max_turns + 2  # 1 private-card chance + turns + 1 terminal

  def max_trajectory_length_no_chance(self):
    return self.max_turns + 1  # turns + terminal

  def depth_chance_outcomes(self, depth: int):
    return self.private_chance_outcomes if depth == 0 else 1

  def depth_chance_valid_outcomes(self, depth: int):
    return self.private_chance_outcomes if depth == 0 else 1

  def public_state_tensor_shape(self):
    # No public card one-hot — only action history
    return self.max_turns * (self.num_actions - 1)

  @functools.partial(jax.jit, static_argnums=(0))
  def get_info(self, game_state: LeducGameState):
    public_state_tensor = game_state.action_history.ravel()
    public_state_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(public_state_tensor), public_state_tensor)
    private_cards_oh = jax.nn.one_hot(game_state.private_cards, self.total_cards)
    p1_player = jax.nn.one_hot(0, 2)
    p1_infoset_tensor = jnp.concatenate([p1_player.ravel(), private_cards_oh[0], public_state_tensor])
    p1_infoset_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(p1_infoset_tensor), p1_infoset_tensor)
    p2_infoset_tensor = jnp.concatenate([1 - p1_player.ravel(), private_cards_oh[1], public_state_tensor])
    p2_infoset_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(p2_infoset_tensor), p2_infoset_tensor)
    state_tensor = jnp.concatenate([private_cards_oh.ravel(), public_state_tensor])
    state_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(state_tensor), state_tensor)
    return state_tensor, p1_infoset_tensor, p2_infoset_tensor, public_state_tensor

  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action_chance(self, game_state: LeducGameState, actions: chex.Array):
    # Only the private-card chance node exists in one-round Leduc.
    outcomes, legals, _ = self.generate_all_private_card_nodes(game_state)
    action_oh = jax.nn.one_hot(actions[1], self.private_chance_outcomes)
    outcome = jax.tree_util.tree_map(
        lambda x: jnp.sum(x * jnp.reshape(action_oh, (action_oh.shape[0],) + (1,) * len(x.shape[1:])), axis=0),
        outcomes)
    legals = jnp.sum(action_oh[..., None, None] * legals, axis=0)
    if self.max_raises_per_round == 0:
      # No betting round: resolve immediately by private card rank.
      # self is static so this branch is chosen at JIT trace time.
      player_card_types = jnp.floor_divide(outcome.private_cards, 2)
      tie = jnp.all(jnp.isclose(player_card_types[0], player_card_types[1]))
      winner = jnp.argmax(player_card_types)
      reward = jnp.where(tie, jnp.array(0.0),
                         ((1.0 - 2.0 * winner) * outcome.current_chips[1 - winner])
                         / self.max_bet_amount).astype(jnp.float32)
      terminal = jnp.array(True)
      new_game_state = LeducGameState(
          action_history=outcome.action_history,
          public_card=outcome.public_card.astype(jnp.int16),
          private_cards=outcome.private_cards.astype(jnp.int16),
          current_chips=outcome.current_chips.astype(jnp.int16),
          turns_this_round=outcome.turns_this_round.astype(jnp.int16),
          terminal=terminal,
          turn=outcome.turn.astype(int),
          is_chance=jnp.array(False))
      return new_game_state, terminal, reward, legals
    new_game_state = LeducGameState(
        action_history=outcome.action_history,
        public_card=outcome.public_card.astype(jnp.int16),
        private_cards=outcome.private_cards.astype(jnp.int16),
        current_chips=outcome.current_chips.astype(jnp.int16),
        turns_this_round=outcome.turns_this_round.astype(jnp.int16),
        terminal=outcome.terminal.astype(bool),
        turn=outcome.turn.astype(int),
        is_chance=outcome.is_chance.astype(bool))
    return new_game_state, jnp.array(False), jnp.array(0, dtype=jnp.float32), legals

  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action_no_chance(self, game_state: LeducGameState, actions: chex.Array):
    oh_actions = jax.nn.one_hot(actions, self.num_actions)
    oh_turn = jax.nn.one_hot(game_state.turn, self.max_turns)
    fold_oh = jax.nn.one_hot(FOLD_ID, self.num_actions)
    raise_oh = jax.nn.one_hot(RAISE_ID, self.num_actions)

    current_player = game_state.turns_this_round % 2
    max_chips = jnp.max(game_state.current_chips)
    oh_valid_action = jax.nn.one_hot(actions[current_player] - 1, self.num_actions - 1)

    # Winner by private card rank only (no public card matching)
    player_card_types = jnp.floor_divide(game_state.private_cards, 2)
    folded = jnp.any(oh_actions[current_player] * fold_oh)
    raised = jnp.sum(oh_actions * raise_oh, axis=1)
    tie = jnp.all(jnp.isclose(player_card_types[0], player_card_types[1]))
    winner = jnp.argmax(player_card_types)
    winner = jnp.where(folded, 1 - current_player, winner)

    this_turn_played = oh_valid_action * oh_turn[..., None]
    action_history = game_state.action_history + this_turn_played

    num_raises = jnp.where(game_state.turn > 0,
                            action_history[game_state.turn - 1, RAISE_ID - 1] + raised[current_player], 0)

    action_chips = jnp.concatenate([
        jnp.repeat(game_state.current_chips[..., None], 2, axis=1),
        jnp.array([max_chips, max_chips])[..., None] * jnp.ones(2)], axis=1)
    current_chips = jnp.sum(action_chips * oh_actions, axis=1)
    current_chips = current_chips + raised * self.raise_amount

    bets_equal = jnp.all(jnp.isclose(current_chips[0], current_chips[1]))

    new_acting_legals = jnp.ones(self.num_actions) - self.invalid_action_mask
    new_acting_legals = jnp.where(num_raises < self.max_raises_per_round,
                                  new_acting_legals, new_acting_legals - raise_oh)
    new_acting_legals = jnp.where(bets_equal, new_acting_legals - fold_oh, new_acting_legals)
    next_player = (game_state.turns_this_round + 1) % 2
    new_legals = jnp.where(
        next_player == 0,
        jnp.stack([new_acting_legals, self.invalid_action_mask], axis=0),
        jnp.stack([self.invalid_action_mask, new_acting_legals], axis=0))

    # Terminal: fold, or bets equal after at least one exchange (turns_this_round >= 1)
    terminal = jnp.logical_or(folded,
                               jnp.logical_and(game_state.turns_this_round >= 1, bets_equal))
    terminal = jnp.squeeze(terminal)

    reward = jnp.where(terminal,
                       jnp.where(jnp.logical_and(tie, ~folded), 0,
                                 ((1 - 2 * winner) * current_chips[1 - winner]) / self.max_bet_amount),
                       0)
    terminal = jnp.logical_or(game_state.terminal, terminal)

    new_game_state = LeducGameState(
        action_history=action_history,
        public_card=game_state.public_card.astype(jnp.int16),
        private_cards=game_state.private_cards.astype(jnp.int16),
        current_chips=current_chips.astype(jnp.int16),
        turns_this_round=(game_state.turns_this_round + 1).astype(jnp.int16),
        terminal=terminal.astype(bool),
        turn=game_state.turn + 1,
        is_chance=jnp.array(False))
    return new_game_state, terminal, reward[0], new_legals

  @functools.partial(jax.jit, static_argnums=(0))
  def get_outcomes_and_probs(self, game_state: LeducGameState):
    outcomes = jnp.stack([jax.nn.one_hot(0, self.private_chance_outcomes),
                          jnp.arange(self.private_chance_outcomes)], axis=-1)
    probs = jax.lax.cond(
        game_state.is_chance,
        lambda s: jnp.ones(self.private_chance_outcomes) / self.private_chance_outcomes,
        lambda s: jnp.zeros(self.private_chance_outcomes),
        game_state)
    return outcomes, probs


def main():
  game = JaxLeduc()
  def _tree_walk(state: LeducGameState, legals, terminal, depth=0):
    legals = np.asarray(legals)
    print(f"In state: {state}")
    if terminal:
      return
    if game.is_chance(state):
      outcomes,  probs = game.get_outcomes_and_probs(state)
      # print(f"Next states: {next_states}")
      # print(f"Probs: {probs}")
      for outcome, prob in zip(outcomes, probs):
        if prob < 1e-5:
          continue
        new_state, new_terminal, reward, new_legals = game.apply_action(state, outcome)
        _tree_walk(new_state, new_legals, False, depth=depth + 1)
      return
    for a1i, a1 in enumerate(legals[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals[1]):
        if a2 < 0.5:
          continue
        joint_action = jnp.array([a1i, a2i])
        new_state, new_terminal, reward, new_legals = game.apply_action(state, joint_action)
        _tree_walk(new_state, new_legals, new_terminal, depth = depth + 1)
  init_state, init_legals = game.initialize_structures()
  _tree_walk(init_state, init_legals, False)

if __name__ == "__main__":
  main()