import jax
import chex
import jax.numpy as jnp

import functools
from games.jax_game import JaxGame, GameState, InformationType



@chex.dataclass(frozen=True)
class GoofspielGameState(GameState):
    point_cards: chex.Array
    played_cards: chex.Array
    p1_points: chex.Array
    turn: int


class JaxGoofspiel(JaxGame):
  def __init__(self, cards, points_order="descending", turns=-1, reward_type: str = "clip", observation_only: bool = False) -> None:
    self.cards = cards
    self.max_turns = turns
    if turns <= 0:
      self.max_turns = cards
    self.points_order = points_order
    self.reward_type_name = reward_type
    self.reward_type = 0 if reward_type == "clip" else 1
    self.observation_only = observation_only
    
  
  def game_name(self):
    return "goofspiel"

  def params_dict(self):
    d = {"num_cards": self.cards, "turns": self.max_turns, "points_order": self.points_order,
         "reward_type": self.reward_type_name}
    if self.observation_only:
      d["observation_only"] = "obs"
    return d

  def information_type(self):
    return InformationType.IIG

  def num_players(self):
    return 2

  def state_tensor_shape(self):
    return self.max_turns * self.cards + self.max_turns * 2 + self.max_turns * self.cards * 3
   
  def information_state_tensor_shape(self):
    return self.max_turns * self.cards + self.max_turns * 2 + self.max_turns * self.cards * 2 + 2
  
  def observation_tensor_shape(self):
    if self.observation_only:
      return self.cards + 4
    return self.information_state_tensor_shape()

  def public_state_tensor_shape(self):
    return self.max_turns * self.cards + self.max_turns * 2 + self.max_turns * self.cards

  def num_distinct_actions(self):
    return self.cards

  def max_trajectory_length(self):
    return self.max_turns
  
  
  @functools.partial(jax.jit, static_argnums=(0))
  def initialize_structures(self):
    if self.points_order == "descending":
      point_cards = jnp.arange(self.cards, self.cards - self.max_turns, -1)
    if self.points_order == "ascending":
      point_cards = jnp.arange(1, 1 + self.max_turns)
    played_cards = jnp.zeros((2, self.max_turns, self.cards))
    p1_points = jnp.zeros(self.max_turns)
    game_state= GoofspielGameState(point_cards=point_cards, played_cards=played_cards, p1_points=p1_points, turn = 0)
    return game_state, jnp.ones((2, self.cards))

  
  # State Tensor -> Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card], Played Cards [Player, Turn, Card], 
  # Iset tensor -> Observing Player, Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card], Played Cards [Turn, Card],
  # Public tensor -> Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card]
  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state:GoofspielGameState):
    played_turns_mask = jnp.sum(game_state.played_cards[0], -1)
    # To set the first to 
    played_turns_mask = jnp.roll(played_turns_mask, 1, axis=0) + jax.nn.one_hot(0, self.max_turns)
    # Every card that is played have value >= 1, non-played has 0. So we just subtract 1 to make sure everything works with one-hot (-1 is all zeros)
    point_cards_masked = game_state.point_cards * played_turns_mask - 1  
    oh_point_cards = jax.nn.one_hot(point_cards_masked, self.cards)
    
    # Tie -1, P1 win 0, P2 win 1
    p2_winned = jnp.where(game_state.p1_points < 0, 1, 0) - (game_state.p1_points == 0)
    winner = jax.nn.one_hot(p2_winned, 2)
    
    tie_cards = jnp.expand_dims(((game_state.p1_points == 0) * played_turns_mask), -1) * game_state.played_cards[0]
    
    public_state_tensor = jnp.concatenate([jnp.ravel(oh_point_cards), jnp.ravel(winner), jnp.ravel(tie_cards)], axis=0)
    
    p1_player = jax.nn.one_hot(0, 2)
    
    p1_infoset_tensor = jnp.concatenate([p1_player, public_state_tensor, jnp.ravel(game_state.played_cards[0])], axis=0)
    p2_infoset_tensor = jnp.concatenate([1 - p1_player, public_state_tensor, jnp.ravel(game_state.played_cards[1])], axis=0)
    
    state_tensor = jnp.concatenate([public_state_tensor, jnp.ravel(game_state.played_cards)], axis=0)

    if self.observation_only:
      oh_current_point = jax.nn.one_hot(game_state.point_cards[game_state.turn] - 1, self.cards)
      prev_idx = jnp.maximum(game_state.turn - 1, 0)
      prev_points = game_state.p1_points[prev_idx]
      p2_won = jnp.where(prev_points < 0, 1, 0) - (prev_points == 0)
      prev_winner = jax.nn.one_hot(p2_won, 2)
      p1_player = jax.nn.one_hot(0, 2)
      public_observation = jnp.concatenate([oh_current_point, prev_winner], axis=0)
      p1_observation = jnp.concatenate([p1_player, public_observation], axis=0)
      p2_observation = jnp.concatenate([1 - p1_player, public_observation], axis=0)
      return state_tensor, p1_observation, p2_observation, public_observation

    return state_tensor, p1_infoset_tensor, p2_infoset_tensor, public_state_tensor
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, game_state:GoofspielGameState, actions):
    oh_actions = jax.nn.one_hot(actions, self.cards) 
    oh_turn = jax.nn.one_hot(game_state.turn, self.max_turns)
    
    winner = jnp.argmax(actions, axis=-1)
    loser = jnp.argmin(actions, axis=-1)
    tie = winner == loser
    
    # Point cards are from 0 to N-1, but points should be from 1 to N
    point = game_state.point_cards[..., game_state.turn] * oh_turn
    
    this_turn_played = oh_actions[..., None, :] * oh_turn[None, :, None]
    
    played_cards = game_state.played_cards + this_turn_played
    
    p1_points = jnp.where(tie, game_state.p1_points, jnp.where(winner == 0, game_state.p1_points + point, game_state.p1_points - point))
    
    legal_actions = 1 - jnp.sum(played_cards, 1)
    
    next_action = jnp.argmax(legal_actions, -1)
    next_winner = jnp.argmax(next_action)
    next_loser = jnp.argmin(next_action)
    next_tie = next_winner == next_loser
    next_point = game_state.point_cards[..., game_state.turn+1] * jax.nn.one_hot(game_state.turn+1, self.max_turns)
    
    # When all cards are played, the last turn is forced and resolved automatically.
    # Otherwise the game simply ends after max_turns played turns.
    forced_last_turn = self.max_turns == self.cards
    last_turn = self.max_turns - 2 if forced_last_turn else self.max_turns - 1
    if forced_last_turn:
      p1_points = jnp.where(game_state.turn != last_turn, p1_points, jnp.where(next_tie, p1_points, jnp.where(next_winner == 0, p1_points + next_point, p1_points - next_point)))
    
    rewards = jnp.where(self.reward_type == 0, jnp.clip(jnp.sum(p1_points), -1, 1), jnp.sum(p1_points))
    
    rewards = jnp.where(game_state.turn != last_turn, 0, rewards)
    terminal = game_state.turn >= last_turn
    
    # rewards = jnp.sum(p1_points)
    # if turn == self.cards-1:
    #   actions = jnp.argmax(legal_actions, -1)
    #   return self.apply_action(point_cards, played_cards, p1_points, turn+1, actions)
    game_state= GoofspielGameState(point_cards=game_state.point_cards, played_cards=played_cards, p1_points=p1_points, turn=game_state.turn + 1)

    
    return game_state, terminal, rewards, legal_actions


@chex.dataclass(frozen=True)
class GoofspielRandomGameState(GameState):
    point_cards: chex.Array
    played_cards: chex.Array
    p1_points: chex.Array
    turn: int
    is_chance: chex.Array

class JaxRandomGoofspiel(JaxGame):
  def __init__(self, cards, reward_type: str = "clip", observation_only: bool = False) -> None:
    """A classic variant of Goofspiel, where the point cards are actually dealt via a chance node.
    And are fully observable. Strategically, it is not harder than the descending variant, but
    it is much larger game.
    """
    self.cards = cards
    self.max_turns = cards
    self.reward_type_name = reward_type
    self.reward_type = 0 if reward_type == "clip" else 1
    self.observation_only = observation_only
    
  
  def game_name(self):
    return "goofspiel_random"
  
  def params_dict(self):
    d = {"num_cards": self.cards, "reward_type": self.reward_type_name}
    if self.observation_only:
      d["observation_only"] = "obs"
    return d
  
  def information_type(self):
    return InformationType.IIG
  
  def num_players(self):
    return 2

  def state_tensor_shape(self):
    return self.max_turns * self.cards + self.max_turns * 2 + self.max_turns * self.cards * 3
   
  def information_state_tensor_shape(self):
    return self.max_turns * self.cards + self.max_turns * 2 + self.max_turns * self.cards * 2 + 2
  
  def observation_tensor_shape(self):
    if self.observation_only:
      return self.cards + 4
    return self.information_state_tensor_shape()
  
  def public_state_tensor_shape(self):
    return self.max_turns * self.cards + self.max_turns * 2 + self.max_turns * self.cards
  
  def num_distinct_actions(self):
    return self.cards
  
  def max_chance_outcomes(self):
    return self.cards
  
  def is_chance(self, game_state: GoofspielRandomGameState):
    return game_state.is_chance
  
  def depth_chance_valid_outcomes(self, depth:int):
    if depth % 2 == 0 and depth < (2 * (self.cards - 1)):
      return self.cards - (depth // 2)
    return 1
  
  def depth_chance_outcomes(self, depth):
    if depth % 2 == 0 and depth < (2 * (self.cards - 1)):
      return self.cards
    return 1
  
  def max_trajectory_length(self):
    #Includes the terminal node, and chance node for each
    # point card, except the last one. 
    return 2 * self.max_turns - 1
  

  def max_trajectory_length_no_chance(self):
    return self.max_turns
  

  @functools.partial(jax.jit, static_argnums=(0))
  def get_outcomes_and_probs(self, game_state:GoofspielRandomGameState) -> tuple[chex.Array, chex.Array]:
    outcomes = jnp.stack([jnp.zeros(self.cards), jnp.arange(self.cards)], axis=-1)
    def invalid_probs(game_state):
      return jnp.zeros(self.cards)

    def chance_probs(game_state:GoofspielRandomGameState):
      already_dealt = game_state.point_cards.sum(axis=0)
      available_point_cards = jnp.ones(self.cards) - already_dealt
      return available_point_cards / jnp.sum(available_point_cards)

    probs = jax.lax.cond(game_state.is_chance, chance_probs, invalid_probs, game_state)
    return outcomes, probs
  
  
  @functools.partial(jax.jit, static_argnums=(0))
  def initialize_structures(self):
    point_cards = jnp.zeros((self.max_turns, self.cards))
    played_cards = jnp.zeros((2, self.max_turns, self.cards))
    p1_points = jnp.zeros(self.max_turns)
    game_state= GoofspielRandomGameState(point_cards=point_cards, played_cards=played_cards, p1_points=p1_points, turn = 0, is_chance= jnp.array(True))
    return game_state, jnp.ones((2, self.cards))

  
  # State Tensor -> Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card], Played Cards [Player, Turn, Card],
  # Iset tensor -> Observing Player, Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card], Played Cards [Turn, Card],
  # Public tensor -> Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card]
  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state:GoofspielRandomGameState):
    played_turns_mask = jnp.sum(game_state.played_cards[0], -1)
    played_turns_mask = jnp.roll(played_turns_mask, 1, axis=0) + jax.nn.one_hot(0, self.max_turns)

    # Tie -1, P1 win 0, P2 win 1
    p2_winned = jnp.where(game_state.p1_points < 0, 1, 0) - (game_state.p1_points == 0)
    winner = jax.nn.one_hot(p2_winned, 2)

    tie_cards = jnp.expand_dims(((game_state.p1_points == 0) * played_turns_mask), -1) * game_state.played_cards[0]

    # point_cards is already [max_turns, cards] one-hot encoded
    public_state_tensor = jnp.concatenate([jnp.ravel(game_state.point_cards), jnp.ravel(winner), jnp.ravel(tie_cards)], axis=0)

    p1_player = jax.nn.one_hot(0, 2)

    p1_infoset_tensor = jnp.concatenate([p1_player, public_state_tensor, jnp.ravel(game_state.played_cards[0])], axis=0)
    p2_infoset_tensor = jnp.concatenate([1 - p1_player, public_state_tensor, jnp.ravel(game_state.played_cards[1])], axis=0)

    state_tensor = jnp.concatenate([public_state_tensor, jnp.ravel(game_state.played_cards)], axis=0)

    if self.observation_only:
      # point_cards is one-hot, the current point card is its row of the current turn.
      current_point = game_state.point_cards[game_state.turn]
      prev_idx = jnp.maximum(game_state.turn - 1, 0)
      prev_points = game_state.p1_points[prev_idx]
      p2_won = jnp.where(prev_points < 0, 1, 0) - (prev_points == 0)
      prev_winner = jax.nn.one_hot(p2_won, 2)
      public_state_tensor = jnp.concatenate([current_point, prev_winner], axis=0)
      p1_infoset_tensor = jnp.concatenate([p1_player, public_state_tensor], axis=0)
      p2_infoset_tensor = jnp.concatenate([1 - p1_player, public_state_tensor], axis=0)

    # Zero out tensors at chance nodes
    state_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(state_tensor), state_tensor)
    p1_infoset_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(p1_infoset_tensor), p1_infoset_tensor)
    p2_infoset_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(p2_infoset_tensor), p2_infoset_tensor)
    public_state_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(public_state_tensor), public_state_tensor)

    return state_tensor, p1_infoset_tensor, p2_infoset_tensor, public_state_tensor

  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, game_state:GoofspielRandomGameState, actions):
    return jax.lax.cond(game_state.is_chance, self.apply_action_chance, self.apply_action_no_chance, game_state, actions)

  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action_chance(self, game_state:GoofspielRandomGameState, actions):
    card_index = actions[1].astype(jnp.int32)
    oh_card = jax.nn.one_hot(card_index, self.cards)
    oh_turn = jax.nn.one_hot(game_state.turn, self.max_turns)

    # Set the point card for current turn (outer product: [max_turns, 1] * [1, cards])
    point_cards = game_state.point_cards + oh_turn[:, None] * oh_card[None, :]

    legal_actions = 1 - jnp.sum(game_state.played_cards, 1)

    new_state = GoofspielRandomGameState(
        point_cards=point_cards,
        played_cards=game_state.played_cards,
        p1_points=game_state.p1_points,
        turn=game_state.turn,
        is_chance=jnp.array(False)
    )
    return new_state, jnp.array(False), jnp.array(0.0), legal_actions

  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action_no_chance(self, game_state:GoofspielRandomGameState, actions):
    oh_actions = jax.nn.one_hot(actions, self.cards)
    oh_turn = jax.nn.one_hot(game_state.turn, self.max_turns)

    winner = jnp.argmax(actions, axis=-1)
    loser = jnp.argmin(actions, axis=-1)
    tie = winner == loser

    # Point value from one-hot encoded point card (card index + 1 = point value)
    point_value = jnp.argmax(game_state.point_cards[game_state.turn]) + 1
    point = point_value * oh_turn

    this_turn_played = oh_actions[..., None, :] * oh_turn[None, :, None]
    played_cards = game_state.played_cards + this_turn_played

    p1_points = jnp.where(tie, game_state.p1_points, jnp.where(winner == 0, game_state.p1_points + point, game_state.p1_points - point))

    legal_actions = 1 - jnp.sum(played_cards, 1)

    # Handle last turn: auto-resolve the final point card (only 1 card left, deterministic)
    dealt_cards = game_state.point_cards.sum(axis=0)
    remaining_point_card = 1 - dealt_cards
    last_point_value = jnp.sum(jnp.arange(1, self.cards + 1) * remaining_point_card)

    next_action = jnp.argmax(legal_actions, -1)
    next_winner = jnp.argmax(next_action)
    next_loser = jnp.argmin(next_action)
    next_tie = next_winner == next_loser
    next_point = last_point_value * jax.nn.one_hot(game_state.turn + 1, self.max_turns)

    p1_points = jnp.where(game_state.turn != self.cards - 2, p1_points, jnp.where(next_tie, p1_points, jnp.where(next_winner == 0, p1_points + next_point, p1_points - next_point)))

    rewards = jnp.where(self.reward_type == 0, jnp.clip(jnp.sum(p1_points), -1, 1), jnp.sum(p1_points))
    rewards = jnp.where(game_state.turn != self.cards - 2, 0, rewards)
    terminal = game_state.turn >= self.cards - 2

    # After an action, the next state is a chance node (unless terminal)
    new_state = GoofspielRandomGameState(
        point_cards=game_state.point_cards,
        played_cards=played_cards,
        p1_points=p1_points,
        turn=game_state.turn + 1,
        is_chance=~terminal
    )
    return new_state, terminal, rewards, legal_actions
  
def main():
  import numpy as np
  game = JaxRandomGoofspiel(cards=3)
  terminal_count = [0]
  reward_sum = [0.0]

  def _tree_walk(state: GoofspielRandomGameState, legals, terminal, depth=0):
    if terminal:
      terminal_count[0] += 1
      return
    if game.is_chance(state):
      outcomes, probs = game.get_outcomes_and_probs(state)
      for outcome, prob in zip(outcomes, probs):
        if prob < 1e-5:
          continue
        new_state, new_terminal, reward, new_legals = game.apply_action(state, outcome)
        _tree_walk(new_state, new_legals, False, depth=depth + 1)
      return
    legals_np = np.asarray(legals)
    for a1i, a1 in enumerate(legals_np[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals_np[1]):
        if a2 < 0.5:
          continue
        joint_action = jnp.array([a1i, a2i])
        new_state, new_terminal, reward, new_legals = game.apply_action(state, joint_action)
        reward_val = float(reward)
        if new_terminal:
          reward_sum[0] += reward_val
        _tree_walk(new_state, new_legals, new_terminal, depth=depth + 1)

  init_state, init_legals = game.initialize_structures()
  _tree_walk(init_state, init_legals, False)
  print(f"Terminal states: {terminal_count[0]}")
  print(f"Sum of rewards across all terminal states: {reward_sum[0]}")
  print(f"Average reward: {reward_sum[0] / terminal_count[0]:.6f}")

if __name__ == '__main__':
  main()