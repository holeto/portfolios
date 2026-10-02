from games.jax_game import JaxGame, GameState, InformationType
from games.jax_goofspiel import JaxGoofspiel, JaxRandomGoofspiel
from games.jax_leduc import JaxLeduc, JaxLeducRound1
from games.jax_rps import JaxRPS, JaxStochasticRPS, JaxTurnBasedRPS


def make_game(name: str, **kwargs) -> JaxGame:
  """Constructs a game by its name."""
  games = {
      "goofspiel": JaxGoofspiel,
      "goofspiel_random": JaxRandomGoofspiel,
      "leduc": JaxLeduc,
      "leduc_round1": JaxLeducRound1,
      "rps": JaxRPS,
      "stochastic_rps": JaxStochasticRPS,
      "turn_based_rps": JaxTurnBasedRPS,
  }
  if name not in games:
    raise ValueError(f"Unknown game {name}, available: {list(games)}")
  return games[name](**kwargs)
