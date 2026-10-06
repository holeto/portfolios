from games.jax_game import JaxGame, GameState, InformationType
from games.jax_goofspiel import JaxGoofspiel, JaxRandomGoofspiel
from games.jax_leduc import JaxLeduc, JaxLeducRound1
from games.jax_rps import JaxRPS, JaxStochasticRPS, JaxTurnBasedRPS


# Games are immutable, so one instance per name and parameters is shared: the functions jitted on a
# game (e.g. `tree_builder.game_functions`) are then compiled once for all the solvers using it.
_GAMES: dict[tuple, JaxGame] = {}


def make_game(name: str, **kwargs) -> JaxGame:
  """The game of the given name and parameters, shared by all the callers."""
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
  key = (name, tuple(sorted(kwargs.items())))
  if key not in _GAMES:
    _GAMES[key] = games[name](**kwargs)
  return _GAMES[key]
