"""Evaluation tests of trained models.

Each test is a module exposing:
  Config: a dataclass with the parameters of the test (the `eval.tests.<name>` config section).
  run(solver, game, search_config, test_config, cache) -> dict of results, where
    `search_config` is the `TestTimeSearchConfig` from the `eval.search` config section
    and `cache` is shared by all tests of a single evaluation (e.g. the full game tree).
New tests are added by registering them here.
"""
from evaluation import blueprint_exploitability, search_exploitability

EVALUATIONS = {
    "search_exploitability": search_exploitability,
    "blueprint_exploitability": blueprint_exploitability,
}

DEFAULT_TESTS = {"search_exploitability": {}}
