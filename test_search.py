import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search import build_toba_graph, uniform_cost_search, a_star_search, heuristic


def test_ucs_finds_optimal_cost():
    graph = build_toba_graph()
    result = uniform_cost_search(graph, "Balige", ["Desa_Ulos_Meat", "Bukit_Pahoda", "Tomok"], budget=100.0)
    assert result is not None
    assert math.isclose(result.g_cost, 70.0, rel_tol=1e-6)


def test_astar_matches_ucs_optimal_cost():
    graph = build_toba_graph()
    ucs = uniform_cost_search(graph, "Balige", ["Desa_Ulos_Meat", "Bukit_Pahoda", "Tomok"], budget=100.0)
    astar = a_star_search(graph, "Balige", ["Desa_Ulos_Meat", "Bukit_Pahoda", "Tomok"], budget=100.0)
    assert ucs is not None and astar is not None
    assert math.isclose(ucs.g_cost, astar.g_cost, rel_tol=1e-6)


def test_infeasible_budget_returns_none():
    graph = build_toba_graph()
    result = uniform_cost_search(graph, "Balige", ["Desa_Ulos_Meat", "Bukit_Pahoda", "Tomok"], budget=5.0)
    assert result is None


def test_heuristic_is_admissible_zero_at_goal():
    graph = build_toba_graph()
    h = heuristic(graph, "Tomok", frozenset())
    assert h == 0.0


def test_heuristic_never_negative():
    graph = build_toba_graph()
    for node in graph.adjacency:
        h = heuristic(graph, node, frozenset(["Bukit_Pahoda"]))
        assert h >= 0.0
