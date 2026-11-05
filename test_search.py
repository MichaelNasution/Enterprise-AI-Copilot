import heapq
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


# ---------------------------------------------------------------------------
# Pengujian admissibility & optimalitas A* yang sesungguhnya
# ---------------------------------------------------------------------------

def _dijkstra(graph, source):
    """Jarak terpendek riil di graf dari source ke semua node (ground truth h*)."""
    dist = {source: 0.0}
    frontier = [(0.0, source)]
    while frontier:
        cost, node = heapq.heappop(frontier)
        if cost > dist[node]:
            continue
        for neighbor, weight in graph.neighbors(node):
            new_cost = cost + weight
            if new_cost < dist.get(neighbor, math.inf):
                dist[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
    return dist


def test_heuristic_admissible_all_pairs():
    """h(a, {b}) tidak boleh melebihi jarak terpendek riil a -> b untuk SEMUA pasangan."""
    graph = build_toba_graph()
    for a in graph.coordinates:
        dist = _dijkstra(graph, a)
        for b in graph.coordinates:
            h = heuristic(graph, a, frozenset([b]))
            assert h <= dist[b] + 1e-9, f"h({a}->{b})={h:.2f} > h*={dist[b]:.2f}"


def test_astar_matches_ucs_all_combinations():
    """A* harus seoptimal UCS pada setiap kombinasi start dan 1–3 target."""
    graph = build_toba_graph()
    nodes = list(graph.coordinates)
    for start in nodes:
        others = [n for n in nodes if n != start]
        for k in range(1, 4):
            for targets in itertools.combinations(others, k):
                ucs = uniform_cost_search(graph, start, list(targets), budget=500.0)
                astar = a_star_search(graph, start, list(targets), budget=500.0)
                assert (ucs is None) == (astar is None)
                if ucs is not None:
                    assert math.isclose(ucs.g_cost, astar.g_cost, rel_tol=1e-9), (
                        f"{start} -> {targets}: UCS={ucs.g_cost} A*={astar.g_cost}"
                    )


def test_regression_balige_muara_parapat():
    """Kasus yang dulu gagal: Haversine murni membuat A* memberi 69.5 km (optimal 59.0 km)."""
    graph = build_toba_graph()
    targets = ["Muara", "Parapat", "Desa_Ulos_Meat"]
    ucs = uniform_cost_search(graph, "Balige", targets, budget=500.0)
    astar = a_star_search(graph, "Balige", targets, budget=500.0)
    assert ucs is not None and astar is not None
    assert math.isclose(ucs.g_cost, 59.0, rel_tol=1e-9)
    assert math.isclose(astar.g_cost, 59.0, rel_tol=1e-9)


def test_state_excludes_budget_and_expansion_is_reported():
    """State = (node, visited); jumlah ekspansi dilaporkan lewat field, bukan di path."""
    graph = build_toba_graph()
    targets = ["Desa_Ulos_Meat", "Bukit_Pahoda", "Tomok"]
    ucs = uniform_cost_search(graph, "Balige", targets, budget=100.0)
    astar = a_star_search(graph, "Balige", targets, budget=100.0)
    assert ucs is not None and astar is not None
    assert len(ucs.state) == 2
    assert all(not str(step).startswith("[") for step in ucs.path)
    assert ucs.expanded == 11
    assert astar.expanded == 9
    assert astar.expanded <= ucs.expanded


def test_expansion_count_independent_of_loose_budget():
    """Anggaran longgar tidak boleh menggelembungkan ruang keadaan."""
    graph = build_toba_graph()
    targets = ["Desa_Ulos_Meat", "Bukit_Pahoda", "Tomok"]
    tight = uniform_cost_search(graph, "Balige", targets, budget=70.0)
    loose = uniform_cost_search(graph, "Balige", targets, budget=10_000.0)
    assert tight is not None and loose is not None
    assert math.isclose(tight.g_cost, 70.0) and math.isclose(loose.g_cost, 70.0)
    assert loose.expanded == tight.expanded


def test_budget_exactly_equal_to_optimum_is_feasible():
    """Kasus batas: budget tepat sama dengan biaya optimal tetap menghasilkan solusi."""
    graph = build_toba_graph()
    result = a_star_search(graph, "Balige", ["Desa_Ulos_Meat", "Bukit_Pahoda", "Tomok"], budget=70.0)
    assert result is not None
    assert math.isclose(result.g_cost, 70.0)
    assert a_star_search(graph, "Balige", ["Desa_Ulos_Meat", "Bukit_Pahoda", "Tomok"], budget=69.99) is None


def test_start_node_in_targets_counts_as_visited():
    """Jika start termasuk target, ia langsung dianggap sudah dikunjungi."""
    graph = build_toba_graph()
    result = uniform_cost_search(graph, "Balige", ["Balige"], budget=0.0)
    assert result is not None
    assert result.g_cost == 0.0
