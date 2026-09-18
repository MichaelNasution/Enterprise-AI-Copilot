"""
Toba Copilot — Baseline Search Module
Milestone 1: Formulasi Ruang Keadaan & Algoritma Search (UCS / A*)

Merepresentasikan graf rute wisata Danau Toba sebagai masalah pencarian
ruang keadaan (state-space search) untuk menyusun itinerary multi-destinasi
dengan biaya (jarak/waktu tempuh) riil minimum, tunduk pada batas anggaran.

State  x = (current_node, frozenset(visited_targets), remaining_budget)
Goal   : visited_targets == target_set
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

Node = str
State = Tuple[Node, FrozenSet[Node], float]


# ---------------------------------------------------------------------------
# 1. Representasi Graf Masalah Bisnis (G = (V, E))
# ---------------------------------------------------------------------------

class TobaGraph:
    """Graf berarah berbobot biaya riil (jarak km) antar node wisata Toba."""

    def __init__(self) -> None:
        # koordinat (lat, lon) tiap node — dipakai heuristik Haversine untuk A*
        self.coordinates: Dict[Node, Tuple[float, float]] = {}
        # daftar ketetanggaan: node -> [(tetangga, bobot_km), ...]
        self.adjacency: Dict[Node, List[Tuple[Node, float]]] = {}

    def add_node(self, name: Node, lat: float, lon: float) -> None:
        self.coordinates[name] = (lat, lon)
        self.adjacency.setdefault(name, [])

    def add_edge(self, a: Node, b: Node, cost_km: float, bidirectional: bool = True) -> None:
        self.adjacency.setdefault(a, []).append((b, cost_km))
        if bidirectional:
            self.adjacency.setdefault(b, []).append((a, cost_km))

    def neighbors(self, node: Node) -> List[Tuple[Node, float]]:
        return self.adjacency.get(node, [])

    def haversine_km(self, a: Node, b: Node) -> float:
        """Jarak garis lurus (great-circle) antar dua node — dasar heuristik admissible."""
        lat1, lon1 = self.coordinates[a]
        lat2, lon2 = self.coordinates[b]
        r = 6371.0  # radius bumi (km)
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
        return 2 * r * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
# 2. Node Pencarian & Struktur Pendukung
# ---------------------------------------------------------------------------

@dataclass(order=True)
class SearchNode:
    priority: float
    g_cost: float = field(compare=False)
    state: State = field(compare=False)
    path: Tuple[Node, ...] = field(compare=False)


def reconstruct_path_str(path: Tuple[Node, ...]) -> str:
    return " -> ".join(path)
# ---------------------------------------------------------------------------
# 3. Uniform Cost Search (UCS)
# ---------------------------------------------------------------------------

def uniform_cost_search(
    graph: TobaGraph,
    start: Node,
    targets: List[Node],
    budget: float,
) -> Optional[SearchNode]:
    """
    UCS: mengekspansi node dengan g(n) (biaya kumulatif riil) terkecil
    menggunakan priority queue (heapq). Optimal untuk graf berbobot non-negatif.
    """
    target_set: FrozenSet[Node] = frozenset(targets)
    start_state: State = (start, frozenset(), budget)

    frontier: List[SearchNode] = []
    heapq.heappush(frontier, SearchNode(priority=0.0, g_cost=0.0, state=start_state, path=(start,)))

    best_cost: Dict[State, float] = {start_state: 0.0}
    expanded = 0

    while frontier:
        current = heapq.heappop(frontier)
        expanded += 1
        node, visited, remaining_budget = current.state

        if visited == target_set:
            current.path = current.path + (f"[expanded={expanded}]",)  # type: ignore
            return current

        if current.g_cost > best_cost.get(current.state, math.inf):
            continue  # entri usang (stale), lewati

        for neighbor, cost in graph.neighbors(node):
            if cost > remaining_budget:
                continue  # ACTIONS(x): aksi tidak valid jika melebihi sisa anggaran

            new_g = current.g_cost + cost
            new_visited = visited | ({neighbor} & target_set)
            new_state: State = (neighbor, new_visited, remaining_budget - cost)

            if new_g < best_cost.get(new_state, math.inf):
                best_cost[new_state] = new_g
                heapq.heappush(
                    frontier,
                    SearchNode(priority=new_g, g_cost=new_g, state=new_state, path=current.path + (neighbor,)),
                )

    return None  # tidak ditemukan solusi dalam batas anggaran

# ---------------------------------------------------------------------------
# 4. A* Search
# ---------------------------------------------------------------------------

def heuristic(graph: TobaGraph, node: Node, remaining_targets: FrozenSet[Node]) -> float:
    """
    h(n) = jarak Haversine (garis lurus) terpendek dari node saat ini
    ke destinasi tersisa TERDEKAT. Admissible karena garis lurus <= jarak jalan riil,
    sehingga h(n) tidak pernah melebihi biaya sisa sebenarnya.
    """
    if not remaining_targets:
        return 0.0
    return min(graph.haversine_km(node, t) for t in remaining_targets)
