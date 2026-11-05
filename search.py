"""
Toba Copilot — Baseline Search Module
Milestone 1: Formulasi Ruang Keadaan & Algoritma Search (UCS / A*)

Merepresentasikan graf rute wisata Danau Toba sebagai masalah pencarian
ruang keadaan (state-space search) untuk menyusun itinerary multi-destinasi
dengan biaya (jarak/waktu tempuh) riil minimum, tunduk pada batas anggaran.

State   x = (current_node, frozenset(visited_targets))
Aksi    : pindah ke tetangga n' jika g(n) + cost(n, n') <= budget
Goal    : visited_targets == target_set
Biaya   : g(n) = akumulasi jarak (km) sepanjang lintasan

Catatan formulasi: anggaran BUKAN komponen state. Sisa anggaran selalu sama dengan
budget - g(n), sehingga memasukkannya ke state membuat setiap lintasan dengan biaya
berbeda menjadi state berbeda dan deduplikasi state tidak pernah bekerja. Anggaran
ditegakkan sebagai syarat validitas aksi pada g(n).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

Node = str
State = Tuple[Node, FrozenSet[Node]]


# ---------------------------------------------------------------------------
# 1. Representasi Graf Masalah Bisnis (G = (V, E))
# ---------------------------------------------------------------------------

class TobaGraph:
    """Graf berbobot biaya riil (jarak km) antar node wisata Toba.

    Edge default dua arah (bidirectional); edge satu arah dapat dibuat dengan
    ``bidirectional=False``.
    """

    def __init__(self) -> None:
        # koordinat (lat, lon) tiap node — dipakai heuristik Haversine untuk A*
        self.coordinates: Dict[Node, Tuple[float, float]] = {}
        # daftar ketetanggaan: node -> [(tetangga, bobot_km), ...]
        self.adjacency: Dict[Node, List[Tuple[Node, float]]] = {}
        # cache rasio skala heuristik; di-reset setiap kali graf berubah
        self._ratio_cache: Optional[float] = None

    def add_node(self, name: Node, lat: float, lon: float) -> None:
        self.coordinates[name] = (lat, lon)
        self.adjacency.setdefault(name, [])
        self._ratio_cache = None

    def add_edge(self, a: Node, b: Node, cost_km: float, bidirectional: bool = True) -> None:
        self.adjacency.setdefault(a, []).append((b, cost_km))
        if bidirectional:
            self.adjacency.setdefault(b, []).append((a, cost_km))
        self._ratio_cache = None

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

    def min_edge_to_haversine_ratio(self) -> float:
        """Rasio terkecil bobot_edge / haversine(edge), dibatasi maksimal 1.0.

        Bobot edge pada graf ini tidak selalu >= jarak garis lurus antar
        koordinatnya (mis. Balige–Muara: edge 12.0 km, Haversine 16.03 km).
        Rasio ini dipakai untuk menskalakan heuristik agar tetap admissible.
        """
        if self._ratio_cache is None:
            ratio = 1.0
            for a, edges in self.adjacency.items():
                for b, cost in edges:
                    straight = self.haversine_km(a, b)
                    if straight > 0:
                        ratio = min(ratio, cost / straight)
            self._ratio_cache = ratio
        return self._ratio_cache


# ---------------------------------------------------------------------------
# 2. Node Pencarian & Struktur Pendukung
# ---------------------------------------------------------------------------

@dataclass(order=True)
class SearchNode:
    priority: float
    g_cost: float = field(compare=False)
    state: State = field(compare=False)
    path: Tuple[Node, ...] = field(compare=False)
    # jumlah node yang diekspansi hingga solusi ditemukan (diisi saat goal)
    expanded: int = field(default=0, compare=False)


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
    start_state: State = (start, frozenset({start}) & target_set)

    frontier: List[SearchNode] = []
    heapq.heappush(frontier, SearchNode(priority=0.0, g_cost=0.0, state=start_state, path=(start,)))

    best_cost: Dict[State, float] = {start_state: 0.0}
    expanded = 0

    while frontier:
        current = heapq.heappop(frontier)
        if current.g_cost > best_cost.get(current.state, math.inf):
            continue  # entri usang (stale), lewati

        expanded += 1
        node, visited = current.state

        if visited == target_set:
            current.expanded = expanded
            return current

        for neighbor, cost in graph.neighbors(node):
            new_g = current.g_cost + cost
            if new_g > budget:
                continue  # ACTIONS(x): aksi tidak valid jika total biaya melebihi anggaran

            new_visited = visited | ({neighbor} & target_set)
            new_state: State = (neighbor, new_visited)

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
    h(n) = rho * min_{t in sisa target} haversine(n, t),
    dengan rho = graph.min_edge_to_haversine_ratio() (0 < rho <= 1).

    Bukti admissible: untuk setiap edge e = (a, b) berlaku w(e) >= rho * hav(a, b)
    menurut definisi rho. Untuk lintasan apa pun dari n ke target t,
        panjang = sum w(e) >= rho * sum hav(e) >= rho * hav(n, t)
    (langkah terakhir: ketidaksamaan segitiga pada jarak great-circle). Biaya sisa
    minimal harus mencapai setidaknya satu target tersisa, sehingga h(n) <= h*(n).

    Mengapa perlu rho: bobot edge pada graf ini tidak selalu >= jarak garis lurus
    koordinatnya. Tanpa rho (Haversine murni) heuristik tidak admissible dan A*
    terbukti dapat menghasilkan rute suboptimal (lihat test regresi di test_search.py).
    Dengan bobot jarak jalan riil yang konsisten dengan koordinat, rho = 1.0.
    """
    if not remaining_targets:
        return 0.0
    rho = graph.min_edge_to_haversine_ratio()
    return rho * min(graph.haversine_km(node, t) for t in remaining_targets)

def a_star_search(
    graph: TobaGraph,
    start: Node,
    targets: List[Node],
    budget: float,
) -> Optional[SearchNode]:
    """A*: f(n) = g(n) + h(n); h(n) admissible -> solusi tetap optimal."""
    target_set: FrozenSet[Node] = frozenset(targets)
    start_state: State = (start, frozenset({start}) & target_set)

    frontier: List[SearchNode] = []
    h0 = heuristic(graph, start, target_set - start_state[1])
    heapq.heappush(frontier, SearchNode(priority=h0, g_cost=0.0, state=start_state, path=(start,)))

    best_cost: Dict[State, float] = {start_state: 0.0}
    expanded = 0

    while frontier:
        current = heapq.heappop(frontier)
        if current.g_cost > best_cost.get(current.state, math.inf):
            continue  # entri usang (stale), lewati

        expanded += 1
        node, visited = current.state

        if visited == target_set:
            current.expanded = expanded
            return current

        for neighbor, cost in graph.neighbors(node):
            new_g = current.g_cost + cost
            if new_g > budget:
                continue

            new_visited = visited | ({neighbor} & target_set)
            new_state: State = (neighbor, new_visited)

            if new_g < best_cost.get(new_state, math.inf):
                best_cost[new_state] = new_g
                h = heuristic(graph, neighbor, target_set - new_visited)
                heapq.heappush(
                    frontier,
                    SearchNode(priority=new_g + h, g_cost=new_g, state=new_state, path=current.path + (neighbor,)),
                )

    return None


# ---------------------------------------------------------------------------
# 5. Contoh Graf Masalah Bisnis: Kawasan Danau Toba (subset destinasi)
# ---------------------------------------------------------------------------

def build_toba_graph() -> TobaGraph:
    g = TobaGraph()
    # (nama, lat, lon) -- koordinat perkiraan
    g.add_node("Balige", 2.3336, 99.0678)
    g.add_node("Muara", 2.2167, 98.9833)
    g.add_node("Parapat", 2.6587, 98.9350)
    g.add_node("Tomok", 2.6167, 98.8500)
    g.add_node("Bukit_Pahoda", 2.6700, 98.8800)
    g.add_node("Desa_Ulos_Meat", 2.3500, 99.0500)
    g.add_node("Ambarita", 2.6667, 98.8333)

    g.add_edge("Balige", "Muara", 12.0)
    g.add_edge("Balige", "Desa_Ulos_Meat", 4.5)
    g.add_edge("Muara", "Parapat", 38.0)
    g.add_edge("Desa_Ulos_Meat", "Parapat", 41.0)
    g.add_edge("Parapat", "Tomok", 15.0)   # penyeberangan danau
    g.add_edge("Tomok", "Ambarita", 6.0)
    g.add_edge("Tomok", "Bukit_Pahoda", 9.5)
    g.add_edge("Ambarita", "Bukit_Pahoda", 5.0)
    return g


if __name__ == "__main__":
    graph = build_toba_graph()

    start_node = "Balige"
    itinerary_targets = ["Desa_Ulos_Meat", "Bukit_Pahoda", "Tomok"]
    wisatawan_budget = 100.0  # satuan: km biaya tempuh maksimum (proxy anggaran)

    print("=== Uniform Cost Search (UCS) ===")
    result_ucs = uniform_cost_search(graph, start_node, itinerary_targets, wisatawan_budget)
    if result_ucs:
        print(f"Rute optimal : {reconstruct_path_str(result_ucs.path)}")
        print(f"Total biaya  : {result_ucs.g_cost:.2f} km")
        print(f"Node expanded: {result_ucs.expanded}")
    else:
        print("Tidak ditemukan rute dalam batas anggaran.")

    print("\n=== A* Search ===")
    result_astar = a_star_search(graph, start_node, itinerary_targets, wisatawan_budget)
    if result_astar:
        print(f"Rute optimal : {reconstruct_path_str(result_astar.path)}")
        print(f"Total biaya  : {result_astar.g_cost:.2f} km")
        print(f"Node expanded: {result_astar.expanded}")
        print(f"Skala heuristik (rho): {graph.min_edge_to_haversine_ratio():.4f}")
    else:
        print("Tidak ditemukan rute dalam batas anggaran.")
