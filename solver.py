"""
Toba Copilot — Business Constraint Solver Module
Milestone 2: Constraint Satisfaction Problem (CSP) dengan AC-3 & Backtracking MRV

Mengalokasikan kombinasi paket wisata (Akomodasi, Kuliner, Destinasi Utama, Destinasi UMKM)
yang memenuhi batasan anggaran dan regulasi bisnis pariwisata Kabupaten Toba.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import inf, isfinite

VARIABLES = ("homestay", "kuliner", "destinasi_utama", "destinasi_umkm")


@dataclass(frozen=True, slots=True)
class ItemWisata:
    """Satu pilihan akomodasi, kuliner, atau destinasi beserta harga dan lokasinya."""

    id: str
    nama: str
    kategori: str
    harga: float
    lokasi: str
    is_umkm: bool = False

    def __post_init__(self) -> None:
        if self.kategori not in VARIABLES:
            raise ValueError(f"Kategori tidak dikenal: {self.kategori!r}")
        if not isfinite(self.harga) or self.harga < 0:
            raise ValueError("Harga harus berupa angka finite dan nonnegatif")


class TobaCSPSolver:
    """Memilih tepat satu item per kategori dengan biaya minimum yang layak.

    Formulasi: setiap variabel dalam ``VARIABLES`` mendapat satu item dari
    domain kategorinya; jumlah harga tidak melebihi ``budget``; dan, bila
    ``require_umkm`` aktif, sedikitnya satu item harus memiliki ``is_umkm``.
    Objektifnya adalah meminimalkan total harga. Solusi dikembalikan sebagai
    mapping variabel ke item, atau ``None`` jika domain tidak feasible.
    """

    def __init__(
        self,
        domain_data: Mapping[str, list[ItemWisata]],
        budget: float,
        require_umkm: bool = True,
    ) -> None:
        if not isfinite(budget) or budget < 0:
            raise ValueError("Budget harus berupa angka finite dan nonnegatif")

        missing = set(VARIABLES) - domain_data.keys()
        extra = domain_data.keys() - set(VARIABLES)
        if missing or extra:
            raise ValueError(f"Domain harus memiliki variabel {VARIABLES}; hilang={missing}, ekstra={extra}")

        self.variables = VARIABLES
        self.domains = {variable: list(domain_data[variable]) for variable in VARIABLES}
        for variable, items in self.domains.items():
            if any(not isinstance(item, ItemWisata) for item in items):
                raise TypeError(f"Domain {variable!r} hanya boleh berisi ItemWisata")
            if any(item.kategori != variable for item in items):
                raise ValueError(f"Setiap item pada domain {variable!r} harus berkategori sama")

        self.budget = budget
        self.require_umkm = require_umkm
        self._best_solution: dict[str, ItemWisata] | None = None
        self._best_cost = inf

    def _minimum_completion_cost(
        self,
        domains: Mapping[str, list[ItemWisata]],
        require_umkm: bool,
    ) -> float:
        """Hitung batas bawah biaya domain tersisa, termasuk kebutuhan UMKM."""
        if any(not items for items in domains.values()):
            return inf

        minima = {variable: min(item.harga for item in items) for variable, items in domains.items()}
        base_cost = sum(minima.values())
        if not require_umkm:
            return base_cost

        umkm_minima = {
            variable: min((item.harga for item in items if item.is_umkm), default=inf)
            for variable, items in domains.items()
        }
        if any(umkm_minima[var] == minima[var] for var in domains):
            return base_cost
        return min(
            (base_cost - minima[var] + umkm_minima[var] for var in domains),
            default=inf,
        )

    def _propagate_global_constraints(
        self,
        domains: dict[str, list[ItemWisata]],
        budget: float,
        require_umkm: bool,
    ) -> bool:
        """Enforce generalized arc consistency for budget and UMKM constraints."""
        changed = True
        while changed:
            changed = False
            for variable in tuple(domains):
                other_domains = {
                    other: items for other, items in domains.items() if other != variable
                }
                supported_items = [
                    item
                    for item in domains[variable]
                    if item.harga
                    + self._minimum_completion_cost(
                        other_domains,
                        require_umkm and not item.is_umkm,
                    )
                    <= budget
                ]
                if len(supported_items) != len(domains[variable]):
                    domains[variable] = supported_items
                    changed = True
                    if not supported_items:
                        return False
        return True

    def apply_ac3_node_and_arc_consistency(self) -> bool:
        """Prune over-budget values, then propagate global constraints to a fixed point.

        The budget and UMKM rules are global constraints rather than binary
        relations, so this applies generalized arc consistency (GAC), the
        appropriate extension of AC-3 for these domains.
        """
        for variable in self.variables:
            self.domains[variable] = [
                item for item in self.domains[variable] if item.harga <= self.budget
            ]
            if not self.domains[variable]:
                return False
        return self._propagate_global_constraints(
            self.domains, self.budget, self.require_umkm
        )

    def select_unassigned_variable_mrv(
        self,
        assignment: Mapping[str, ItemWisata],
        domains: Mapping[str, list[ItemWisata]] | None = None,
    ) -> str:
        """Select the unassigned variable with the fewest remaining values."""
        candidates = [variable for variable in self.variables if variable not in assignment]
        if not candidates:
            raise ValueError("Tidak ada variabel yang belum ditugaskan")
        active_domains = self.domains if domains is None else domains
        return min(candidates, key=lambda variable: len(active_domains[variable]))

    def is_consistent(
        self,
        var: str,
        value: ItemWisata,
        assignment: Mapping[str, ItemWisata],
    ) -> bool:
        """Check category, partial budget, and completed UMKM constraints."""
        if var not in self.variables or var in assignment or value.kategori != var:
            return False
        if any(key not in self.variables for key in assignment):
            return False

        total_cost = sum(item.harga for item in assignment.values()) + value.harga
        if total_cost > self.budget:
            return False

        has_umkm = value.is_umkm or any(item.is_umkm for item in assignment.values())
        completes_assignment = len(assignment) + 1 == len(self.variables)
        return not (completes_assignment and self.require_umkm and not has_umkm)

    def solve(self) -> dict[str, ItemWisata] | None:
        """Return the least expensive feasible package, or ``None`` if infeasible."""
        self._best_solution = None
        self._best_cost = inf
        if not self.apply_ac3_node_and_arc_consistency():
            return None
        self._backtrack({})
        return self._best_solution

    def _backtrack(
        self, assignment: dict[str, ItemWisata]
    ) -> dict[str, ItemWisata] | None:
        """Explore candidates with MRV, GAC propagation, and an optimality bound."""
        current_cost = sum(item.harga for item in assignment.values())
        if len(assignment) == len(self.variables):
            if current_cost < self._best_cost:
                self._best_solution = assignment.copy()
                self._best_cost = current_cost
            return self._best_solution

        remaining_domains = {
            variable: list(self.domains[variable])
            for variable in self.variables
            if variable not in assignment
        }
        remaining_budget = self.budget - current_cost
        needs_umkm = self.require_umkm and not any(
            item.is_umkm for item in assignment.values()
        )
        if not self._propagate_global_constraints(
            remaining_domains, remaining_budget, needs_umkm
        ):
            return self._best_solution

        lower_bound = current_cost + self._minimum_completion_cost(
            remaining_domains, needs_umkm
        )
        if lower_bound >= self._best_cost:
            return self._best_solution

        variable = self.select_unassigned_variable_mrv(assignment, remaining_domains)
        for item in sorted(remaining_domains[variable], key=lambda candidate: candidate.harga):
            if self.is_consistent(variable, item, assignment):
                next_assignment = assignment | {variable: item}
                self._backtrack(next_assignment)
        return self._best_solution


# ---------------------------------------------------------------------------
# Sample Dataset Dummy untuk Uji Coba Langsung
# ---------------------------------------------------------------------------
def get_sample_toba_domain() -> dict[str, list[ItemWisata]]:
    return {
        "homestay": [
            ItemWisata("H1", "Homestay Meat Hill", "homestay", 150000, "Desa Ulos Meat", is_umkm=True),
            ItemWisata("H2", "Hotel Grand Balige", "homestay", 350000, "Balige", is_umkm=False),
        ],
        "kuliner": [
            ItemWisata("K1", "Mie Gombal Balige", "kuliner", 25000, "Balige", is_umkm=True),
            ItemWisata("K2", "Resto Waterfront Parapat", "kuliner", 85000, "Parapat", is_umkm=False),
        ],
        "destinasi_utama": [
            ItemWisata("D1", "Wisata Tomok Samosir", "destinasi_utama", 15000, "Tomok", is_umkm=False),
            ItemWisata("D2", "Pantai Bebas Parapat", "destinasi_utama", 10000, "Parapat", is_umkm=False),
        ],
        "destinasi_umkm": [
            ItemWisata("U1", "Desa Ulos Meat", "destinasi_umkm", 20000, "Meat", is_umkm=True),
            ItemWisata("U2", "Sentra Kerajinan Tenun Balige", "destinasi_umkm", 15000, "Balige", is_umkm=True),
        ],
    }


if __name__ == "__main__":
    domain_data = get_sample_toba_domain()
    print("=== Toba Copilot: CSP Business Constraint Solver ===")
    try:
        budget_wisatawan = float(input("Masukkan batas anggaran wisatawan (Rp): "))
    except ValueError:
        print("Input budget tidak valid. Masukkan angka, misalnya 250000.")
    else:
        if not isfinite(budget_wisatawan) or budget_wisatawan < 0:
            print("Budget harus berupa angka finite dan tidak boleh negatif.")
        else:
            solver = TobaCSPSolver(
                domain_data, budget=budget_wisatawan, require_umkm=True
            )
            solusi = solver.solve()

            if solusi:
                total_harga = sum(item.harga for item in solusi.values())
                sisa_budget = budget_wisatawan - total_harga
                nama_kategori = {
                    "homestay": "Homestay",
                    "kuliner": "Kuliner",
                    "destinasi_utama": "Destinasi Utama",
                    "destinasi_umkm": "Destinasi UMKM",
                }

                print("\nStatus: Solusi paket ditemukan!")
                print(f"Total biaya: Rp {total_harga:,.0f}")
                print(f"Sisa budget: Rp {sisa_budget:,.0f}")
                print("Rincian paket:")
                for var, item in solusi.items():
                    status_umkm = "Ya" if item.is_umkm else "Tidak"
                    print(
                        f" - {nama_kategori[var]}: {item.nama} | "
                        f"Harga: Rp {item.harga:,.0f} | UMKM: {status_umkm}"
                    )
            else:
                print(
                    "Solusi tidak ditemukan. Budget tidak mencukupi "
                    "(termasuk jika budget Rp 0)."
                )