import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver import ItemWisata, TobaCSPSolver, get_sample_toba_domain


def test_solver_finds_valid_package_within_budget():
    """Memastikan solver menemukan paket yang valid jika anggaran mencukupi."""
    domain_data = get_sample_toba_domain()
    budget = 300000.0  # Rp 300.000 (Cukup)
    
    solver = TobaCSPSolver(domain_data, budget=budget, require_umkm=True)
    solution = solver.solve()

    assert solution is not None
    assert set(solution) == {
        "homestay",
        "kuliner",
        "destinasi_utama",
        "destinasi_umkm",
    }
    total_cost = sum(item.harga for item in solution.values())
    assert total_cost <= budget
    assert any(item.is_umkm for item in solution.values())


def test_solver_zero_or_insufficient_budget_returns_none():
    """Memastikan budget Rp 0 atau yang tidak mencukupi mengembalikan None secara aman."""
    domain_data = get_sample_toba_domain()
    
    # Uji Budget Rp 0
    solver_zero = TobaCSPSolver(domain_data, budget=0.0, require_umkm=True)
    assert solver_zero.solve() is None

    # Uji Budget Terlalu Kecil (Rp 10.000)
    solver_low = TobaCSPSolver(domain_data, budget=10000.0, require_umkm=True)
    assert solver_low.solve() is None


def test_solver_enforces_umkm_constraint():
    """Memastikan solusi yang dihasilkan selalu memuat minimal 1 item UMKM lokal."""
    domain_data = get_sample_toba_domain()
    budget = 500000.0  # Budget melimpah

    solver = TobaCSPSolver(domain_data, budget=budget, require_umkm=True)
    solution = solver.solve()

    assert solution is not None
    has_umkm = any(item.is_umkm for item in solution.values())
    assert has_umkm is True


def test_ac3_prunes_overbudget_domain_items():
    """Memastikan propagasi AC-3 berhasil memangkas item yang harganya melebihi budget."""
    domain_data = get_sample_toba_domain()
    budget = 200000.0  # Hotel Grand Balige (Rp 350.000) harus dipangkas oleh AC-3

    solver = TobaCSPSolver(domain_data, budget=budget)
    solver.apply_ac3_node_and_arc_consistency()

    # Cek apakah Hotel Grand Balige sudah terhapus dari domain homestay
    remaining_homestay_names = [item.nama for item in solver.domains["homestay"]]
    assert "Hotel Grand Balige" not in remaining_homestay_names


def test_solver_returns_minimum_cost_package_at_exact_budget():
    """The optimizer accepts the exact budget boundary and minimizes total cost."""
    solver = TobaCSPSolver(get_sample_toba_domain(), budget=200000.0)

    solution = solver.solve()

    assert solution is not None
    assert sum(item.harga for item in solution.values()) == 200000.0


def test_solver_returns_none_when_no_domain_contains_umkm():
    """A feasible budget cannot bypass the mandatory UMKM rule."""
    domain_data = get_sample_toba_domain()
    domain_data = {
        variable: [
            ItemWisata(item.id, item.nama, item.kategori, item.harga, item.lokasi, False)
            for item in items
        ]
        for variable, items in domain_data.items()
    }

    assert TobaCSPSolver(domain_data, budget=500000.0).solve() is None
    solution = TobaCSPSolver(
        domain_data, budget=500000.0, require_umkm=False
    ).solve()
    assert solution is not None
    assert not any(item.is_umkm for item in solution.values())


def test_solver_returns_none_when_combined_cost_exceeds_budget():
    """Individually affordable items may still be infeasible as a complete package."""
    solver = TobaCSPSolver(get_sample_toba_domain(), budget=199999.0)

    assert solver.solve() is None


def test_solver_returns_none_for_empty_required_domain():
    """An empty required category makes the CSP unsatisfiable."""
    domain_data = get_sample_toba_domain()
    domain_data["kuliner"] = []

    assert TobaCSPSolver(domain_data, budget=500000.0).solve() is None


def test_solver_rejects_negative_budget():
    """Invalid budget input is rejected explicitly instead of silently searched."""
    import pytest

    with pytest.raises(ValueError, match="Budget"):
        TobaCSPSolver(get_sample_toba_domain(), budget=-1.0)