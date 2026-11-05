"""
Analisis Sensitivitas TobaCSPSolver (Milestone 2) — reproducible.

Eksperimen A: skala domain (jumlah item per kategori) pada budget tetap.
Eksperimen B: ketatnya anggaran pada ukuran domain tetap.

Seluruh angka di laporan M2 harus berasal dari output skrip ini:
    uv run python analysis/sensitivity_analysis.py

Keluaran (folder analysis/):
    sensitivity_scale.csv, sensitivity_budget.csv,
    scale_space_vs_nodes.png, scale_time.png

Kolom closed_form_* membandingkan solver dengan perhitungan langsung
_minimum_completion_cost() pada domain penuh. Karena C1 (budget) dan C2 (UMKM)
tidak mengikat pasangan variabel tertentu, optimum model ini dapat dihitung
langsung dalam O(n); kolom ini mendokumentasikan keterbatasan tersebut secara jujur.
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
import time
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # grafik bersifat opsional
    plt = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver import VARIABLES, ItemWisata, TobaCSPSolver  # noqa: E402

SEED = 42
REPEATS = 5  # waktu dilaporkan sebagai median dari beberapa kali eksekusi
SCALE_SIZES = [2, 5, 10, 25, 50, 100, 200]
SCALE_BUDGET = 1_000_000.0
BUDGET_N = 50
BUDGET_LEVELS = [50_000.0, 100_000.0, 200_000.0, 500_000.0, 1_000_000.0]
OUTPUT_DIR = Path(__file__).resolve().parent


def generate_synthetic_domain(n_items_per_category: int, seed: int = SEED) -> dict[str, list[ItemWisata]]:
    """Domain sintetis dengan rentang harga per kategori (distribusi seragam)."""
    rng = random.Random(seed)
    domain_data: dict[str, list[ItemWisata]] = {}
    for category in VARIABLES:
        items = []
        for i in range(n_items_per_category):
            if category == "homestay":
                harga = rng.randint(100, 1000) * 1000
            elif category == "kuliner":
                harga = rng.randint(15, 200) * 1000
            elif category == "destinasi_utama":
                harga = rng.randint(0, 150) * 1000
            else:  # destinasi_umkm
                harga = rng.randint(0, 50) * 1000
            items.append(
                ItemWisata(
                    id=f"{category[:2].upper()}{i}",
                    nama=f"Sintetis {category} {i}",
                    kategori=category,
                    harga=float(harga),
                    lokasi="Lokasi Dummy",
                    is_umkm=rng.random() < 0.3,
                )
            )
        domain_data[category] = items
    return domain_data


def run_solver(domain_data: dict[str, list[ItemWisata]], budget: float) -> dict:
    """Jalankan solver REPEATS kali; kembalikan metrik dari eksekusi terakhir + median waktu."""
    times = []
    solver = None
    solution = None
    for _ in range(REPEATS):
        solver = TobaCSPSolver(domain_data, budget=budget, require_umkm=True)
        start = time.perf_counter()
        solution = solver.solve()
        times.append((time.perf_counter() - start) * 1000)

    total_cost = sum(item.harga for item in solution.values()) if solution else None

    # Pembanding: optimum langsung dari batas bawah pada domain penuh (tanpa pencarian)
    reference = TobaCSPSolver(domain_data, budget=budget, require_umkm=True)
    start = time.perf_counter()
    closed = reference._minimum_completion_cost(reference.domains, True)
    closed_ms = (time.perf_counter() - start) * 1000
    closed_cost = closed if closed <= budget else None

    return {
        "status": "SOLVED" if solution else "NO_SOLUTION",
        "total_cost": total_cost,
        "nodes_expanded": solver.nodes_expanded,
        "time_ms": statistics.median(times),
        "closed_form_ms": closed_ms,
        "closed_form_matches": closed_cost == total_cost,
    }


def experiment_scale() -> list[dict]:
    rows = []
    for n in SCALE_SIZES:
        result = run_solver(generate_synthetic_domain(n), SCALE_BUDGET)
        space = n ** len(VARIABLES)
        rows.append(
            {
                "n": n,
                "brute_force_space": space,
                "exploration_ratio_pct": 100 * result["nodes_expanded"] / space,
                **result,
            }
        )
        print(
            f"  n={n:>3} | space={space:>13,} | nodes={result['nodes_expanded']:>5} | "
            f"{result['time_ms']:9.3f} ms | {result['status']} | cost={result['total_cost']}"
        )
    return rows


def experiment_budget() -> list[dict]:
    rows = []
    domain = generate_synthetic_domain(BUDGET_N)
    for budget in BUDGET_LEVELS:
        result = run_solver(domain, budget)
        rows.append({"n": BUDGET_N, "budget": budget, **result})
        print(
            f"  budget={budget:>11,.0f} | nodes={result['nodes_expanded']:>5} | "
            f"{result['time_ms']:9.3f} ms | {result['status']} | cost={result['total_cost']}"
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] {path.name}")


def plot(scale_rows: list[dict]) -> None:
    if plt is None:
        print("[!] matplotlib tidak tersedia; grafik dilewati.")
        return
    ns = [r["n"] for r in scale_rows]

    plt.figure(figsize=(9, 5.5))
    plt.plot(ns, [r["brute_force_space"] for r in scale_rows], marker="o", label="Ruang kombinasi brute-force (n^4)")
    plt.plot(ns, [max(r["nodes_expanded"], 1) for r in scale_rows], marker="s", label="Node expanded (GAC + Branch-and-Bound)")
    plt.yscale("log")
    plt.xlabel("Jumlah item per kategori (n)")
    plt.ylabel("Jumlah (skala log)")
    plt.title(f"Ruang Pencarian vs Node Dieksplorasi (budget Rp {SCALE_BUDGET:,.0f}, seed {SEED})")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(OUTPUT_DIR / "scale_space_vs_nodes.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.plot(ns, [r["time_ms"] for r in scale_rows], marker="o", label="Solver (median waktu)")
    plt.plot(ns, [r["closed_form_ms"] for r in scale_rows], marker="s", label="Perhitungan langsung O(n)")
    plt.yscale("log")
    plt.xlabel("Jumlah item per kategori (n)")
    plt.ylabel("Waktu (ms, skala log)")
    plt.title("Waktu Eksekusi terhadap Ukuran Domain")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(OUTPUT_DIR / "scale_time.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] scale_space_vs_nodes.png, scale_time.png")


def main() -> None:
    print(f"Eksperimen A: skala domain (budget Rp {SCALE_BUDGET:,.0f}, seed {SEED}, median {REPEATS} run)")
    scale_rows = experiment_scale()
    print(f"\nEksperimen B: ketatnya anggaran (n={BUDGET_N}, seed {SEED})")
    budget_rows = experiment_budget()
    print()
    write_csv(OUTPUT_DIR / "sensitivity_scale.csv", scale_rows)
    write_csv(OUTPUT_DIR / "sensitivity_budget.csv", budget_rows)
    plot(scale_rows)


if __name__ == "__main__":
    main()
