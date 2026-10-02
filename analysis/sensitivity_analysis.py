import sys
import time
import random
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Add root dir to sys.path so we can import solver
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver import VARIABLES, ItemWisata, TobaCSPSolver

def generate_synthetic_domain(n_items_per_category: int, seed: int = 42) -> dict[str, list[ItemWisata]]:
    random.seed(seed)
    domain_data = {}
    
    for category in VARIABLES:
        items = []
        for i in range(n_items_per_category):
            # Range harga disimulasikan sesuai realita masing-masing kategori
            if category == "homestay":
                harga = random.randint(100, 1000) * 1000
            elif category == "kuliner":
                harga = random.randint(15, 200) * 1000
            elif category == "destinasi_utama":
                harga = random.randint(0, 150) * 1000
            else: # destinasi_umkm
                harga = random.randint(0, 50) * 1000
                
            is_umkm = random.random() < 0.3
            
            items.append(
                ItemWisata(
                    id=f"{category[:2].upper()}{i}",
                    nama=f"Sintetis {category} {i}",
                    kategori=category,
                    harga=float(harga),
                    lokasi="Lokasi Dummy",
                    is_umkm=is_umkm
                )
            )
        domain_data[category] = items
    return domain_data

def measure_pruning_effectiveness(domain_data: dict[str, list[ItemWisata]], budget: float) -> tuple[int, int]:
    """Mengukur jumlah nilai tersisa sebelum vs sesudah inisialisasi GAC"""
    solver = TobaCSPSolver(domain_data, budget=budget, require_umkm=True)
    initial_count = sum(len(items) for items in solver.domains.values())
    
    solver.apply_ac3_node_and_arc_consistency()
    
    final_count = sum(len(items) for items in solver.domains.values())
    return initial_count, final_count

def main():
    sizes = [2, 5, 10, 20, 50]
    
    # Skenario Budget (Min total base cost sekitar 115k, max ~1.400k)
    budgets = {
        "strict": 100000.0,   # Sangat ketat, kemungkinan besar gagal (infeasible)
        "tight": 250000.0,    # Pas-pasan, memicu backtracking yang lebih dalam
        "loose": 1000000.0    # Longgar, banyak solusi yang sangat mudah ditemukan
    }
    
    results = []
    
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(exist_ok=True)
    
    print("Memulai Analisis Sensitivitas TobaCSPSolver...\n")
    
    for n in sizes:
        print(f"=== Ukuran Domain (n={n}) per kategori ===")
        domain_data = generate_synthetic_domain(n)
        
        for scenario, budget in budgets.items():
            solver = TobaCSPSolver(domain_data, budget=budget, require_umkm=True)
            
            start_time = time.perf_counter()
            solution = solver.solve()
            end_time = time.perf_counter()
            
            elapsed_ms = (end_time - start_time) * 1000
            
            total_cost = 0.0
            found = False
            if solution is not None:
                total_cost = sum(item.harga for item in solution.values())
                found = True
                
            results.append({
                "n": n,
                "scenario": scenario,
                "budget": budget,
                "found": found,
                "total_cost": total_cost,
                "time_ms": elapsed_ms
            })
            
            print(f"  Skenario '{scenario}' (B={budget:,.0f}): "
                  f"{'Found' if found else 'No Sol'} | {elapsed_ms:.2f} ms")
    
    # 1. Simpan CSV
    csv_path = output_dir / "sensitivity_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "scenario", "budget", "found", "total_cost", "time_ms"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n[OK] Hasil CSV disimpan ke: {csv_path.relative_to(Path.cwd())}")

    # 2. Visualisasi Grafik (Matplotlib)
    if plt is not None:
        plt.figure(figsize=(10, 6))
        for scenario in budgets.keys():
            x = [r["n"] for r in results if r["scenario"] == scenario]
            y = [r["time_ms"] for r in results if r["scenario"] == scenario]
            plt.plot(x, y, marker="o", label=f"Budget: {scenario}")
            
        plt.title("Sensitivitas Waktu Eksekusi CSP terhadap Ukuran Domain")
        plt.xlabel("Ukuran Domain per Kategori (n)")
        plt.ylabel("Waktu Eksekusi (ms)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        chart_path = output_dir / "convergence_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Grafik konvergensi disimpan ke: {chart_path.relative_to(Path.cwd())}")
    else:
        print("[!] Matplotlib tidak terinstall. Grafik tidak dibuat.")

    # 3. Analisis dan Ringkasan
    print("\n=== RINGKASAN ANALISIS ===")
    
    print("1. Kinerja Skalabilitas Waktu (n=20 vs n=50):")
    for scenario in budgets.keys():
        t20 = next(r["time_ms"] for r in results if r["n"] == 20 and r["scenario"] == scenario)
        t50 = next(r["time_ms"] for r in results if r["n"] == 50 and r["scenario"] == scenario)
        print(f"   - Skenario '{scenario}': n=20 -> {t20:.2f} ms | n=50 -> {t50:.2f} ms")
    print("   -> Pada ukuran n=50 (total 200 kombinasi variabel), waktu meningkat namun")
    print("      biasanya masih tertangani dengan baik berkat pemangkasan solver.")
    
    print("\n2. Insight Efektivitas GAC Pruning (untuk n=50, budget 'tight'):")
    domain_50 = generate_synthetic_domain(50)
    init_count, final_count = measure_pruning_effectiveness(domain_50, budgets["tight"])
    print(f"   - Jumlah kandidat keseluruhan AWAL     : {init_count} items")
    print(f"   - Jumlah kandidat SETELAH GAC pruning  : {final_count} items")
    if init_count > 0:
        pruned_pct = (init_count - final_count) / init_count * 100
        print(f"   - Hasil: {pruned_pct:.1f}% kandidat (yang tidak feasible/terlalu mahal)")
        print("            berhasil dipangkas SEBELUM proses rekursif Backtracking dimulai!")

if __name__ == "__main__":
    main()
