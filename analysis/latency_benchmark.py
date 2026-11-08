import csv
import os
import sys
import time
from pathlib import Path

# Set HF offline mode agar tidak timeout saat sandbox tidak memiliki akses internet
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Pastikan root direktori proyek ada di sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


def run_latency_benchmark():
    """
    Eksperimen Komparasi Efisiensi Pencarian:
    HNSW ANN (ChromaDB) vs Brute-Force Linear Scan (Exact Matrix Dot-Product)
    
    Tujuan Ilmiah (Slide W07 hal. 14-19):
    1. Mengukur latency pencarian (pure search time) antara graf HNSW (O(log N))
       dan Linear Scan (O(N * d)) pada korpus 3.308 chunks.
    2. Menghitung recall akurasi HNSW relatif terhadap exact ground truth.
    3. Mengestimasi kebutuhan memori (RAM footprint) dan proyeksi skalabilitas
       hingga skala 100.000 dan 1.000.000 dokumen korpus enterprise.
    """
    print("=" * 80)
    print("BENCHMARK EFISIENSI & LATENSI: HNSW ANN VS BRUTE-FORCE LINEAR SCAN (W07)")
    print("=" * 80)

    db_path = ROOT_DIR / "data" / "chroma_db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database ChromaDB tidak ditemukan di {db_path}!")

    # 1. Buka ChromaDB Client & Collection
    print(f"Membuka database ChromaDB: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))
    col = client.get_collection(name="toba_knowledge_base")
    total_chunks = col.count()
    print(f"Total dokumen terindeks: {total_chunks:,} chunks")

    # 2. Ambil seluruh embeddings untuk representasi Brute-Force Matrix
    print("Mengekstraksi representasi vektor embedding dari database...")
    all_data = col.get(include=["embeddings", "metadatas"])
    doc_ids = all_data["ids"]
    raw_embeddings = np.array(all_data["embeddings"], dtype=np.float32)
    num_docs, dim = raw_embeddings.shape
    print(f"Dimensi matriks embedding: {num_docs} vektor x {dim} dimensi (float32)")

    # Normalisasi vektor untuk komputasi cosine similarity via dot product
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    normalized_embeddings = raw_embeddings / (norms + 1e-10)

    # 3. Model Embedding untuk Kueri
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Memuat model embedding ('{model_name}')...")
    model = SentenceTransformer(model_name)

    test_queries = [
        "tarif retribusi tempat rekreasi dan sarana olah raga",
        "persyaratan standar kamar tidur pondok wisata",
        "rekomendasi daya tarik wisata pantai pesisir toba",
        "desa wisata dan pelestarian kebudayaan batak",
        "kuliner khas dan oleh-oleh tradisional di samosir"
    ]

    k = 5
    n_runs = 20  # Jumlah pengulangan untuk stabilitas pengukuran latency
    print(f"\nMemulai pengujian latency ({n_runs} iterasi per kueri, k={k})...\n")

    benchmark_rows = []
    hnsw_latencies = []
    brute_latencies = []
    recalls = []

    print("-" * 96)
    print(f"{'No':<3} | {'Kueri':<42} | {'Brute (ms)':<12} | {'HNSW (ms)':<12} | {'Recall@5':<10}")
    print("-" * 96)

    for idx, query in enumerate(test_queries, 1):
        q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

        # A. Pengukuran Pure Brute-Force Linear Scan (NumPy dot product + top-k argpartition)
        # Warm-up
        _ = np.dot(normalized_embeddings, q_vec.T)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            scores = np.dot(normalized_embeddings, q_vec.T).flatten()
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        t1 = time.perf_counter()
        avg_brute_ms = ((t1 - t0) / n_runs) * 1000.0
        exact_top_ids = [doc_ids[i] for i in top_k_sorted]

        # B. Pengukuran Pure HNSW ANN Search (ChromaDB query by precomputed embedding)
        # Warm-up
        _ = col.query(query_embeddings=q_vec.tolist(), n_results=k)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            res = col.query(query_embeddings=q_vec.tolist(), n_results=k)
        t1 = time.perf_counter()
        avg_hnsw_ms = ((t1 - t0) / n_runs) * 1000.0
        hnsw_top_ids = res["ids"][0]

        # C. Hitung Recall@k HNSW terhadap Ground Truth Brute-Force
        overlap = len(set(hnsw_top_ids).intersection(set(exact_top_ids)))
        recall_at_k = overlap / k

        hnsw_latencies.append(avg_hnsw_ms)
        brute_latencies.append(avg_brute_ms)
        recalls.append(recall_at_k)

        short_q = (query[:39] + "...") if len(query) > 42 else query
        print(f"{idx:<3} | {short_q:<42} | {avg_brute_ms:<12.3f} | {avg_hnsw_ms:<12.3f} | {recall_at_k:<10.2%}")

        benchmark_rows.append({
            "query_id": idx,
            "query": query,
            "brute_force_ms": round(avg_brute_ms, 3),
            "hnsw_ann_ms": round(avg_hnsw_ms, 3),
            "recall_at_5": round(recall_at_k, 4)
        })

    print("-" * 96)
    mean_brute = float(np.mean(brute_latencies))
    mean_hnsw = float(np.mean(hnsw_latencies))
    mean_recall = float(np.mean(recalls))

    print(f"\nRATA-RATA PADA KORPUS N={num_docs:,}:")
    print(f"• Brute-Force Linear Scan : {mean_brute:.3f} ms")
    print(f"• HNSW Graph ANN Search   : {mean_hnsw:.3f} ms")
    print(f"• Rata-rata Recall@5      : {mean_recall:.2%}")

    # 4. Analisis Teoretis Memory Footprint & Proyeksi Skalabilitas (Slide W07)
    bytes_per_float = 4
    raw_memory_bytes = num_docs * dim * bytes_per_float
    raw_memory_mb = raw_memory_bytes / (1024 * 1024)
    # Estimasi overhead struktur graf HNSW (M=16 connections per node, pointers & level lists ~ 1.5x - 2.0x raw)
    hnsw_overhead_mb = raw_memory_mb * 0.75
    total_hnsw_memory_mb = raw_memory_mb + hnsw_overhead_mb

    print("\n" + "=" * 80)
    print("ANALISIS MEMORI & PROYEKSI SKALABILITAS (W07 SLIDE 18-20)")
    print("=" * 80)
    print(f"• Ukuran Vektor Raw (float32) : {raw_memory_mb:.2f} MB ({num_docs:,} x {dim} x 4 bytes)")
    print(f"• Overhead Indeks Graf HNSW   : ~{hnsw_overhead_mb:.2f} MB")
    print(f"• Total RAM Footprint         : ~{total_hnsw_memory_mb:.2f} MB")

    print("\nProyeksi Kompleksitas Komputasi:")
    print("  - Brute Force : Kompleksitas O(N * d). Latensi bertambah secara linear seiring bertambahnya dokumen.")
    print("  - HNSW Graph  : Kompleksitas O(log N). Latensi bertambah secara sub-linear (logaritmik).")

    print("-" * 80)
    print(f"{'Skala Dokumen (N)':<20} | {'Brute-Force Est (ms)':<22} | {'HNSW Est (ms)':<20} | {'RAM Footprint':<14}")
    print("-" * 80)

    scales = [
        (3308, mean_brute, mean_hnsw, total_hnsw_memory_mb),
        (10000, mean_brute * (10000 / 3308), mean_hnsw * (np.log2(10000) / np.log2(3308)), total_hnsw_memory_mb * (10000 / 3308)),
        (100000, mean_brute * (100000 / 3308), mean_hnsw * (np.log2(100000) / np.log2(3308)), total_hnsw_memory_mb * (100000 / 3308)),
        (1000000, mean_brute * (1000000 / 3308), mean_hnsw * (np.log2(1000000) / np.log2(3308)), total_hnsw_memory_mb * (1000000 / 3308))
    ]

    for n_doc, b_est, h_est, ram_est in scales:
        print(f"N = {n_doc:<16,d} | {b_est:<22.2f} | {h_est:<20.2f} | ~{ram_est:<10.1f} MB")
    print("-" * 80)

    # Simpan hasil benchmark ke CSV
    out_csv = ROOT_DIR / "analysis" / "latency_benchmark.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query", "brute_force_ms", "hnsw_ann_ms", "recall_at_5"])
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(f"\n✅ Hasil benchmark latensi tersimpan di: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    run_latency_benchmark()

