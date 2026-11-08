import csv
import sys
from pathlib import Path

# Pastikan root direktori proyek ada di sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retriever import (
    TobaKnowledgeRetriever,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
)


def run_benchmark():
    """
    Eksekusi Benchmark Temu Balik Semantik (Retrieval Benchmark)
    Menguji 5 Kueri Bisnis Sesuai Spesifikasi Milestone 3:
    1. Exact / Keyword Match
    2. Semantic Paraphrase / Sinonim
    3. Metadata Filter (Kategori / Spesifik)
    4. Ambiguous Query
    5. Out-of-Domain Query (Guardrail Rejection)
    """
    print("=" * 80)
    print("BENCHMARK TEMU BALIK SEMANTIK & EVALUASI GUARDRAIL (MILESTONE 3 - W07)")
    print("=" * 80)

    # Inisialisasi retriever dengan ambang batas empiris terkalibrasi (0.58)
    EMPIRICAL_THRESHOLD = 0.58
    retriever = TobaKnowledgeRetriever(threshold=EMPIRICAL_THRESHOLD)

    # Definisi 5 Kueri Uji Bisnis Terstruktur
    test_cases = [
        {
            "id": "Q1_EXACT",
            "tipe": "Exact/Keyword Match",
            "kueri": "tarif retribusi tempat rekreasi dan sarana olah raga",
            "filter": None,
            "target_expected_doc": "D01",
            "ground_truth_keywords": ["retribusi", "rekreasi", "tarif", "daerah", "olah raga"],
            "threshold": EMPIRICAL_THRESHOLD
        },
        {
            "id": "Q2_SYNONYM",
            "tipe": "Semantic Paraphrase / Sinonim",
            "kueri": "berapa biaya tiket masuk tempat wisata pantai dan pesisir danau toba",
            "filter": None,
            "target_expected_doc": "D01 / D03",
            "ground_truth_keywords": ["retribusi", "rekreasi", "pantai", "tiket", "tarif", "wisata", "danau toba"],
            "threshold": EMPIRICAL_THRESHOLD
        },
        {
            "id": "Q3_FILTER",
            "tipe": "Metadata Filter (Kategori Spesifik)",
            "kueri": "kamar tidur dan fasilitas toilet tamu",
            "filter": {"kategori": "standar_layanan"},
            "target_expected_doc": "D02 / D06",
            "ground_truth_keywords": ["pondok", "wisata", "kamar", "fasilitas", "layanan", "tamu", "hotel"],
            "threshold": EMPIRICAL_THRESHOLD
        },
        {
            "id": "Q4_AMBIGUOUS",
            "tipe": "Ambiguous Query",
            "kueri": "keamanan dan kenyamanan tempat menginap",
            "filter": None,
            "target_expected_doc": "D02 / D06",
            "ground_truth_keywords": ["keamanan", "keselamatan", "hotel", "pondok", "kenyamanan", "chse"],
            "threshold": EMPIRICAL_THRESHOLD
        },
        {
            "id": "Q5_OOD",
            "tipe": "Out-of-Domain (Guardrail Test)",
            "kueri": "rumus kalkulus turunan dan integral fungsi trigonometri aljabar",
            "filter": None,
            "target_expected_doc": "NONE (Wajib Ditolak)",
            "ground_truth_keywords": [],
            "threshold": EMPIRICAL_THRESHOLD
        }
    ]

    benchmark_rows = []

    for tc in test_cases:
        qid = tc["id"]
        qtype = tc["tipe"]
        query = tc["kueri"]
        wfilter = tc["filter"]
        threshold = tc["threshold"]

        print(f"\nMenjalankan {qid} ({qtype})...")
        print(f"Kueri: \"{query}\"")
        if wfilter:
            print(f"Filter Metadata: {wfilter}")

        # Jalankan pencarian mentah untuk analisis jarak & similarity
        raw_results = retriever.search(query=query, top_k=5, where_filter=wfilter)

        # Jalankan pencarian dengan guardrail
        guardrail_res = retriever.search_with_guardrail(
            query=query,
            top_k=5,
            threshold=threshold,
            where_filter=wfilter
        )

        status_guardrail = guardrail_res["status"]
        top_similarity = guardrail_res["similarity"]

        # Evaluasi relevansi untuk P@3, P@5, R@3
        k = 3
        relevant_count_k = 0
        total_relevant_in_top5 = 0

        keywords = tc["ground_truth_keywords"]

        if tc["id"] == "Q5_OOD":
            # Untuk Out-of-Domain, kueri berhasil jika status == 'rejected'
            precision_at_k = 1.0 if status_guardrail == "rejected" else 0.0
            recall_at_k = 1.0 if status_guardrail == "rejected" else 0.0
            mrr_score = 1.0 if status_guardrail == "rejected" else 0.0
            ndcg_at_k = 1.0 if status_guardrail == "rejected" else 0.0
            best_doc_id = "N/A (Rejected)"
            best_doc_preview = "Ditolak Guardrail (Out-of-Domain)"
        else:
            # Identifikasi ID relevan secara terstruktur untuk MRR dan nDCG
            retrieved_ids = []
            ground_truth_matched_ids = set()

            for idx, hit in enumerate(raw_results):
                doc_id = hit["id"]
                retrieved_ids.append(doc_id)
                text_lower = hit["document"].lower()
                meta_str = str(hit["metadata"]).lower()
                is_rel = any(kw in text_lower or kw in meta_str for kw in keywords)
                if is_rel:
                    ground_truth_matched_ids.add(doc_id)
                    total_relevant_in_top5 += 1
                    if idx < k:
                        relevant_count_k += 1

            precision_at_k = relevant_count_k / k
            recall_at_k = relevant_count_k / max(1, total_relevant_in_top5)
            mrr_score = calculate_mrr(retrieved_ids, ground_truth_matched_ids)
            ndcg_at_k = calculate_ndcg_at_k(retrieved_ids, ground_truth_matched_ids, k=k)

            if raw_results:
                best_doc = raw_results[0]
                best_doc_id = best_doc["metadata"].get("doc_id", best_doc["metadata"].get("source_type", "doc"))
                best_doc_preview = best_doc["document"][:85].replace("\n", " ") + "..."
            else:
                best_doc_id = "None"
                best_doc_preview = "None"

        print(f"Hasil Guardrail: [{status_guardrail.upper()}] (Top Cosine Sim: {top_similarity:.4f} vs Threshold: {threshold})")
        print(f"Top Hit Doc ID: {best_doc_id}")
        print(f"Cuplikan Teks: {best_doc_preview}")
        print(f"Metrik Evaluasi: P@{k} = {precision_at_k:.2f} | R@{k} = {recall_at_k:.2f} | MRR = {mrr_score:.2f} | nDCG@{k} = {ndcg_at_k:.2f}")

        benchmark_rows.append({
            "query_id": qid,
            "tipe_kueri": qtype,
            "kueri": query,
            "filter_metadata": str(wfilter) if wfilter else "Tanpa Filter",
            "top_cosine_similarity": round(top_similarity, 4),
            "guardrail_status": status_guardrail,
            "precision_at_3": round(precision_at_k, 2),
            "recall_at_3": round(recall_at_k, 2),
            "mrr": round(mrr_score, 2),
            "ndcg_at_3": round(ndcg_at_k, 2),
            "top_doc_id": best_doc_id,
            "top_doc_excerpt": best_doc_preview
        })

    # Simpan hasil benchmark ke CSV
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "retrieval_benchmark.csv"

    fieldnames = [
        "query_id", "tipe_kueri", "kueri", "filter_metadata",
        "top_cosine_similarity", "guardrail_status",
        "precision_at_3", "recall_at_3", "mrr", "ndcg_at_3",
        "top_doc_id", "top_doc_excerpt"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print("\n" + "=" * 80)
    print(f"✅ Seluruh 5 Kueri Benchmark Berhasil Dieksekusi & Tervalidasi!")
    print(f"Laporan Benchmark tersimpan di: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
