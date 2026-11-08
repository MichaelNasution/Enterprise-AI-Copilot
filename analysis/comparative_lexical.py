import csv
import json
import sys
from pathlib import Path

# Pastikan root direktori proyek ada di sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.retriever import TobaKnowledgeRetriever


def run_lexical_vs_dense_comparison():
    """
    Eksperimen Komparasi Pendekatan Leksikal Klasik (TF-IDF) vs Dense Embedding (Sentence-Transformers)
    Membuktikan dalil materi perkuliahan W07 (Slide 9-12):
    1. TF-IDF kuat pada pencocokan kata persis (exact match).
    2. TF-IDF gagal total (skor mendekati/sama dengan 0) pada kueri sinonim / parafrasa.
    3. Dense Embedding berhasil menangkap hubungan makna (sinonim) dalam ruang vektor kontinu.
    """
    print("=" * 80)
    print("EKSPERIMEN KOMPARASI: TF-IDF (LEKSIKAL) VS DENSE VECTOR EMBEDDING (M3)")
    print("=" * 80)

    chunks_file = ROOT_DIR / "data" / "chunks" / "chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Berkas chunks {chunks_file} tidak ditemukan!")

    # 1. Membaca korpus teks (3.308 chunks)
    print("Membaca seluruh dokumen korpus teks...")
    documents = []
    doc_ids = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            documents.append(item.get("text", item.get("content", "")))
            doc_ids.append(item.get("id", ""))

    print(f"Total dokumen korpus: {len(documents)}")

    # 2. Fit TF-IDF Vectorizer pada Korpus
    print("Membangun matriks TF-IDF pada korpus dokumen...")
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),  # unigram & bigram
        max_df=0.85,
        min_df=2
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    print(f"Bentuk matriks TF-IDF: {tfidf_matrix.shape} (Sparse Matrix)")

    # 3. Inisialisasi Dense Retriever
    print("Menghubungkan ke ChromaDB Persistent Dense Retriever...")
    retriever = TobaKnowledgeRetriever()

    # 4. Kueri Uji Komparasi
    test_queries = [
        {
            "id": "Q1_EXACT",
            "tipe": "Exact/Keyword Match",
            "kueri": "tarif retribusi tempat rekreasi dan sarana olah raga",
            "catatan": "Kata kunci terdapat persis di dokumen Perda D01"
        },
        {
            "id": "Q2_SYNONYM",
            "tipe": "Semantic Paraphrase / Sinonim",
            "kueri": "berapa biaya tiket masuk tempat wisata pantai dan pesisir danau toba",
            "catatan": "Menggunakan sinonim (biaya, tiket masuk) bukan kata retribusi"
        },
        {
            "id": "Q3_FILTER",
            "tipe": "Category Specific (Layanan)",
            "kueri": "kamar tidur dan fasilitas toilet tamu",
            "catatan": "Persyaratan fasilitas standar operasional homestay/hotel"
        },
        {
            "id": "Q4_AMBIGUOUS",
            "tipe": "Ambiguous / Variatif",
            "kueri": "keamanan dan kenyamanan tempat menginap",
            "catatan": "Konsep umum CHSE penginapan"
        },
        {
            "id": "Q5_OOD",
            "tipe": "Out-of-Domain",
            "kueri": "rumus kalkulus turunan dan integral fungsi trigonometri aljabar",
            "catatan": "Topik matematika di luar domain pariwisata"
        }
    ]

    comparison_results = []

    print("\n" + "-" * 80)
    print(f"{'ID':<11} | {'Tipe Kueri':<24} | {'TF-IDF Sim':<11} | {'Dense Sim':<11} | {'Selisih (Dense - TFIDF)':<22}")
    print("-" * 80)

    for item in test_queries:
        qid = item["id"]
        qtype = item["tipe"]
        query = item["kueri"]

        # A. Hitung Kemiripan TF-IDF
        query_vec = tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_tfidf_idx = int(np.argmax(tfidf_similarities))
        top_tfidf_sim = float(tfidf_similarities[top_tfidf_idx])
        top_tfidf_doc_id = doc_ids[top_tfidf_idx]

        # B. Hitung Kemiripan Dense Embedding ChromaDB
        dense_hits = retriever.search(query=query, top_k=1)
        if dense_hits:
            top_dense_sim = float(dense_hits[0]["similarity"])
            top_dense_doc_id = dense_hits[0]["id"]
            top_dense_excerpt = dense_hits[0]["document"][:80].replace("\n", " ") + "..."
        else:
            top_dense_sim = 0.0
            top_dense_doc_id = "None"
            top_dense_excerpt = "None"

        diff = top_dense_sim - top_tfidf_sim

        print(f"{qid:<11} | {qtype[:24]:<24} | {top_tfidf_sim:<11.4f} | {top_dense_sim:<11.4f} | {diff:<+22.4f}")

        comparison_results.append({
            "query_id": qid,
            "tipe_kueri": qtype,
            "kueri": query,
            "top_tfidf_similarity": round(top_tfidf_sim, 4),
            "top_tfidf_doc_id": top_tfidf_doc_id,
            "top_dense_similarity": round(top_dense_sim, 4),
            "top_dense_doc_id": top_dense_doc_id,
            "similarity_delta": round(diff, 4),
            "analisis_fenomena": item["catatan"],
            "top_dense_excerpt": top_dense_excerpt
        })

    print("-" * 80)

    # Simpan laporan komparasi ke CSV
    out_dir = ROOT_DIR / "analysis"
    out_dir.mkdir(exist_ok=True)
    csv_file = out_dir / "lexical_vs_dense.csv"

    fieldnames = [
        "query_id", "tipe_kueri", "kueri",
        "top_tfidf_similarity", "top_tfidf_doc_id",
        "top_dense_similarity", "top_dense_doc_id",
        "similarity_delta", "analisis_fenomena", "top_dense_excerpt"
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_results)

    print(f"\n✅ Laporan komparasi tersimpan di: {csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_lexical_vs_dense_comparison()

