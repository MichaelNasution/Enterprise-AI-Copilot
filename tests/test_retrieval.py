import pytest
from pathlib import Path


DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "toba_knowledge_base"


@pytest.fixture(scope="module")
def retriever():
    """Fixture untuk memuat TobaKnowledgeRetriever."""
    from src.retriever import TobaKnowledgeRetriever
    return TobaKnowledgeRetriever(db_path=DB_PATH, collection_name=COLLECTION_NAME)


def test_retriever_initialization(retriever):
    """Memastikan retriever berhasil terhubung ke basis data ChromaDB dan model aktif."""
    assert retriever.collection is not None
    assert retriever.collection.name == COLLECTION_NAME
    assert retriever.collection.count() == 3308


def test_retriever_exact_keyword_query(retriever):
    """
    Kueri Uji 1 (Harfiah/Keyword Match):
    Mencari tarif retribusi tempat rekreasi.
    Dokumen target: Perda Retribusi Rekreasi (D01).
    """
    query = "retribusi tempat rekreasi dan sarana olah raga"
    results = retriever.search(query, top_k=3)

    assert len(results) > 0, "Pencarian harus mengembalikan hasil."
    top_hit = results[0]
    assert top_hit["similarity"] > 0.40, f"Similarity harus tinggi (>0.40), didapat: {top_hit['similarity']}"
    assert top_hit["metadata"].get("doc_id") == "D01" or "retribusi" in top_hit["document"].lower()


def test_retriever_synonym_paraphrase_query(retriever):
    """
    Kueri Uji 2 (Sinonim / Parafrasa Semantik):
    Menguji kemampuan semantik menemukan makna yang sama tanpa kata kunci persis.
    Kueri: 'biaya tiket masuk tempat wisata pantai dan pesisir danau toba'
    Dokumen target: D01 (Perda Retribusi) atau objek wisata terkait.
    """
    query = "biaya tiket masuk tempat wisata pantai dan pesisir danau toba"
    results = retriever.search(query, top_k=3)

    assert len(results) > 0
    top_hit = results[0]
    # Model semantik harus memberikan similarity yang memadai untuk sinonim
    assert top_hit["similarity"] > 0.30
    assert any(
        kw in top_hit["document"].lower() or kw in str(top_hit["metadata"]).lower()
        for kw in ["retribusi", "pantai", "rekreasi", "tarif", "wisata"]
    )


def test_retriever_metadata_filtering(retriever):
    """
    Kueri Uji 3 (Filter Kategori / Metadata):
    Mencari persyaratan kamar dan fasilitas dengan filter kategori 'standar_layanan'.
    Target: Permenparekraf Standar Pondok Wisata (D02).
    """
    query = "kamar tidur dan fasilitas toilet tamu"
    results = retriever.search(query, top_k=3, category="standar_layanan")

    assert len(results) > 0
    for hit in results:
        assert hit["metadata"].get("kategori") == "standar_layanan", (
            f"Setiap hasil harus memiliki kategori 'standar_layanan', didapat: {hit['metadata'].get('kategori')}"
        )


def test_retriever_dataset_filter(retriever):
    """
    Kueri Uji 4 (Pencarian pada Dataset Pariwisata Fakta):
    Mencari air terjun di kawasan Danau Toba dengan filter source_type='dataset'.
    """
    query = "air terjun dengan fasilitas warung makan dan camping"
    results = retriever.search(query, top_k=3, source_type="dataset")

    assert len(results) > 0
    for hit in results:
        assert hit["metadata"].get("source_type") == "dataset"


def test_retriever_guardrail_rejects_out_of_domain(retriever):
    """
    Kueri Uji 5 (Out-of-Domain Guardrail Rejection):
    Kueri yang sama sekali tidak berhubungan dengan pariwisata Toba
    harus ditolak oleh ambang batas (Guardrail Threshold).
    """
    out_of_domain_queries = [
        "rumus fisika gaya gravitasi newton",
        "komputasi kuantum superkonduktor qubit entanglement",
        "rumus fisika medan magnetik hukum ampere"
    ]

    for ood_query in out_of_domain_queries:
        response = retriever.search_with_guardrail(ood_query, top_k=3, threshold=0.25)
        assert response["status"] == "rejected", (
            f"Kueri OOD '{ood_query}' harus ditolak (rejected), tapi status: {response['status']}"
        )
        assert "di luar cakupan" in response["message"].lower()


def test_precision_and_recall_calculation():
    """Memastikan utilitas perhitungan metrik P@k dan R@k bekerja secara matematis."""
    from src.retriever import calculate_precision_at_k, calculate_recall_at_k

    # Simulasi: Dokumen relevan sesungguhnya ada pada ID [1, 2, 3, 4]
    ground_truth = {"doc_1", "doc_2", "doc_3", "doc_4"}
    retrieved_k5 = ["doc_1", "doc_5", "doc_2", "doc_3", "doc_6"]

    # Dari 5 teratas, 3 relevan: P@5 = 3/5 = 0.6
    p5 = calculate_precision_at_k(retrieved_k5, ground_truth, k=5)
    assert pytest.approx(p5, 0.01) == 0.60

    # Dari 4 seluruh relevan, 3 ditemukan: R@5 = 3/4 = 0.75
    r5 = calculate_recall_at_k(retrieved_k5, ground_truth, k=5)
    assert pytest.approx(r5, 0.01) == 0.75


def test_mrr_and_ndcg_calculation():
    """Memastikan utilitas metrik ranking lanjutan (MRR & nDCG@k) bekerja secara matematis."""
    from src.retriever import calculate_mrr, calculate_ndcg_at_k

    ground_truth = {"doc_A", "doc_B"}

    # Kasus 1: Dokumen relevan ada di peringkat 1
    # MRR = 1/1 = 1.0
    # DCG@2 = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    # IDCG@2 = 1.6309 -> nDCG@2 = 1.0
    ranked_perfect = ["doc_A", "doc_B", "doc_C"]
    assert pytest.approx(calculate_mrr(ranked_perfect, ground_truth), 0.01) == 1.00
    assert pytest.approx(calculate_ndcg_at_k(ranked_perfect, ground_truth, k=2), 0.01) == 1.00

    # Kasus 2: Dokumen relevan pertama ada di peringkat 2 (doc_C bukan relevan)
    # MRR = 1/2 = 0.5
    ranked_suboptimal = ["doc_C", "doc_A", "doc_D"]
    assert pytest.approx(calculate_mrr(ranked_suboptimal, ground_truth), 0.01) == 0.50
    # DCG@2 = 0 + 1/log2(3) = 0.6309, IDCG@2 = 1.6309 -> nDCG@2 = 0.6309/1.6309 ≈ 0.3868
    ndcg = calculate_ndcg_at_k(ranked_suboptimal, ground_truth, k=2)
    assert 0.38 <= ndcg <= 0.40

    # Kasus 3: Tidak ada dokumen relevan
    ranked_none = ["doc_X", "doc_Y"]
    assert calculate_mrr(ranked_none, ground_truth) == 0.0
    assert calculate_ndcg_at_k(ranked_none, ground_truth, k=2) == 0.0

