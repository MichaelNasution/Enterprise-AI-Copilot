import json
from pathlib import Path
import pytest
import chromadb
from src.ingest import run_ingestion


DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "toba_knowledge_base"
CHUNKS_PATH = "./data/chunks/chunks.jsonl"


@pytest.fixture(scope="module")
def persistent_collection():
    """Fixture untuk memuat koleksi persisten ChromaDB yang sudah di-generate."""
    assert Path(DB_PATH).exists(), f"Direktori database {DB_PATH} tidak ditemukan!"
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def test_database_persistence_and_collection_exists(persistent_collection):
    """Memastikan koneksi persisten berhasil dan koleksi toba_knowledge_base ditemukan."""
    assert persistent_collection is not None
    assert persistent_collection.name == COLLECTION_NAME


def test_collection_hnsw_metric_is_cosine(persistent_collection):
    """
    KUNCI PENILAIAN RUBRIK:
    Memastikan konfigurasi ruang metrik HNSW ChromaDB dikunci ke 'cosine'.
    """
    metadata = persistent_collection.metadata
    assert metadata is not None, "Metadata koleksi tidak boleh kosong!"
    assert metadata.get("hnsw:space") == "cosine", (
        f"Ruang jarak koleksi harus 'cosine', namun ditemukan '{metadata.get('hnsw:space')}'"
    )


def test_total_indexed_documents_matches_chunks(persistent_collection):
    """Memastikan seluruh 3.308 chunk berhasil diindeks tanpa ada data yang tercecer."""
    # Hitung jumlah baris di chunks.jsonl
    total_chunks = 0
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for _ in f:
            total_chunks += 1

    indexed_count = persistent_collection.count()
    assert indexed_count == total_chunks, (
        f"Jumlah dokumen di ChromaDB ({indexed_count}) tidak sesuai dengan jumlah di chunks.jsonl ({total_chunks})"
    )
    assert indexed_count == 3308


def test_embedding_dimensions_and_validity(persistent_collection):
    """Memastikan embedding berdimensi 384 (MiniLM) dan tersimpan dengan benar."""
    sample = persistent_collection.get(
        ids=["D01-c001"],
        include=["embeddings", "documents", "metadatas"]
    )
    assert len(sample["ids"]) == 1
    assert len(sample["documents"]) == 1
    assert len(sample["metadatas"]) == 1
    assert sample["embeddings"] is not None

    embedding = sample["embeddings"][0]
    assert len(embedding) == 384, f"Dimensi embedding harus 384, namun didapat {len(embedding)}"
    assert all(isinstance(val, float) for val in embedding[:10]), "Nilai embedding harus berupa float."


def test_metadata_fields_are_intact(persistent_collection):
    """Memastikan atribut metadata penting (kategori, doc_id/title, dsb) tersimpan utuh."""
    sample = persistent_collection.get(
        ids=["D01-c001"],
        include=["metadatas"]
    )
    meta = sample["metadatas"][0]
    assert meta["doc_id"] == "D01"
    assert meta["kategori"] == "tarif_retribusi"
    assert "source_type" in meta
    assert meta["source_type"] == "dokumen"


def test_dataset_record_metadata(persistent_collection):
    """Memastikan record dari dataset hackathon juga terindeks dengan metadata yang tepat."""
    # Ambil sampel chunk dari dataset
    results = persistent_collection.get(
        where={"source_type": "dataset"},
        limit=5,
        include=["metadatas", "documents"]
    )
    assert len(results["ids"]) > 0, "Harus ada data dengan source_type='dataset'"
    sample_meta = results["metadatas"][0]
    assert sample_meta["source_type"] == "dataset"
    assert "record_type" in sample_meta
    assert "title" in sample_meta


def test_modular_run_ingestion_with_temp_data(tmp_path):
    """
    Test TDD untuk modularitas run_ingestion:
    Fungsi harus dapat menerima parameter path kustom untuk pengujian independen.
    """
    # Siapkan file JSONL mini sementara
    temp_jsonl = tmp_path / "dummy_chunks.jsonl"
    dummy_data = [
        {"id": "test-001", "text": "Ini adalah contoh SOP penginapan Danau Toba.", "metadata": {"kategori": "uji"}},
        {"id": "test-002", "text": "Tarif retribusi masuk pelabuhan Ajibata.", "metadata": {"kategori": "uji"}}
    ]
    with open(temp_jsonl, "w", encoding="utf-8") as f:
        for item in dummy_data:
            f.write(json.dumps(item) + "\n")

    temp_db = str(tmp_path / "temp_chroma")
    
    # Jalankan ingestion ke direktori temporer
    count = run_ingestion(
        file_path=str(temp_jsonl),
        db_path=temp_db,
        collection_name="test_collection"
    )
    assert count == 2

    # Verifikasi isi database temporer
    client = chromadb.PersistentClient(path=temp_db)
    col = client.get_collection("test_collection")
    assert col.count() == 2
    assert col.metadata.get("hnsw:space") == "cosine"

