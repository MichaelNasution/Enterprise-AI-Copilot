import json
from pathlib import Path
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def run_ingestion(
    file_path: str = "data/chunks/chunks.jsonl",
    db_path: str = "./data/chroma_db",
    collection_name: str = "toba_knowledge_base",
    batch_size: int = 64,
    batch_size_db: int = 500,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> int:
    """
    Menjalankan pipeline ingestion teks korpus pariwisata Toba ke ChromaDB persisten.

    Langkah-langkah:
    1. Membaca potongan teks (chunks) dan metadata dari file JSONL.
    2. Menghasilkan dense vector embeddings menggunakan model Sentence-Transformers multibahasa.
    3. Menginisialisasi disk persisten ChromaDB dengan ruang metrik Cosine eksplisit.
    4. Melakukan batch upsert dokumen, embedding, dan metadata ke dalam koleksi.

    Returns:
        int: Jumlah total dokumen yang berhasil diindeks.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Berkas chunks {file_path} tidak ditemukan!")

    # 1. Membaca potongan teks dari JSONL
    documents = []
    metadatas = []
    ids = []

    print(f"Membaca data dari {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line_str = line.strip()
            if not line_str:
                continue
            chunk = json.loads(line_str)
            documents.append(chunk.get("text", chunk.get("content", "")))
            metadatas.append(chunk.get("metadata", {"source": "toba_dataset"}))
            ids.append(str(chunk.get("id", f"doc_{i}")))

    total_docs = len(documents)
    print(f"Berhasil membaca {total_docs} potongan teks.")
    if total_docs == 0:
        print("Peringatan: Berkas JSONL tidak memuat dokumen.")
        return 0

    # 2. Melakukan dense embedding secara batch
    print(f"Memuat model SentenceTransformer ('{model_name}')...")
    model = SentenceTransformer(model_name)

    print("Menghasilkan embeddings (secara batch)...")
    embeddings = model.encode(documents, batch_size=batch_size, show_progress_bar=True).tolist()

    # 3. Mengindeks ke disk persisten ChromaDB
    print(f"Menyiapkan ChromaDB Persistent Client di '{db_path}'...")
    client = chromadb.PersistentClient(path=db_path)

    # Kunci Penilaian: Mengunci konfigurasi ruang jarak ke Cosine eksplisit
    # Mengisi configuration dan metadata agar kompatibel di seluruh versi ChromaDB
    collection = client.get_or_create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Memasukkan data ke dalam koleksi (Upsert secara batching)
    print(f"Mengindeks {total_docs} dokumen ke dalam koleksi '{collection_name}'...")
    for i in tqdm(range(0, total_docs, batch_size_db), desc="Indexing"):
        collection.upsert(
            ids=ids[i : i + batch_size_db],
            embeddings=embeddings[i : i + batch_size_db],
            documents=documents[i : i + batch_size_db],
            metadatas=metadatas[i : i + batch_size_db]
        )

    print(f"Pipeline Ingestion Fase 3 selesai! Total {total_docs} dokumen terindeks di ChromaDB ({db_path}).")
    return total_docs


if __name__ == "__main__":
    run_ingestion()
