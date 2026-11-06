import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def run_ingestion():
    # 1. Membaca potongan teks dari JSONL
    file_path = "data/chunks/chunks.jsonl"
    documents = []
    metadatas = []
    ids = []

    print(f"Membaca data dari {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            chunk = json.loads(line)
            # Sesuaikan key "text", "metadata", dan "id" dengan format asli JSONL Anda
            documents.append(chunk.get("text", chunk.get("content", "")))
            metadatas.append(chunk.get("metadata", {"source": "toba_dataset"}))
            ids.append(str(chunk.get("id", f"doc_{i}")))
            
    print(f"Berhasil membaca {len(documents)} potongan teks.")

    # 2. Melakukan dense embedding secara batch
    print("Memuat model SentenceTransformer ('paraphrase-multilingual-MiniLM-L12-v2')...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    print("Menghasilkan embeddings (secara batch)...")
    # batch_size bisa disesuaikan dengan kapasitas RAM/VRAM
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=True).tolist()

    # 3. Mengindeks ke disk persisten ChromaDB
    print("Menyiapkan ChromaDB Persistent Client...")
    client = chromadb.PersistentClient(path="./data/chroma_db")

    # Kunci Penilaian: Mengunci konfigurasi ruang jarak ke Cosine
    # Catatan: API standar ChromaDB menggunakan parameter `metadata` untuk konfigurasi hnsw
    collection = client.get_or_create_collection(
        name="toba_knowledge_base",
        metadata={"hnsw:space": "cosine"} 
    )

    # 4. Memasukkan data ke dalam koleksi (Upsert)
    print("Mengindeks dokumen ke dalam koleksi 'toba_knowledge_base'...")
    batch_size_db = 500
    
    # Proses upsert dilakukan secara batching untuk mencegah kelebihan muatan memori
    for i in tqdm(range(0, len(documents), batch_size_db), desc="Indexing"):
        collection.upsert(
            ids=ids[i : i + batch_size_db],
            embeddings=embeddings[i : i + batch_size_db],
            documents=documents[i : i + batch_size_db],
            metadatas=metadatas[i : i + batch_size_db]
        )

    print("Pipeline Ingestion Fase 3 selesai!")

if __name__ == "__main__":
    run_ingestion()