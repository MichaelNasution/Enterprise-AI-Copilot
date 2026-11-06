import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import chromadb
from sentence_transformers import SentenceTransformer


class TobaKnowledgeRetriever:
    """
    Mesin pencarian semantik (Semantic Search Engine) untuk basis pengetahuan pariwisata Toba
    berbasis vektor embedding dengan ChromaDB persisten dan HNSW Cosine distance.
    Dilengkapi dengan penyaringan metadata dan mekanisme guardrail penolakan kueri out-of-domain.
    """

    def __init__(
        self,
        db_path: str = "./data/chroma_db",
        collection_name: str = "toba_knowledge_base",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        threshold: float = 0.20
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.threshold = threshold

        # Inisialisasi klien ChromaDB persisten dan ambil koleksi
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_collection(name=self.collection_name)

        # Inisialisasi model SentenceTransformer multibahasa
        self.model = SentenceTransformer(self.model_name)

    def _build_where_filter(
        self,
        category: Optional[str] = None,
        source_type: Optional[str] = None,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Menyusun struktur filter metadata ChromaDB."""
        filters = []
        if where_filter:
            if "$and" in where_filter:
                filters.extend(where_filter["$and"])
            else:
                filters.append(where_filter)
        if category:
            filters.append({"kategori": category})
        if source_type:
            filters.append({"source_type": source_type})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    def search(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
        source_type: Optional[str] = None,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Melakukan pencarian semantik terhadap basis data pengetahuan.

        Args:
            query: Teks pertanyaan / kueri pencarian.
            top_k: Jumlah dokumen teratas yang ingin dikembalikan.
            category: Filter opsional berdasarkan field 'kategori' metadata.
            source_type: Filter opsional berdasarkan field 'source_type' metadata.
            where_filter: Filter kustom dict ChromaDB.

        Returns:
            List[Dict]: Daftar potongan dokumen beserta metadata dan skor cosine similarity.
        """
        # Konversi teks kueri menjadi representasi vektor dense
        query_embedding = self.model.encode(query).tolist()

        # Bangun filter where
        effective_where = self._build_where_filter(
            category=category,
            source_type=source_type,
            where_filter=where_filter
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=effective_where,
            include=["documents", "metadatas", "distances"]
        )

        docs = results.get("documents")
        metas = results.get("metadatas")
        dists = results.get("distances")
        doc_ids = results.get("ids")

        if not docs or not docs[0]:
            return []

        retrieved_docs = []
        for i in range(len(docs[0])):
            doc_text = docs[0][i]
            doc_meta = metas[0][i] if metas and metas[0] else {}
            doc_dist = dists[0][i] if dists and dists[0] else 1.0
            doc_id = doc_ids[0][i] if doc_ids and doc_ids[0] else f"doc_{i}"

            # Menghitung nilai kesamaan: Cosine Similarity = 1 - Distance
            doc_sim = float(1.0 - doc_dist)

            retrieved_docs.append({
                "id": doc_id,
                "document": doc_text,
                "dokumen": doc_text,  # alias bahasa Indonesia
                "metadata": doc_meta,
                "similarity": doc_sim,
                "distance": float(doc_dist)
            })

        return retrieved_docs

    def search_with_guardrail(
        self,
        query: str,
        top_k: int = 3,
        threshold: Optional[float] = None,
        category: Optional[str] = None,
        source_type: Optional[str] = None,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Melakukan pencarian semantik dengan validasi guardrail threshold empiris.
        Kueri di luar domain dengan kesamaan di bawah threshold akan ditolak.

        Returns:
            Dict berisi status ('accepted' / 'rejected'), pesan keterangan, dan hasil pencarian.
        """
        active_threshold = threshold if threshold is not None else self.threshold
        results = self.search(
            query=query,
            top_k=top_k,
            category=category,
            source_type=source_type,
            where_filter=where_filter
        )

        if not results:
            return {
                "status": "rejected",
                "message": "Di luar cakupan dokumen korpus",
                "similarity": 0.0,
                "results": []
            }

        top_similarity = results[0]["similarity"]

        # Validasi Guardrail Threshold
        if top_similarity < active_threshold:
            return {
                "status": "rejected",
                "message": (
                    f"Di luar cakupan dokumen korpus (skor kesamaan {top_similarity:.4f} "
                    f"di bawah ambang batas {active_threshold:.2f})"
                ),
                "similarity": top_similarity,
                "results": []
            }

        return {
            "status": "accepted",
            "message": "Pencarian berhasil",
            "similarity": top_similarity,
            "results": results
        }


# Singleton instance retriever untuk penggunaan modular
_default_retriever: Optional[TobaKnowledgeRetriever] = None


def get_default_retriever() -> TobaKnowledgeRetriever:
    """Mengambil atau menginisialisasi singleton instance TobaKnowledgeRetriever."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = TobaKnowledgeRetriever()
    return _default_retriever


def query_knowledge_base(
    query_text: str,
    top_k: int = 3,
    where_filter: Optional[Dict[str, Any]] = None,
    threshold: float = 0.20
) -> Union[List[Dict[str, Any]], str]:
    """
    Fungsi antarmuka pencarian semantik utama sesuai spesifikasi Fase 4.

    Args:
        query_text: Teks kueri pencarian pengguna.
        top_k: Jumlah dokumen teratas yang dicari.
        where_filter: Filter metadata ChromaDB (opsional).
        threshold: Ambang batas toleransi penolakan guardrail.

    Returns:
        List[Dict]: Daftar dokumen relevan jika lolos guardrail.
        str: Pesan penolakan 'Di luar cakupan dokumen korpus' jika di luar domain.
    """
    retriever = get_default_retriever()
    guardrail_response = retriever.search_with_guardrail(
        query=query_text,
        top_k=top_k,
        threshold=threshold,
        where_filter=where_filter
    )

    if guardrail_response["status"] == "rejected":
        return "Di luar cakupan dokumen korpus"

    return guardrail_response["results"]


def calculate_precision_at_k(retrieved: list, ground_truth: set, k: int) -> float:
    """
    Menghitung Precision@k (P@k):
    Proporsi dokumen relevan di antara k dokumen teratas yang terambil.
    Formula: |retrieved[:k] ∩ ground_truth| / k
    """
    if k <= 0:
        return 0.0
    top_k_retrieved = retrieved[:k]
    relevant_retrieved = [item for item in top_k_retrieved if item in ground_truth]
    return len(relevant_retrieved) / k


def calculate_recall_at_k(retrieved: list, ground_truth: set, k: int) -> float:
    """
    Menghitung Recall@k (R@k):
    Proporsi dokumen relevan yang berhasil terambil dari seluruh dokumen relevan yang ada.
    Formula: |retrieved[:k] ∩ ground_truth| / |ground_truth|
    """
    if not ground_truth:
        return 0.0
    top_k_retrieved = retrieved[:k]
    relevant_retrieved = [item for item in top_k_retrieved if item in ground_truth]
    return len(relevant_retrieved) / len(ground_truth)


if __name__ == "__main__":
    print("\n--- Uji Coba Mesin Pencari Semantik & Guardrail (Fase 4) ---")

    # Skenario 1: Kueri yang relevan (lolos guardrail)
    print("\n[Kueri 1 - Relevan Harfiah]: 'Apa saja destinasi wisata di Danau Toba?'")
    hasil_1 = query_knowledge_base("Apa saja destinasi wisata di Danau Toba?")
    if isinstance(hasil_1, list):
        print(f"Berhasil menemukan {len(hasil_1)} dokumen:")
        for idx, item in enumerate(hasil_1, 1):
            print(f"  {idx}. [Sim: {item['similarity']:.4f}] {item['dokumen'][:90]}...")
    else:
        print(hasil_1)

    # Skenario 2: Kueri luar domain (terblokir guardrail)
    print("\n[Kueri 2 - Out-of-Domain]: 'Rumus fisika gaya gravitasi newton'")
    hasil_2 = query_knowledge_base("Rumus fisika gaya gravitasi newton", threshold=0.25)
    print(f"Respons Guardrail: {hasil_2}")

    # Skenario 3: Menggunakan filter metadata
    print("\n[Kueri 3 - Metadata Filter]: 'Berapa biaya masuknya?' (kategori='tarif_retribusi')")
    hasil_3 = query_knowledge_base(
        query_text="Berapa biaya masuknya?",
        top_k=2,
        where_filter={"kategori": "tarif_retribusi"}
    )
    if isinstance(hasil_3, list):
        print(f"Berhasil menemukan {len(hasil_3)} dokumen berfilter:")
        for idx, item in enumerate(hasil_3, 1):
            print(f"  {idx}. [Sim: {item['similarity']:.4f} | Kategori: {item['metadata'].get('kategori')}] {item['dokumen'][:90]}...")
    else:
        print(hasil_3)
