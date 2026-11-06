# Panduan Pengerjaan & Pembagian Peran Milestone 3 (W07)
**Mata Kuliah:** 10S3001 – Kecerdasan Buatan (+P) | Gasal 2026/2027  
**Proyek:** Toba Copilot (Enterprise AI Assistant & Vector Semantic Search)  
**Judul Tugas:** Pembangunan Basis Pengetahuan Korpus & Vector Semantic Search  
**Bobot Penilaian:** 9.0% Komponen Proyek (2.7% Nilai Akhir Semester)  

---

## 1. Pemahaman Singkat Milestone 3

Milestone 3 berfokus pada pembangunan memori semantik perusahaan (*Enterprise Knowledge Base*) menggunakan basis data vektor **ChromaDB**. Sistem ini mengonversi korpus regulasi, standar operasional prosedur (SOP), dan kebijakan pariwisata Danau Toba ke dalam bentuk vektor padat (*dense vector embeddings*) untuk mendukung temu balik informasi semantik (*semantic retrieval*) yang akurat serta menjadi fondasi sistem *Retrieval-Augmented Generation* (RAG).

### Komponen Rubrik Penilaian (Skala 100)
1. **Kualitas Data Korpus & Prapemrosesan (25%):**
   * Korpus representatif (minimal 10–15 dokumen SOP/regulasi bisnis).
   * Pembersihan teks rapi dan terstruktur (tidak merusak semantik/aturan).
   * Strategi *chunking* (*size* & *overlap*) memiliki landasan dan justifikasi ilmiah.
2. **Pipeline Embedding & Vector Store ChromaDB (45%):**
   * Pemilihan model *Sentence-Transformers* tepat (multilingual untuk teks Bahasa Indonesia).
   * Penyimpanan berbasis disk/persisten (`chromadb.PersistentClient`).
   * Konfigurasi ruang metrik HNSW secara eksplisit menggunakan *Cosine Similarity* (`{"hnsw": {"space": "cosine"}}`).
   * Struktur *metadata* tersimpan lengkap untuk mendukung *metadata filtering* / *hybrid search*.
3. **Akurasi & Analisis Temu Balik Semantik (30%):**
   * Kueri semantik berhasil menarik dokumen target dengan skor *Cosine Similarity* tinggi.
   * Uji coba minimal 5 kueri bisnis (kueri relevan, kueri sinonim/parafrasa, dan kueri *out-of-domain*).
   * Penanganan kueri luar domain teruji presisi melalui mekanisme ambang batas (*guardrail threshold*).
   * Analisis metrik temu balik: $P@k$ (*Precision@k*) dan $R@k$ (*Recall@k*).

---

## 2. Poin Kunci & Jebakan Teknis (Berdasarkan Materi Minggu 07)

* **Jarak Bawaan ChromaDB Bukan Cosine:**  
  ChromaDB secara *default* menggunakan metrik jarak kuadrat Euclidean (*squared* $L_2$). Koleksi **wajib** dikonfigurasi eksplisit ke ruang *cosine*:
  ```python
  collection = client.get_or_create_collection(
      name="toba_knowledge_base",
      configuration={"hnsw": {"space": "cosine"}}
  )
  ```
  Pada ruang jarak *cosine*, skor kemiripan dihitung dengan rumus:
  $$\text{Similarity} = 1 - \text{Distance}$$
* **Model Multilingual vs Monolingual English:**  
  Model bahasa Inggris murni (seperti `all-MiniLM-L6-v2`) memberikan skor kesamaan tinggi secara tidak wajar pada teks Bahasa Indonesia di luar domain (tidak memiliki *separability* yang baik). Gunakan model multibahasa resmi seperti `paraphrase-multilingual-MiniLM-L12-v2` agar pemisahan kueri sah dan kueri luar domain menjadi signifikan.
* **Stopword Removal Hati-hati:**  
  Menghapus kata-kata negasi seperti *"tidak"*, *"tanpa"*, atau *"dilarang"* dari dokumen regulasi/SOP akan membalikkan makna aturan secara fatal. Prapemrosesan harus mempertahankan konteks kalimat.
* **Pencarian Cepat ANN (HNSW):**  
  *Hierarchical Navigable Small World* (HNSW) menggantikan pencarian linear $O(N)$ menjadi skala logaritmik $O(\log N)$, menukar sedikit ketelitian demi kecepatan penelusuran di jutaan dimensi vektor.
* **Ambang Batas (*Guardrail Threshold*):**  
  Nilai ambang penolakan kueri di luar domain (misal $0.15 - 0.25$) harus diperoleh melalui kalibrasi uji empiris, bukan sekadar tebakan.

---

## 3. Pembagian Tugas Berdasarkan Peran Tim

### Tim Pengembang — Kelompok 7

| NIM | Nama | Peran (*Role*) |
|---|---|---|
| **12S24042** | **Ventyola Rohati Napitupulu** | Data & Knowledge Engineer |
| **12S24003** | **Michael Pratama Nasution** | AI Architect & Model Lead |
| **12S24027** | **Grasia Simanullang** | Integration & Interface Engineer |

---

### A. Ventyola Rohati Napitupulu — *Data & Knowledge Engineer*
**Fokus Area:** Kurasi Korpus Bisnis, Metadata, Prapemrosesan, dan Strategi Chunking (Kontribusi Rubrik 25%)

* [x] **Inisialisasi Dependensi & Lingkungan Kerja M3:**
  * Menambahkan dependensi `sentence-transformers`, `chromadb`, dan `pypdf` ke `pyproject.toml` dan `uv.lock`.
  * Mengonfigurasi `.gitignore` untuk mengabaikan artefak database vektor persisten `chroma_db/`.
* [x] **Pengumpulan Korpus Bisnis (10–15 Dokumen):**
  * Terkumpul **12 dokumen resmi** pariwisata Toba (`D01` s.d. `D12`) di `data/corpus/` lengkap dengan `corpus/manifest.csv` dan PDF asli di `corpus/raw/`.
  * Tambahan dataset **404 record fakta** destinasi, fasilitas, dan operasional pariwisata Toba di `data/corpus/hackathon_tourism_clean.jsonl`.
* [x] **Desain Skema Metadata:**
  * Metadata terstruktur lengkap (`source_type`, `doc_id`, `judul`, `jenis`, `penerbit`, `tahun`, `status`, `bahasa`, `kategori`, `url_sumber`, `chunk_index`, `char_count`, `token_count`) tertanam di setiap chunk.
* [x] **Prapemrosesan Teks (*Text Cleaning*):**
  * Modul `src/preprocess.py` dan `src/preprocess_tourism_dataset.py` berhasil dibuat untuk membersihkan watermark BPS, nomor halaman, header OCR, perbaikan hyphenation, dan redaksi PII.
  * Kata negasi (*"tidak"*, *"dilarang"*, *"tanpa"*, *"wajib"*) dipertahankan secara utuh.
  * Unit tests `tests/test_preprocess.py` & `tests/test_preprocess_tourism_dataset.py` lulus (15 tests passed).
* [x] **Implementasi & Justifikasi Chunking:**
  * Modul `src/chunking.py` mengimplementasikan *word-aware recursive chunking* dengan sliding window overlap menggunakan tokenizer `paraphrase-multilingual-MiniLM-L12-v2`.
  * Eksperimen grid parameter chunking (300, 400, 500, 600, 800) x overlap (0, 50, 100) terdokumentasi di `data/chunks/chunking_experiment.csv`, `chunk_stats.csv`, dan grafik `token_distribution.png`.
  * Menghasilkan **3.308 chunk siap indeks** di `data/chunks/chunks.jsonl`.
  * Unit tests `tests/test_chunking.py` lulus (7 tests passed).
* [ ] **Kontribusi Laporan Teknis:**
  * Menyusun Bab II (Kualitas Korpus & Prapemrosesan) pada draf laporan `Grup07-Tugas03.pdf` menyajikan tabel manifest, tabel eksperimen chunking, dan grafik distribusi token.

---

### B. Michael Pratama Nasution — *AI Architect & Model Lead*
**Fokus Area:** Seleksi Model Vektor, Desain Vector Store, Formulasi Guardrail, dan Analisis Matematis (Kontribusi Rubrik 45% & 30%)

* [x] **Analisis & Seleksi Model Embedding:**
  * Pemilihan model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensi, mendukung multibahasa/Bahasa Indonesia dengan pemisahan semantik optimal).
* [ ] **Arsitektur ChromaDB & Ruang Metrik:**
  * Memastikan konfigurasi ruang jarak HNSW eksplisit menggunakan `cosine` pada koleksi persisten.
* [ ] **Perancangan Ambang Batas (*Guardrail Threshold*):**
  * Merancang formulasi batas toleransi skor similarity empiris (misal $\ge 0.20$ diterima, $< 0.20$ ditolak) untuk mendeteksi kueri luar cakupan korpus (*out-of-domain*).
* [ ] **Evaluasi Matematis & Tolok Ukur Temu Balik:**
  * Menghitung nilai metrik evaluasi temu balik $P@k$ (*Precision@k*) dan $R@k$ (*Recall@k*) untuk 5 kueri uji bisnis.
* [ ] **Penyusunan Laporan Teknis PDF (`Grup07-Tugas03.pdf`):**
  * Memimpin penulisan dokumen laporan teknis (Bab Teori NLP, Arsitektur Vektor, Analisis Metrik Cosine vs $L_2$, HNSW ANN, dan Pembahasan Hasil Evaluasi).

---

### C. Grasia Simanullang — *Integration & Interface Engineer*
**Fokus Area:** Rekayasa Pipeline Ingestion, Modul Retriever, Pengujian Otomatis, dan Repositori (Kontribusi Rubrik 45% & 30%)

* [x] **Sinkronisasi Repositori & Dependensi Proyek (Inisial):**
  * Dependensi `chromadb`, `sentence-transformers`, `pypdf` terpasang di `pyproject.toml` via Astral `uv`.
* [ ] **Pembangunan Pipeline Ingestion End-to-End (`src/ingest.py`):**
  * Mengembangkan skrip ingestion yang membaca 3.308 chunk dari `data/chunks/chunks.jsonl`.
  * Melakukan *batch embedding* menggunakan `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")`.
  * Mengindeks seluruh vektor beserta metadata ke dalam `chromadb.PersistentClient(path="./data/chroma_db")` dengan koleksi `toba_knowledge_base` berspasi `cosine`.
* [ ] **Pengembangan Modul Pencarian Semantik (`src/retriever.py`):**
  * Membangun kelas/fungsi pencarian semantik yang menerima kueri teks, menghasilkan query embedding, mengeksekusi pencarian k-NN, dan menghitung `similarity = 1 - distance`.
  * Mendukung filter metadata (`where={"kategori": "tarif_retribusi"}` atau `where={"source_type": "dokumen"}`).
  * Mengintegrasikan *guardrail threshold* penolakan kueri di luar cakupan dokumen.
* [ ] **Penyusunan Pengujian Benchmark (`tests/test_retrieval.py` / `benchmark.py`):**
  * Menguji 5 skenario kueri bisnis wajib:
    1. Kueri Relevan Harfiah (*Exact/Keyword match*)
    2. Kueri Relevan Parafrasa/Sinonim (*Semantic variation*)
    3. Kueri Berbasis Spesifikasi Kategori dengan Filter Metadata
    4. Kueri Ambigu
    5. Kueri *Out-of-Domain* (luar cakupan dokumen)
* [ ] **Pembaruan Dokumentasi `README.md`:**
  * Menuliskan cara menjalankan `uv run python src/ingest.py` dan cara melakukan kueri via `retriever.py`.

---

## 4. Roadmap Eksekusi Milestone 3 (Status Terkini)

```mermaid
gantt
    title Roadmap Eksekusi Milestone 3 - Toba Copilot
    dateFormat  YYYY-MM-DD
    section Fase 1: Data & Korpus
    Kurasi 10-15 Dokumen SOP & Metadata (Ventyola)    :done, f1, 2026-11-05, 2d
    section Fase 2: Chunking & Model
    Word-aware Recursive Chunker (Ventyola)           :done, f2a, 2026-11-06, 1d
    Konfigurasi Sentence-Transformers (Michael)       :done, f2b, 2026-11-06, 1d
    section Fase 3: ChromaDB Vector Store
    Pipeline Ingestion & Indexing Persisten (Grasia)  :active, f3, 2026-11-06, 2d
    section Fase 4: Retriever & Benchmark
    Engine Retrieval & Guardrail Threshold (Grasia)   :f4a, after f3, 1d
    Eksekusi 5 Kueri Benchmark, P@k, R@k (Michael)    :f4b, after f4a, 1d
    section Fase 5: Pelaporan & Deliverables
    Penyusunan Laporan PDF Grup07-Tugas03 (Michael)   :f5a, after f4b, 2d
    Finalisasi README & Push Repo GitHub (Tim)        :f5b, after f5a, 1d
```

### Rincian Fase & Checklist Capaian

| Fase | Target Luaran & Aktivitas Utama | Penanggung Jawab | Status |
|---|---|---|:---:|
| **Fase 1: Fondasi Data & Korpus Enterprise** | • 12 dokumen regulasi + 404 record pariwisata Toba di `data/corpus/`<br>• Skema metadata lengkap tertanam di manifest & jsonl<br>• Normalisasi teks & preservasi kata negasi via `preprocess.py` | **Ventyola Rohati** | 🟢 **SELESAI (100%)** |
| **Fase 2: Arsitektur Model & Chunking Engine** | • Pemilihan model `paraphrase-multilingual-MiniLM-L12-v2`<br>• Word-aware recursive chunker `src/chunking.py`<br>• Eksperimen parameter chunking & 3.308 chunk di `chunks.jsonl` | **Ventyola Rohati** & **Michael Pratama** | 🟢 **SELESAI (100%)** |
| **Fase 3: Persistent Indexing & ChromaDB** | • Skrip `src/ingest.py` (baca 3.308 chunk $\rightarrow$ batch embedding $\rightarrow$ ChromaDB)<br>• Penyimpanan disk persisten `chromadb.PersistentClient`<br>• Konfigurasi eksplisit HNSW ruang `cosine` | **Grasia Simanullang** | 🟡 **PRIORITAS BERIKUTNYA (0%)** |
| **Fase 4: Semantic Retriever & Benchmarking** | • Skrip `src/retriever.py` (cosine similarity + metadata filter)<br>• Ambang batas (*guardrail threshold*) penolakan kueri luar domain<br>• Pengujian 5 kueri bisnis & penghitungan metrik $P@k$ dan $R@k$ | **Grasia Simanullang** & **Michael Pratama** | ⚪ *Pending* (0%) |
| **Fase 5: Laporan Teknis PDF & Finalisasi** | • Kompilasi dokumen laporan teknis `Grup07-Tugas03.pdf`<br>• Pembaruan panduan eksekusi di `README.md`<br>• Pengujian unit test via `pytest` & final push ke GitHub | **Michael Pratama** (Lead) & Seluruh Tim | ⚪ *Pending* (0%) |

---

## 5. Alur Kerja Antar-Peran (Collaboration Workflow)

```mermaid
flowchart TD
    A[Ventyola: Kurasi 12 Dokumen SOP & Metadata] -->|Selesai| B[Ventyola: Preprocessing & Word-Aware Chunking]
    B -->|Tersedia: chunks.jsonl 3.308 chunks| C[Michael: Inisialisasi Multilingual Model & Konfigurasi HNSW Cosine]
    C -->|Langkah Saat Ini| D[Grasia: Skrip Ingestion & Persistent Indexing ChromaDB]
    D --> E[Grasia: Implementasi Modul Retriever & Guardrail]
    E --> F[Grasia & Michael: Eksekusi 5 Kueri Benchmark]
    F --> G[Michael: Analisis Metrik P@k, R@k & Penulisan Laporan PDF]
    G --> H[Tim: Finalisasi Repositori GitHub & Pengumpulan Dokumen]
```

---

## 6. Rencana Struktur Berkas Proyek

```
Enterprise-AI-Copilot/
├── data/
│   ├── corpus/                     # Korpus dokumen SOP/Regulasi & dataset pariwisata (Ventyola - Selesai)
│   │   ├── D01_perda_tobasa_12_2012_retribusi_rekreasi.txt
│   │   ├── ... (12 dokumen regulasi resmi)
│   │   └── hackathon_tourism_clean.jsonl
│   ├── chunks/                     # Hasil chunking & eksperimen (Ventyola - Selesai)
│   │   ├── chunks.jsonl            # 3.308 chunk siap indeks ke ChromaDB
│   │   ├── chunking_experiment.csv # Data eksperimen grid search chunking
│   │   └── token_distribution.png  # Visualisasi distribusi token
│   └── chroma_db/                  # Database persisten ChromaDB (Grasia - Target Fase 3)
├── src/
│   ├── preprocess.py               # Pembersihan teks PDF regulasi (Ventyola - Selesai)
│   ├── preprocess_tourism_dataset.py # Pembersihan dataset hackathon (Ventyola - Selesai)
│   ├── chunking.py                 # Word-aware recursive chunker (Ventyola - Selesai)
│   ├── ingest.py                   # Pipeline ingestion & embedding (Grasia - Target Fase 3)
│   └── retriever.py                # Semantic search & guardrail (Grasia & Michael - Target Fase 4)
├── tests/
│   ├── test_preprocess.py          # Unit test cleaning PDF (Ventyola - Selesai)
│   ├── test_preprocess_tourism_dataset.py # Unit test cleaning dataset (Ventyola - Selesai)
│   ├── test_chunking.py            # Unit test chunking logic (Ventyola - Selesai)
│   └── test_retrieval.py           # Unit test & benchmark kueri bisnis (Grasia & Michael - Target Fase 4)
├── docs/
│   ├── milestone3_plan.md          # Dokumen perencanaan ini (Updated)
│   └── Grup07-Tugas03.pdf          # Laporan teknis akhir (Michael Lead - Target Fase 5)
├── pyproject.toml                  # Dependensi uv (chromadb, sentence-transformers)
└── README.md                       # Dokumentasi instruksi eksekusi
```

---

## 7. Luaran yang Harus Dikumpulkan

1. **Laporan Teknis PDF:** Berkas `Grup07-Tugas03.pdf` yang diunggah ke portal e-Course Del, memuat:
   * Landasan teori evolusi NLP, embedding, dan perbandingan ruang vektor.
   * Deskripsi korpus enterprise beserta justifikasi parameter *chunking* (tabel eksperimen & grafik token).
   * Arsitektur model embedding dan ChromaDB (ruang metrik cosine & disk persisten).
   * Tabel hasil pengujian 5 kueri bisnis (disertai analisis skor *similarity*, presisi penolakan kueri *out-of-domain*, serta metrik $P@k$ dan $R@k$).
2. **Repositori GitHub:**
   * Seluruh kode sumber (`preprocess.py`, `chunking.py`, `ingest.py`, `retriever.py`, skrip uji).
   * Korpus dokumen enterprise di folder `data/corpus/` dan hasil chunk di `data/chunks/`.
   * Berkas `README.md` terbaharui dengan dokumentasi lengkap cara menjalankan pipeline ingestion dan pengujian temu balik semantik.
