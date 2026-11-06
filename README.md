# Toba Copilot

**Enterprise AI Assistant/Copilot** untuk rekomendasi itinerary wisata personalisasi di Kawasan Danau Toba — dikembangkan untuk mata kuliah 10S3001 Kecerdasan Buatan, Institut Teknologi Del.

> **Status:** Milestone 1 (Baseline Search UCS/A\*) dan Milestone 2 (CSP Constraint Solver) selesai — Milestone 3 (Knowledge Base & Vector Search) sedang berjalan.

## Tim — Kelompok 7

| NIM | Nama | Role |
|---|---|---|
| 12S24003 | Michael Pratama Nasution | AI Architect & Model Lead |
| 12S24027 | Grasia Simanullang | Integration & Interface Engineer |
| 12S24042 | Ventyola Rohati Napitupulu | Data & Knowledge Engineer |

## Latar Belakang Singkat

Dinas Kebudayaan dan Pariwisata Kabupaten Toba menghadapi fragmentasi informasi pariwisata, keterbatasan pencarian berbasis kata kunci, serta ketimpangan eksposur destinasi/UMKM lokal. Toba Copilot menjembatani kebutuhan personalisasi wisatawan dengan data resmi pariwisata melalui agen cerdas berbasis RAG (pemahaman bahasa alami) dan algoritma pencarian graf (optimasi rute).

## Arsitektur

```mermaid
flowchart TD
    U[Wisatawan] -->|kueri bahasa alami| UI["Antarmuka Web<br/>Gradio — M5"]
    UI --> AG["Agent LLM + RAG<br/>M4"]
    AG -->|temu balik semantik| KB[("Knowledge Base<br/>ChromaDB — M3")]
    AG -->|optimasi rute| RS["Route Search UCS / A*<br/>search.py — M1"]
    AG -->|paket termurah yang valid| CS["Package Solver CSP<br/>solver.py — M2"]
    KB -.->|data destinasi, harga, UMKM| RS
    KB -.->|domain ItemWisata| CS
    AG -->|itinerary + rincian biaya| UI
```

## Struktur Repositori

```
Enterprise-AI-Copilot/
├── search.py                      # Baseline search UCS & A* — Milestone 1
├── solver.py                      # CSP / COP constraint solver — Milestone 2
├── test_search.py                 # Unit test search (admissibility & optimalitas A*)
├── test_solver.py                 # Unit test solver (edge cases)
├── analysis/
│   ├── sensitivity_analysis.py    # Analisis sensitivitas solver (reproducible, seed 42)
│   ├── sensitivity_scale.csv      # Hasil eksperimen skala domain
│   ├── sensitivity_budget.csv     # Hasil eksperimen ketatnya anggaran
│   ├── scale_space_vs_nodes.png   # Grafik ruang pencarian vs node
│   └── scale_time.png             # Grafik waktu eksekusi
├── docs/
│   └── milestone2_formal_model.md # Formalisasi matematis CSP
├── src/certan_ai/                 # Package proyek (entry point)
├── pyproject.toml / uv.lock       # Dependensi (Astral uv)
├── LICENSE
└── README.md
```

## Formulasi Ruang Keadaan (Milestone 1)

Masalah rute itinerary dimodelkan sebagai graf berbobot $G=(V,E)$, dengan state $x=(n, V_{visited}, b_{sisa})$, aksi berpindah antar node bertetangga dalam batas anggaran, dan goal tercapai saat seluruh destinasi target sudah dikunjungi. Detail formal (X, A, T, G, C) ada di laporan Bab III.

Diselesaikan dengan dua algoritma:
- **Uniform Cost Search (UCS)** — optimal, mengekspansi berdasarkan `g(n)` murni.
- **A\*** — `f(n) = g(n) + h(n)` dengan heuristik Haversine yang diskalakan `ρ = min(bobot_edge / haversine_edge)` sehingga terbukti admissible (diuji untuk semua pasangan node dan semua kombinasi target di `test_search.py`).

> Catatan: budget adalah batasan pada `g(n)`, bukan komponen state. State yang dipakai adalah `(node, visited)`.

## Formulasi CSP (Milestone 2)

Pemilihan kombinasi paket wisata (homestay, kuliner, destinasi utama, dan destinasi UMKM) diformulasikan sebagai *Constraint Optimization Problem* (COP) untuk meminimalkan total biaya dengan mempertimbangkan batasan anggaran (budget) wisatawan serta kewajiban melibatkan minimal satu UMKM lokal.
- **Generalized Arc Consistency (GAC)**: Digunakan untuk memangkas (pruning) kandidat yang secara absolut melanggar batasan global (budget & UMKM) sebelum fase pencarian dimulai.
- **Backtracking dengan Heuristik MRV**: Memprioritaskan variabel dengan kandidat paling sedikit (*Minimum Remaining Values*), dipadukan dengan *Branch-and-Bound* untuk menjamin penemuan paket wisata termurah secara mangkus.

## Milestone 3: Basis Pengetahuan Korpus & Vector Semantic Search

Pada Milestone 3 ini dibangun memori semantik perusahaan (*Enterprise Knowledge Base*) menggunakan **ChromaDB persisten** dengan ruang metrik **Cosine Similarity** dan model embedding multibahasa **`paraphrase-multilingual-MiniLM-L12-v2`** (384 dimensi), mencakup:
- **Korpus Regulasi & SOP**: 12 dokumen regulasi resmi Danau Toba + 404 record dataset fakta destinasi dan operasional (3.308 chunk terstruktur).
- **Chunking Engine**: *Word-aware recursive chunking* dengan *sliding window overlap* teruji ilmiah.
- **Pipeline Ingestion**: Skrip batch embedding dan persistent indexing ke ChromaDB disk (`src/ingest.py`).
- **Semantic Retriever & Guardrail**: Mesin pencari semantik (`src/retriever.py`) dengan dukungan *metadata filtering* dan *guardrail threshold* penolakan kueri di luar cakupan dokumen (*out-of-domain*).
- **Benchmarking Suite**: Evaluasi 5 kueri bisnis (`analysis/benchmark.py`) dengan penghitungan skor Cosine Similarity serta metrik $P@k$ dan $R@k$.

## Instalasi & Menjalankan (Astral uv)

```bash
# instal uv jika belum ada: https://docs.astral.sh/uv/
uv sync                                    # instal seluruh dependensi dari pyproject.toml
uv run python search.py                    # jalankan contoh pencarian rute (M1)
uv run python solver.py                    # jalankan contoh penyelesaian paket wisata CSP (M2)
uv run python src/ingest.py                # jalankan pipeline ingestion ChromaDB (M3)
uv run python src/retriever.py             # jalankan pengujian pencarian semantik & guardrail (M3)
uv run python analysis/benchmark.py        # jalankan evaluasi benchmark 5 kueri bisnis (M3)
uv run pytest -v                           # jalankan seluruh unit test (36 tests passed)
uv run python analysis/sensitivity_analysis.py  # reproduksi analisis sensitivitas M2
```

## Roadmap Proyek (5 Milestone)

| Milestone | Fokus |
|---|---|
| M1 (W02) | Problem Framing, PEAS, Baseline Search |
| M2 (W04) | Business Constraint Solver (CSP/GA) |
| M3 (W07) | Knowledge Base & Vector Search (ChromaDB) — **(saat ini)** |
| M4 (W11) | Agent Pipeline (LLM + RAG + MCP) |
| M5 (W13) | Dashboard Web Interaktif (Gradio) |


## Lisensi

Dirilis di bawah [MIT License](LICENSE).
