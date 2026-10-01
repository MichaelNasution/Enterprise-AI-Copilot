# Toba Copilot

**Enterprise AI Assistant/Copilot** untuk rekomendasi itinerary wisata personalisasi di Kawasan Danau Toba — dikembangkan untuk mata kuliah 10S3001 Kecerdasan Buatan, Institut Teknologi Del.

> Milestone 1 (W02): Business Problem Framing, Spesifikasi PEAS, & Baseline Search (UCS/A*)

## Tim — Kelompok 7

| NIM | Nama | Role |
|---|---|---|
| 12S24003 | Michael Pratama Nasution | AI Architect & Model Lead |
| 12S24027 | Grasia Simanullang | Integration & Interface Engineer |
| 12S24042 | Ventyola Rohati Napitupulu | Data & Knowledge Engineer |

## Latar Belakang Singkat

Dinas Kebudayaan dan Pariwisata Kabupaten Toba menghadapi fragmentasi informasi pariwisata, keterbatasan pencarian berbasis kata kunci, serta ketimpangan eksposur destinasi/UMKM lokal. Toba Copilot menjembatani kebutuhan personalisasi wisatawan dengan data resmi pariwisata melalui agen cerdas berbasis RAG (pemahaman bahasa alami) dan algoritma pencarian graf (optimasi rute).

## Struktur Repositori

```
toba-copilot/
├── search.py           # Modul baseline search (UCS & A*) — Milestone 1
├── tests/
│   └── test_search.py  # Unit test pytest
├── pyproject.toml       # Konfigurasi dependensi (Astral uv)
├── .gitignore
└── README.md
```

## Formulasi Ruang Keadaan (Milestone 1)

Masalah rute itinerary dimodelkan sebagai graf berbobot $G=(V,E)$, dengan state $x=(n, V_{visited}, b_{sisa})$, aksi berpindah antar node bertetangga dalam batas anggaran, dan goal tercapai saat seluruh destinasi target sudah dikunjungi. Detail formal (X, A, T, G, C) ada di laporan Bab III.

Diselesaikan dengan dua algoritma:
- **Uniform Cost Search (UCS)** — optimal, mengekspansi berdasarkan `g(n)` murni.
- **A\*** — optimal & lebih efisien, `f(n) = g(n) + h(n)` dengan heuristik jarak Haversine (admissible).

## Instalasi & Menjalankan (Astral uv)

```bash
# instal uv jika belum ada: https://docs.astral.sh/uv/
uv sync                 # instal seluruh dependensi dari pyproject.toml
uv run python search.py # jalankan contoh pencarian rute
uv run pytest -v        # jalankan seluruh unit test
```

## Roadmap Proyek (5 Milestone)

| Milestone | Fokus |
|---|---|
| M1 (W02) | Problem Framing, PEAS, Baseline Search — **(saat ini)** |
| M2 (W04) | Business Constraint Solver (CSP/GA) |
| M3 (W07) | Knowledge Base & Vector Search (ChromaDB) |
| M4 (W11) | Agent Pipeline (LLM + RAG + MCP) |
| M5 (W13) | Dashboard Web Interaktif (Gradio) |

## Lisensi

MIT License
