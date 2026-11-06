from src.preprocess import (
    clean_page,
    is_toc_page,
    join_lines,
    remove_repeated_headers_footers,
)


def test_clean_page_removes_page_number_and_copy_code_but_keeps_year() -> None:
    cleaned = clean_page("Judul\n-2-\nSK No 226014 A\n2024")
    assert cleaned == "Judul\n2024"


def test_clean_page_removes_bps_watermark_standalone_and_inline() -> None:
    cleaned = clean_page(
        "https://tobakab.bps.go.id\nData tersedia https://sumut.bps.go.id"
    )
    assert cleaned == "Data tersedia"


def test_join_lines_handles_hyphenated_words_and_reduplication() -> None:
    assert join_lines("pen-\nyusunan") == "penyusunan"
    assert join_lines("kupu-\nkupu") == "kupu-kupu"


def test_join_lines_keeps_list_items_and_articles_separate() -> None:
    assert join_lines("Kalimat pengantar\n a. Butir pertama") == (
        "Kalimat pengantar\na. Butir pertama"
    )
    assert join_lines("Kalimat pengantar\nPasal 3") == (
        "Kalimat pengantar\nPasal 3"
    )


def test_negation_words_survive_cleaning() -> None:
    text = "tidak boleh tanpa izin; dilarang, wajib, bukan pilihan, jangan"
    assert clean_page(text) == text


def test_ocr_header_variants_are_removed_without_removing_article_titles() -> None:
    pages = [
        "REPUBLIK INDONESIA\nPasal 1\nAturan berlaku khusus untuk wilayah timur.",
        "REPUELIK INDONESIA\nPasal 2\nSetiap pelaku usaha mencatat kunjungan harian.",
        "REFUBLIK INDONESIA\nPasal 3\nPengelola menyediakan fasilitas bagi pengunjung.",
        "REPUBLIK INDONESIA\nPasal 4\nEvaluasi dilakukan secara berkala setiap tahun.",
    ]
    cleaned = remove_repeated_headers_footers(pages)
    assert all("INDONESIA" not in page for page in cleaned)
    assert all(f"Pasal {index}" in page for index, page in enumerate(cleaned, 1))


def test_pasals_at_page_top_are_not_removed_as_repeated_headers() -> None:
    pages = [
        "Pasal 11\nKetentuan tentang layanan pengunjung di kawasan.",
        "Pasal 12\nPencatatan kegiatan dilakukan setiap bulan kalender.",
        "Pasal 13\nPemantauan sarana dilakukan oleh petugas berwenang.",
        "Pasal 14\nPelaporan hasil disampaikan kepada kepala daerah.",
    ]
    cleaned = remove_repeated_headers_footers(pages)
    assert [page.splitlines()[0] for page in cleaned] == [
        "Pasal 11",
        "Pasal 12",
        "Pasal 13",
        "Pasal 14",
    ]


def test_table_of_contents_page_is_detected() -> None:
    page = (
        "DAFTAR ISI\n"
        "Pendahuluan ........................................ 3\n"
        "Ketentuan Umum ..................................... 7\n"
        "Pelaksanaan ........................................ 12\n"
        "Penutup"
    )
    assert is_toc_page(page)
    assert not is_toc_page("Judul\nIsi kalimat yang tidak berisi titik penuntun.")
