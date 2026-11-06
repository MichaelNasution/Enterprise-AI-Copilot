import json

from src.chunking import (
    _dataset_metadata,
    _document_metadata,
    _document_chunk_id,
    split_text,
)


class CharacterTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text) + (2 if add_special_tokens else 0)))}


def _words(text: str) -> list[str]:
    import re

    return re.findall(r"\S+", text)


def test_chunks_respect_character_limit_except_for_single_long_word() -> None:
    text = " ".join(f"kata{i}" for i in range(120))
    chunks = split_text(text, 120, 0)
    longest_word = max(map(len, _words(text)))
    assert all(len(chunk) <= 120 + longest_word for chunk in chunks)

    long_word = "x" * 140
    assert split_text(long_word, 100, 0) == [long_word]


def test_word_sequence_is_preserved_in_order_without_overlap() -> None:
    text = " ".join(f"kata{i}" for i in range(100))
    chunks = split_text(text, 120, 0)
    assert [word for chunk in chunks for word in _words(chunk)] == _words(text)


def test_overlap_contains_previous_chunk_tail_and_zero_overlap_does_not() -> None:
    text = " ".join(f"kata{i}" for i in range(100))
    overlapping = split_text(text, 120, 30)
    assert len(overlapping) > 1
    previous_words = _words(overlapping[0])
    following_words = _words(overlapping[1])
    assert any(
        following_words[:count] == previous_words[-count:]
        for count in range(1, min(len(previous_words), len(following_words)) + 1)
    )

    without_overlap = split_text(text, 120, 0)
    assert _words(without_overlap[1])[:2] != _words(without_overlap[0])[-2:]


def test_short_article_heading_is_joined_to_following_content() -> None:
    text = "Pasal 5\n" + ("Ketentuan ini wajib dipenuhi oleh setiap pengelola. " * 5)
    chunks = split_text(text, 180, 0)
    assert all(chunk != "Pasal 5" for chunk in chunks)
    assert any(chunk.startswith("Pasal 5") and "Ketentuan" in chunk for chunk in chunks)


def test_negations_are_preserved() -> None:
    text = "Pengunjung tidak boleh masuk dan dilarang merusak fasilitas umum. " * 8
    assert "tidak" in " ".join(split_text(text, 150, 25))
    assert "dilarang" in " ".join(split_text(text, 150, 25))


def test_short_text_produces_one_chunk() -> None:
    assert split_text("Teks ini cukup singkat.", 300, 50) == ["Teks ini cukup singkat."]


def test_metadata_is_flat_and_ids_are_deterministic() -> None:
    tokenizer = CharacterTokenizer()
    document = {
        "doc_id": "D02",
        "judul": "Peraturan",
        "jenis": "peraturan",
        "penerbit": "Instansi",
        "tahun": "2014",
        "status": "berlaku",
        "bahasa": "id",
        "kategori": "layanan",
        "url_sumber": "",
    }
    metadata = _document_metadata(document, 1, "Isi Pasal", tokenizer)
    assert metadata["tahun"] == 2014
    assert "url_sumber" not in metadata
    assert all(value is not None and not isinstance(value, dict) for value in metadata.values())

    record = {
        "record_id": "abc123",
        "record_type": "attraction",
        "title": "Pantai",
        "text": "Pantai buka setiap hari.",
        "source_sheet": "wisata-metadata",
        "lat": 2.3,
        "lon": 99.0,
        "rating": None,
        "harga_min": 0,
        "harga_max": 0,
        "status": "",
        "kabupaten": "Toba",
    }
    dataset_metadata = _dataset_metadata(record, 1, record["text"], tokenizer)
    assert "rating" not in dataset_metadata
    assert "status" not in dataset_metadata
    assert all(not isinstance(value, dict) for value in dataset_metadata.values())

    first_id = _document_chunk_id("D02", 7)
    assert first_id == _document_chunk_id("D02", 7)
    assert first_id != _document_chunk_id("D02", 8)
