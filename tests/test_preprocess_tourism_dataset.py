import json

from src import preprocess_tourism_dataset
from src.preprocess_tourism_dataset import (
    _parse_coordinates,
    _parse_price,
    _parse_rating,
    _review_record,
    _redact_review_text,
)


def test_review_record_does_not_store_reviewer_identity() -> None:
    keys = [
        "place_name",
        "reviewer_id",
        "name",
        "reviewer_rating",
        "review_text",
        "published_at",
        "collect_date",
        "reviewer_type",
    ]
    values = [
        "Homestay Toba",
        "private-id",
        "Nama Reviewer",
        "5.0",
        "Tempat nyaman untuk keluarga saat liburan.",
        "a day ago",
        "45866.0",
        "Liburan, Teman",
    ]
    record, redactions, was_too_short = _review_record(
        "resto-hotel-v2", keys, values
    )

    assert record is not None
    assert redactions == 0
    assert not was_too_short
    assert record["record_type"] == "opinion"
    assert "Ulasan pengunjung (opini)" in record["text"]
    assert "Nama Reviewer" not in repr(record)
    assert "private-id" not in repr(record)
    assert "reviewer_type" not in record["attributes"]


def test_review_text_redacts_email_and_phone_number() -> None:
    cleaned, count = _redact_review_text(
        "Hubungi wisatawan@example.com atau 0812-3456-7890"
    )

    assert cleaned == "Hubungi [email dihapus] atau [nomor telepon dihapus]"
    assert count == 2


def test_price_parser_handles_rupiah_and_unreadable_values() -> None:
    assert _parse_price("Gratis") == (0, 0)
    assert _parse_price("5.000 - 10.000") == (5000, 10000)
    assert _parse_price("70000") == (70000, 70000)
    assert _parse_price("100000.0") == (100000, 100000)
    assert _parse_price("Rp 25.000") == (25000, 25000)
    assert _parse_price("harga tidak terbaca") is None


def test_rating_parser_rejects_unreadable_and_out_of_range_values() -> None:
    assert _parse_rating("4,3") == 4.3
    assert _parse_rating("abc") is None
    assert _parse_rating("7") is None


def test_coordinate_parser_reads_lat_long_pair() -> None:
    assert _parse_coordinates({"lat_long": "2.3494, 99.0403"}) == (
        2.3494,
        99.0403,
    )


def test_short_reviews_are_dropped_and_six_word_reviews_are_kept() -> None:
    keys = ["place_name", "reviewer_id", "name", "reviewer_rating", "review_text"]
    short_record, _, too_short = _review_record(
        "wisata-v2",
        keys,
        ["Pantai Toba", "id1", "Reviewer", "5.0", "Pantai sangat indah"],
    )
    long_record, _, long_too_short = _review_record(
        "wisata-v2",
        keys,
        ["Pantai Toba", "id2", "Reviewer", "5.0", "Tempat ini sangat nyaman untuk keluarga"],
    )

    assert short_record is None
    assert too_short
    assert long_record is not None
    assert not long_too_short


def test_opinions_are_written_only_to_private_file(tmp_path, monkeypatch) -> None:
    sheets = {
        "wisata-metadata": [
            (1, ["place-name", "lat-long", "place-rating", "entry-fee", "status", "address"]),
            (
                2,
                [
                    "Pantai Toba",
                    "2.3494, 99.0403",
                    "4,3",
                    "5.000 - 10.000",
                    "BEROPERASI",
                    "Balige, Toba, Sumatera Utara",
                ],
            ),
        ],
        "wisata-v2": [
            (1, ["place-name", "reviewer-id", "name", "reviewer-rating", "review-text"]),
            (2, ["Pantai Toba", "id1", "Nama", "5.0", "Pantai sangat indah"]),
            (
                3,
                [
                    "Pantai Toba",
                    "id2",
                    "Nama",
                    "5.0",
                    "Tempat ini sangat nyaman untuk keluarga",
                ],
            ),
        ],
    }
    monkeypatch.setattr(preprocess_tourism_dataset, "read_workbook", lambda _: sheets)
    fact_path = tmp_path / "hackathon_tourism_clean.jsonl"
    report_path = tmp_path / "hackathon_tourism_cleaning_report.csv"

    facts, report = preprocess_tourism_dataset.clean_workbook(
        input_path=tmp_path / "input.xlsx",
        output_path=fact_path,
        report_path=report_path,
    )

    fact_records = [json.loads(line) for line in fact_path.read_text(encoding="utf-8").splitlines()]
    opinion_records = [
        json.loads(line)
        for line in (tmp_path / "hackathon_reviews_clean.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert facts == fact_records
    assert all(record["record_type"] != "opinion" for record in fact_records)
    assert len(opinion_records) == 1
    assert opinion_records[0]["record_type"] == "opinion"
    assert report[1]["ulasan_terlalu_pendek_dilewati"] == 1
    assert fact_records[0]["lat"] == 2.3494
    assert fact_records[0]["lon"] == 99.0403
    assert fact_records[0]["rating"] == 4.3
    assert fact_records[0]["harga_min"] == 5000
    assert fact_records[0]["harga_max"] == 10000
    assert fact_records[0]["status"] == "beroperasi"
    assert fact_records[0]["kabupaten"] == "Toba"
