import csv
import hashlib
import json
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "dataset_hackathon_tourism.xlsx"
OUTPUT_PATH = ROOT / "data" / "corpus" / "hackathon_tourism_clean.jsonl"
REVIEW_OUTPUT_PATH = ROOT / "data" / "private" / "hackathon_reviews_clean.jsonl"
REPORT_PATH = ROOT / "data" / "corpus" / "hackathon_tourism_cleaning_report.csv"

NAMESPACE = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

SHEET_CONFIG = {
    "wisata-metadata": ("attraction", 1, 1),
    "hotel-metadata": ("accommodation", 1, 1),
    "resto-metadata": ("restaurant", 1, 1),
    "transportasi": ("transport", 1, 1),
    "kuliner": ("local_food", 1, 1),
    "Attractions Info": ("attraction", 2, 1),
    "waktu operasional destinasi": ("destination_operations", 1, 1),
}
REVIEW_SHEETS = {"resto-hotel-v2", "wisata-v2"}
EXCLUDED_SHEETS = {
    "hotel-resto-v1": "snapshot campuran/versi lama; tumpang tindih dengan metadata dan ulasan",
    "tempat-wisata-v1": "snapshot campuran/versi lama; tumpang tindih dengan metadata dan ulasan",
    "Info Seputar Danau Toba (TOP 3)": "tabel ringkasan multi-header dengan ulasan berulang; struktur sumber ambigu",
    "prompt": "contoh kueri/prompt, bukan fakta korpus",
    "Artikel Danau Toba": "salinan artikel dari situs pihak ketiga (hak cipta); satu tautan tidak sesuai judul",
}
EXCLUDED_DATA_START_ROW = {
    "hotel-resto-v1": 2,
    "tempat-wisata-v1": 2,
    "Info Seputar Danau Toba (TOP 3)": 5,
    "prompt": 4,
    "Artikel Danau Toba": 2,
}

IDENTIFIER_KEYS = {
    "no",
    "place_id",
    "transport_id",
    "kuliner_id",
    "reviewer_id",
    "id",
}
NON_TEXT_KEYS = IDENTIFIER_KEYS | {
    "lat_long",
    "latitude",
    "longitude",
    "link",
    "url",
    "source_url",
}
REVIEWER_KEYS = {
    "reviewer_id",
    "name",
    "reviewer_name",
    "reviewer_type",
}

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?62|0)[ .()/-]?\d(?:[ .()/-]?\d){7,13}(?!\w)"
)
WHITESPACE_PATTERN = re.compile(r"\s+")


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if not letters:
        raise ValueError(f"Referensi sel tidak valid: {cell_reference}")
    index = 0
    for letter in letters.group():
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.findall(".//main:t", NAMESPACE))
        for item in root.findall("main:si", NAMESPACE)
    ]


def _cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//main:t", NAMESPACE))
    value = cell.find("main:v", NAMESPACE)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text


def read_workbook(path: str | Path) -> dict[str, list[tuple[int, list[str]]]]:
    sheets: dict[str, list[tuple[int, list[str]]]] = {}
    with zipfile.ZipFile(path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        relationship_targets = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships
        }
        shared_strings = _shared_strings(archive)
        for sheet in workbook.findall("main:sheets/main:sheet", NAMESPACE):
            name = sheet.attrib["name"]
            relationship_id = sheet.attrib[f"{{{NAMESPACE['office_rel']}}}id"]
            target = relationship_targets[relationship_id]
            sheet_path = (
                target.lstrip("/")
                if target.startswith("/")
                else posixpath.normpath(posixpath.join("xl", target))
            )
            root = ElementTree.fromstring(archive.read(sheet_path))
            sheet_rows: list[tuple[int, list[str]]] = []
            for row in root.findall(".//main:sheetData/main:row", NAMESPACE):
                values = [""] * max(
                    (
                        _column_index(cell.attrib["r"]) + 1
                        for cell in row.findall("main:c", NAMESPACE)
                    ),
                    default=0,
                )
                for cell in row.findall("main:c", NAMESPACE):
                    values[_column_index(cell.attrib["r"])] = _cell_value(
                        cell, shared_strings
                    )
                sheet_rows.append((int(row.attrib["r"]), values))
            sheets[name] = sheet_rows
    return sheets


def _normalize_header(value: str, index: int) -> str:
    header = re.sub(r"\W+", "_", value.strip().casefold(), flags=re.UNICODE).strip("_")
    return header or f"column_{index + 1}"


def _headers(header_rows: list[list[str]]) -> tuple[list[str], list[str]]:
    width = max((len(row) for row in header_rows), default=0)
    display_headers: list[str] = []
    keys: list[str] = []
    for index in range(width):
        labels: list[str] = []
        for row in header_rows:
            label = row[index].strip() if index < len(row) else ""
            if label and label.casefold() not in {item.casefold() for item in labels}:
                labels.append(label)
        display_headers.append(" - ".join(labels) or f"Kolom {index + 1}")
        keys.append(_normalize_header(labels[-1] if labels else "", index))
    return display_headers, keys


def _clean_value(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def _record_id(source_sheet: str, record_type: str, title: str, text: str) -> str:
    material = "\0".join((source_sheet, record_type, title, text)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:20]


def _parse_number(value: str) -> float | None:
    try:
        number = float(value.strip().replace(",", "."))
    except (AttributeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _parse_rating(value: str | None) -> float | None:
    if not value:
        return None
    rating = _parse_number(value)
    if rating is None or not 0 <= rating <= 5:
        return None
    return rating


def _parse_price(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.casefold() == "gratis":
        return 0, 0
    excel_integer = re.fullmatch(r"\s*(?:rp\.?|idr)?\s*(\d{4,})\.0+\s*", normalized, re.IGNORECASE)
    if excel_integer:
        amount = int(excel_integer.group(1))
        return amount, amount
    match = re.fullmatch(
        r"\s*(?:rp\.?|idr)?\s*([\d.,]+)"
        r"(?:\s*(?:-|–|—|sampai|hingga|to)\s*(?:rp\.?|idr)?\s*([\d.,]+))?\s*",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return None

    def to_rupiah(number: str) -> int | None:
        digits = number.replace(".", "").replace(",", "")
        return int(digits) if digits else None

    first = to_rupiah(match.group(1))
    second = to_rupiah(match.group(2)) if match.group(2) else first
    if first is None or second is None:
        return None
    return min(first, second), max(first, second)


def _parse_coordinates(attributes: dict[str, str]) -> tuple[float | None, float | None]:
    latitude = attributes.get("latitude", "")
    longitude = attributes.get("longitude", "")
    combined = attributes.get("lat_long", "")
    if combined:
        coordinates = re.fullmatch(
            r"\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*",
            combined,
        )
        if not coordinates:
            return None, None
        latitude, longitude = coordinates.groups()
    if not latitude or not longitude:
        return None, None
    lat = _parse_number(latitude)
    lon = _parse_number(longitude)
    if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return None, None
    return lat, lon


def _parse_status(attributes: dict[str, str]) -> str | None:
    status = attributes.get("status", "").strip().lower()
    return status or None


def _parse_kabupaten(attributes: dict[str, str]) -> str | None:
    address = attributes.get("address", attributes.get("alamat", ""))
    for name in (
        "Tapanuli Utara",
        "Humbang Hasundutan",
        "Simalungun",
        "Samosir",
        "Toba",
    ):
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", address, re.IGNORECASE):
            return name
    return None


def _first_attribute(attributes: dict[str, str], keys: tuple[str, ...]) -> str | None:
    return next((attributes[key] for key in keys if attributes.get(key)), None)


def _fact_metadata(attributes: dict[str, str]) -> dict[str, object | None]:
    latitude, longitude = _parse_coordinates(attributes)
    rating = _parse_rating(
        _first_attribute(attributes, ("rating", "place_rating"))
    )
    price = _parse_price(
        _first_attribute(
            attributes,
            ("entry_fee", "ticket_price", "ticket_prices", "price_per_head", "harga", "tarif"),
        )
    )
    return {
        "lat": latitude,
        "lon": longitude,
        "rating": rating,
        "harga_min": price[0] if price else None,
        "harga_max": price[1] if price else None,
        "status": _parse_status(attributes),
        "kabupaten": _parse_kabupaten(attributes),
    }


def _fact_record(
    source_sheet: str,
    record_type: str,
    display_headers: list[str],
    keys: list[str],
    values: list[str],
) -> dict[str, object] | None:
    attributes = {
        key: _clean_value(values[index])
        for index, key in enumerate(keys)
        if index < len(values) and _clean_value(values[index])
    }
    if not attributes:
        return None

    title_keys = (
        "place_name",
        "place",
        "nama_atraksi",
        "nama_artikel",
        "judul_artikel",
        "transport_name",
        "kuliner_name",
        "objek_destinasi_wisata",
    )
    title = next(
        (str(attributes[key]) for key in title_keys if attributes.get(key)),
        "",
    )
    parts = [
        f"{display_headers[index]}: {attributes[key]}"
        for index, key in enumerate(keys)
        if key in attributes and key not in NON_TEXT_KEYS
    ]
    text = " | ".join(parts)
    if not text:
        return None
    return {
        "record_id": _record_id(source_sheet, record_type, title, text),
        "record_type": record_type,
        "title": title,
        "text": text,
        "source_file": INPUT_PATH.name,
        "source_sheet": source_sheet,
        **_fact_metadata(attributes),
        "attributes": attributes,
    }


def _redact_review_text(value: str) -> tuple[str, int]:
    text = _clean_value(value)
    redactions = 0
    text, email_count = EMAIL_PATTERN.subn("[email dihapus]", text)
    text, phone_count = PHONE_PATTERN.subn("[nomor telepon dihapus]", text)
    return text, email_count + phone_count


def _review_record(
    source_sheet: str,
    keys: list[str],
    values: list[str],
) -> tuple[dict[str, object] | None, int, bool]:
    row = {
        key: _clean_value(values[index]) if index < len(values) else ""
        for index, key in enumerate(keys)
    }
    place = next(
        (row[key] for key in ("place_name", "place", "nama_tempat") if row.get(key)),
        "",
    )
    review = next(
        (row[key] for key in ("review_text", "review", "ulasan") if row.get(key)),
        "",
    )
    if not review:
        return None, 0, False
    review, redactions = _redact_review_text(review)
    if not review:
        return None, redactions, False
    if len(review.split()) < 5:
        return None, redactions, True

    rating = next(
        (row[key] for key in ("reviewer_rating", "rating", "review_rating") if row.get(key)),
        "",
    )
    published = row.get("published_at", "")
    text_parts = []
    if place:
        text_parts.append(f"Tempat: {place}")
    if rating:
        text_parts.append(f"Rating pengulas: {rating}")
    text_parts.append(f"Ulasan pengunjung (opini): {review}")
    text = " | ".join(text_parts)
    title = place or "Ulasan wisata"
    attributes = {
        key: value
        for key, value in {
            "place_name": place,
            "reviewer_rating": rating,
            "published_at": published,
            "collect_date": row.get("collect_date", ""),
        }.items()
        if value
    }
    return (
        {
            "record_id": _record_id(source_sheet, "opinion", title, text),
            "record_type": "opinion",
            "title": title,
            "text": text,
            "source_file": INPUT_PATH.name,
            "source_sheet": source_sheet,
            "attributes": attributes,
        },
        redactions,
        False,
    )


def _row_key(record: dict[str, object]) -> str:
    return WHITESPACE_PATTERN.sub(" ", str(record["text"]).casefold()).strip()


def clean_workbook(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    report_path: str | Path = REPORT_PATH,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sheets = read_workbook(input_path)
    records: list[dict[str, object]] = []
    review_records: list[dict[str, object]] = []
    report: list[dict[str, object]] = []
    seen_facts: set[tuple[str, str]] = set()
    seen_reviews: set[str] = set()

    for sheet_name, rows in sheets.items():
        if sheet_name in SHEET_CONFIG:
            record_type, header_count, header_start = SHEET_CONFIG[sheet_name]
            header_rows = [
                next((values for row_number, values in rows if row_number == number), [])
                for number in range(header_start, header_start + header_count)
            ]
            display_headers, keys = _headers(header_rows)
            data_start = header_start + header_count
            input_rows = written = blank = duplicate = 0
            filled_lat_lon = filled_rating = filled_price = filled_status = filled_kabupaten = 0
            for row_number, values in rows:
                if row_number < data_start:
                    continue
                if not any(_clean_value(value) for value in values):
                    continue
                input_rows += 1
                record = _fact_record(
                    sheet_name, record_type, display_headers, keys, values
                )
                if record is None:
                    blank += 1
                    continue
                unique_key = (sheet_name, _row_key(record))
                if unique_key in seen_facts:
                    duplicate += 1
                    continue
                seen_facts.add(unique_key)
                records.append(record)
                written += 1
                filled_lat_lon += int(record["lat"] is not None and record["lon"] is not None)
                filled_rating += int(record["rating"] is not None)
                filled_price += int(
                    record["harga_min"] is not None and record["harga_max"] is not None
                )
                filled_status += int(record["status"] is not None)
                filled_kabupaten += int(record["kabupaten"] is not None)
            report.append(
                {
                    "sheet": sheet_name,
                    "status": "diproses",
                    "alasan": "",
                    "baris_data_terisi": input_rows,
                    "record_ditulis": written,
                    "baris_kosong_dilewati": blank,
                    "duplikat_dilewati": duplicate,
                    "ulasan_tanpa_teks_dilewati": 0,
                    "ulasan_terlalu_pendek_dilewati": 0,
                    "redaksi_pii_review": 0,
                    "terisi_lat_lon": filled_lat_lon,
                    "terisi_rating": filled_rating,
                    "terisi_harga": filled_price,
                    "terisi_status": filled_status,
                    "terisi_kabupaten": filled_kabupaten,
                }
            )
        elif sheet_name in REVIEW_SHEETS:
            header_row = next((values for row_number, values in rows if row_number == 1), [])
            _, keys = _headers([header_row])
            input_rows = written = blank = duplicate = redactions = too_short = 0
            for row_number, values in rows:
                if row_number <= 1:
                    continue
                if not any(_clean_value(value) for value in values):
                    continue
                input_rows += 1
                record, redaction_count, was_too_short = _review_record(
                    sheet_name, keys, values
                )
                redactions += redaction_count
                if was_too_short:
                    too_short += 1
                    continue
                if record is None:
                    blank += 1
                    continue
                unique_key = _row_key(record)
                if unique_key in seen_reviews:
                    duplicate += 1
                    continue
                seen_reviews.add(unique_key)
                review_records.append(record)
                written += 1
            report.append(
                {
                    "sheet": sheet_name,
                    "status": "diproses_anonim",
                    "alasan": "nama/ID/tipe reviewer tidak disimpan; record ditandai sebagai opini",
                    "baris_data_terisi": input_rows,
                    "record_ditulis": written,
                    "baris_kosong_dilewati": 0,
                    "duplikat_dilewati": duplicate,
                    "ulasan_tanpa_teks_dilewati": blank,
                    "ulasan_terlalu_pendek_dilewati": too_short,
                    "redaksi_pii_review": redactions,
                    "terisi_lat_lon": 0,
                    "terisi_rating": 0,
                    "terisi_harga": 0,
                    "terisi_status": 0,
                    "terisi_kabupaten": 0,
                }
            )
        elif sheet_name in EXCLUDED_SHEETS:
            report.append(
                {
                    "sheet": sheet_name,
                    "status": "dikecualikan",
                    "alasan": EXCLUDED_SHEETS[sheet_name],
                    "baris_data_terisi": sum(
                        row_number >= EXCLUDED_DATA_START_ROW[sheet_name]
                        and any(_clean_value(value) for value in values)
                        for row_number, values in rows
                    ),
                    "record_ditulis": 0,
                    "baris_kosong_dilewati": 0,
                    "duplikat_dilewati": 0,
                    "ulasan_tanpa_teks_dilewati": 0,
                    "ulasan_terlalu_pendek_dilewati": 0,
                    "redaksi_pii_review": 0,
                    "terisi_lat_lon": 0,
                    "terisi_rating": 0,
                    "terisi_harga": 0,
                    "terisi_status": 0,
                    "terisi_kabupaten": 0,
                }
            )
        else:
            report.append(
                {
                    "sheet": sheet_name,
                    "status": "dikecualikan",
                    "alasan": "sheet tidak termasuk daftar sumber terkurasi",
                    "baris_data_terisi": sum(bool(any(_clean_value(v) for v in values)) for _, values in rows),
                    "record_ditulis": 0,
                    "baris_kosong_dilewati": 0,
                    "duplikat_dilewati": 0,
                    "ulasan_tanpa_teks_dilewati": 0,
                    "ulasan_terlalu_pendek_dilewati": 0,
                    "redaksi_pii_review": 0,
                    "terisi_lat_lon": 0,
                    "terisi_rating": 0,
                    "terisi_harga": 0,
                    "terisi_status": 0,
                    "terisi_kabupaten": 0,
                }
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    review_output = REVIEW_OUTPUT_PATH
    if output_path != OUTPUT_PATH:
        review_output = Path(output_path).parent / "hackathon_reviews_clean.jsonl"
    review_output.parent.mkdir(parents=True, exist_ok=True)
    with review_output.open("w", encoding="utf-8", newline="\n") as destination:
        for record in review_records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_fields = [
        "sheet",
        "status",
        "alasan",
        "baris_data_terisi",
        "record_ditulis",
        "baris_kosong_dilewati",
        "duplikat_dilewati",
        "ulasan_tanpa_teks_dilewati",
        "ulasan_terlalu_pendek_dilewati",
        "redaksi_pii_review",
        "terisi_lat_lon",
        "terisi_rating",
        "terisi_harga",
        "terisi_status",
        "terisi_kabupaten",
    ]
    with report_file.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(report)
    return records, report


def main() -> None:
    records, report = clean_workbook()
    print(f"Record korpus ditulis: {len(records)}")
    print(f"Record ulasan privat ditulis: {sum(int(row['record_ditulis']) for row in report if row['status'] == 'diproses_anonim')}")
    print(f"Baris laporan sheet: {len(report)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Output ulasan privat: {REVIEW_OUTPUT_PATH}")
    print(f"Laporan: {REPORT_PATH}")


if __name__ == "__main__":
    main()
