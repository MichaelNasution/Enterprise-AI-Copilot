import csv
import difflib
import re
from bisect import bisect_right
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
NEGATION_PATTERN = re.compile(r"\b(?:tidak|tanpa|dilarang|wajib|bukan|jangan)\b", re.IGNORECASE)
PROTECTED_EDGE_PATTERN = re.compile(r"^\s*(?:Pasal|BAB|Bagian|Paragraf)\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"^\s*(?:19|20)\d{2}\s*$")


def extract_pages(pdf_path: str | Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def _normalize_header(line: str) -> str:
    return re.sub(r"\d", "#", " ".join(line.lower().split()))


def _repeated_edge_line_indices(
    pages: list[str], edge: int, ratio: float
) -> list[set[int]]:
    removed = [set() for _ in pages]
    if len(pages) < 4 or edge <= 0 or not pages:
        return removed

    occurrences: dict[str, list[tuple[int, int]]] = {}
    for page_index, page in enumerate(pages):
        lines = page.splitlines()
        nonempty_indices = [i for i, line in enumerate(lines) if line.strip()]
        edge_indices = set(nonempty_indices[:edge] + nonempty_indices[-edge:])
        for line_index in edge_indices:
            line = lines[line_index]
            if PROTECTED_EDGE_PATTERN.match(line) or YEAR_PATTERN.match(line):
                continue
            normalized = _normalize_header(line)
            if normalized:
                occurrences.setdefault(normalized, []).append((page_index, line_index))

    values = sorted(occurrences, key=lambda value: (len(value), value))
    lengths = [len(value) for value in values]
    parents = list(range(len(values)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(values):
        max_length = int(len(first) * 1.5)
        stop = bisect_right(lengths, max_length, lo=first_index + 1)
        for second_index in range(first_index + 1, stop):
            second = values[second_index]
            matcher = difflib.SequenceMatcher(None, first, second, autojunk=False)
            if matcher.real_quick_ratio() < 0.8 or matcher.quick_ratio() < 0.8:
                continue
            if matcher.ratio() >= 0.8:
                union(first_index, second_index)

    cluster_pages: dict[int, set[int]] = {}
    for index, value in enumerate(values):
        root = find(index)
        cluster_pages.setdefault(root, set()).update(
            page_index for page_index, _ in occurrences[value]
        )

    minimum_pages = len(pages) * ratio
    repeated_roots = {
        root for root, page_indices in cluster_pages.items()
        if len(page_indices) >= minimum_pages
    }
    for index, value in enumerate(values):
        if find(index) in repeated_roots:
            for page_index, line_index in occurrences[value]:
                removed[page_index].add(line_index)
    return removed


def remove_repeated_headers_footers(
    pages: list[str], edge: int = 3, ratio: float = 0.5
) -> list[str]:
    removed_indices = _repeated_edge_line_indices(pages, edge, ratio)
    cleaned_pages: list[str] = []
    for page, removed in zip(pages, removed_indices):
        cleaned_pages.append(
            "\n".join(line for index, line in enumerate(page.splitlines()) if index not in removed)
        )
    return cleaned_pages


def is_toc_page(text: str) -> bool:
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return False
    leader_lines = sum(
        bool(re.search(r"\.{2,}\s*\d{1,4}\s*$", line)) for line in nonempty_lines
    )
    return leader_lines / len(nonempty_lines) >= 0.3


def clean_page(text: str) -> str:
    cleaned_lines: list[str] = []
    page_number = re.compile(r"^(?:-?\s*\d{1,3}\s*-?|i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)$")
    copy_code = re.compile(r"\bSK\s*No\s*\d+\s*A\b", re.IGNORECASE)
    watermark = re.compile(r"https?://\S*bps\.go\.id\S*", re.IGNORECASE)
    spaced_letters = re.compile(r"^\s*(?:[A-Za-z]\s+){2,}[A-Za-z]\s*$")

    for original_line in text.splitlines():
        line = original_line
        if page_number.fullmatch(line.strip()):
            continue
        if copy_code.search(line):
            continue
        line = watermark.sub("", line).strip()
        if not line:
            continue
        line = re.sub(r"\b(Pasal)(?=\d)", r"\1 ", line, flags=re.IGNORECASE)
        if re.search(r"(?:\.\s*){3,}$", line.strip()) and len(line.strip()) < 60:
            continue
        if spaced_letters.fullmatch(line):
            continue
        if re.search(r"\.{10,}", line):
            continue
        if line.strip() and not re.search(r"\w", line, re.UNICODE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


_NEXT_LINE_START_PATTERN = re.compile(
    r"^(?:[A-Za-z]\.|1\.|\(1\)|1\)|-|•|BAB\b|Pasal\b|Bagian\b|Paragraf\b|"
    r"Tabel\b|Table\b|Gambar\b|Lampiran\b)",
    re.IGNORECASE,
)
_SHORT_SECTION_TITLE_PATTERN = re.compile(r"^(?:Pasal|BAB)\b", re.IGNORECASE)
_LEGAL_LABEL_PATTERN = re.compile(
    r"^(?:Menimbang|Mengingat|Menetapkan|Memutuskan|TENTANG)\b",
    re.IGNORECASE,
)


def _is_section_title(line: str) -> bool:
    return bool(_SHORT_SECTION_TITLE_PATTERN.match(line.strip())) and len(line.strip()) < 40


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return (
        (stripped.isupper() and any(character.isalpha() for character in stripped) and len(stripped) < 80)
        or (bool(_LEGAL_LABEL_PATTERN.match(stripped)) and len(stripped) < 40)
    )


def join_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    joined: list[str] = []
    for line in lines:
        if not line:
            if joined and joined[-1] != "":
                joined.append("")
            continue
        if not joined or joined[-1] == "":
            joined.append(line)
            continue

        previous = joined[-1]
        should_keep_break = (
            previous.endswith((".", ":", ";", "?", "!"))
            or bool(_NEXT_LINE_START_PATTERN.match(line))
            or _is_section_title(previous)
            or _is_heading(previous)
            or _is_heading(line)
        )
        if should_keep_break:
            joined.append(line)
            continue

        if previous.endswith("-"):
            continuation = re.match(r"^([a-z]+)\b", line)
            if not continuation:
                joined.append(line)
                continue
            previous_word = re.search(r"([^\W\d_]+)-$", previous, re.UNICODE)
            if previous_word and previous_word.group(1).lower() == continuation.group(1):
                joined[-1] = f"{previous}{line}"
            else:
                joined[-1] = f"{previous[:-1]}{line}"
            continue

        joined[-1] = f"{previous} {line}"
    return "\n".join(joined)


def normalize_whitespace(text: str) -> str:
    normalized: list[str] = []
    previous_blank = False
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            if normalized and not previous_blank:
                normalized.append("")
            previous_blank = True
            continue
        normalized.append(line)
        previous_blank = False
    return "\n".join(normalized).strip()


def _count_negations(text: str) -> int:
    return len(NEGATION_PATTERN.findall(text))


def main() -> None:
    manifest_path = ROOT / "corpus" / "manifest.csv"
    output_dir = ROOT / "data" / "corpus"
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(encoding="utf-8-sig", newline="") as manifest_file:
        documents = list(csv.DictReader(manifest_file))

    report_rows: list[dict[str, object]] = []
    for document in documents:
        doc_id = document["doc_id"]
        filename = document["nama_file"]
        title = document["judul"]
        raw_pages = extract_pages(ROOT / "corpus" / "raw" / filename)
        toc_flags = [is_toc_page(page) for page in raw_pages]
        header_cleaned_pages = remove_repeated_headers_footers(raw_pages)
        cleaned_pages: list[str] = []
        for page_index, page in enumerate(header_cleaned_pages):
            if toc_flags[page_index]:
                continue
            cleaned_pages.append(
                normalize_whitespace(join_lines(clean_page(page)))
            )

        raw_text = "\n".join(raw_pages)
        cleaned_text = "\n\n".join(page for page in cleaned_pages if page)
        toc_negations = sum(
            _count_negations(page)
            for page, is_toc in zip(raw_pages, toc_flags)
            if is_toc
        )
        raw_negations = _count_negations(raw_text)
        clean_negations = _count_negations(cleaned_text)
        if clean_negations + toc_negations != raw_negations:
            raise RuntimeError(
                f"Negasi tidak terkonservasi untuk {doc_id}: "
                f"mentah={raw_negations}, bersih={clean_negations}, "
                f"daftar_isi={toc_negations}"
            )

        output_path = output_dir / f"{Path(filename).stem}.txt"
        output_path.write_text(f"# {doc_id} | {title}\n\n{cleaned_text}\n", encoding="utf-8")

        raw_characters = sum(len(page) for page in raw_pages)
        clean_characters = sum(len(page) for page in cleaned_pages)
        reduction = (
            (raw_characters - clean_characters) / raw_characters * 100
            if raw_characters
            else 0.0
        )
        report_rows.append(
            {
                "doc_id": doc_id,
                "nama_file": filename,
                "halaman": len(raw_pages),
                "halaman_daftar_isi_dibuang": sum(toc_flags),
                "karakter_mentah": raw_characters,
                "karakter_bersih": clean_characters,
                "persen_berkurang": f"{reduction:.2f}",
                "jumlah_negasi_mentah": raw_negations,
                "jumlah_negasi_bersih": clean_negations,
                "negasi_di_daftar_isi": toc_negations,
            }
        )

    report_path = output_dir / "cleaning_report.csv"
    fieldnames = [
        "doc_id",
        "nama_file",
        "halaman",
        "halaman_daftar_isi_dibuang",
        "karakter_mentah",
        "karakter_bersih",
        "persen_berkurang",
        "jumlah_negasi_mentah",
        "jumlah_negasi_bersih",
        "negasi_di_daftar_isi",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    for row in report_rows:
        reduction = float(row["persen_berkurang"])
        if not 1 <= reduction <= 40:
            print(
                f"PERINGATAN {row['doc_id']}: persen_berkurang "
                f"{reduction:.2f}% di luar rentang 1%-40%"
            )


if __name__ == "__main__":
    main()
