import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
CHUNKS_DIR = ROOT / "data" / "chunks"
MANIFEST_PATH = ROOT / "corpus" / "manifest.csv"
DATASET_PATH = CORPUS_DIR / "hackathon_tourism_clean.jsonl"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_TOKENS = 128
MIN_CHUNK_CHARS = 100
CHUNK_SIZES = [300, 400, 500, 600, 800]
OVERLAPS = [0, 50, 100]

_SENTENCE_BOUNDARY = re.compile(r"[.?!;]\s+")
_PARAGRAPH_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def _split_preserving_separator(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    while True:
        position = text.find(separator, start)
        if position < 0:
            break
        end = position + len(separator)
        parts.append(text[start:end])
        start = end
    if start < len(text):
        parts.append(text[start:])
    return parts or [text]


def _recursive_segments(text: str, chunk_size: int, separator_index: int = 0) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    for index in range(separator_index, len(_PARAGRAPH_SEPARATORS)):
        pieces = _split_preserving_separator(text, _PARAGRAPH_SEPARATORS[index])
        if len(pieces) > 1:
            result: list[str] = []
            for piece in pieces:
                if len(piece) > chunk_size:
                    result.extend(_recursive_segments(piece, chunk_size, index + 1))
                else:
                    result.append(piece)
            return result
    return [text]


def _merge_short_chunks(chunks: list[str], chunk_size: int) -> list[str]:
    merged: list[str] = []
    pending = ""
    for chunk in chunks:
        if pending:
            pending = f"{pending} {chunk}".strip()
            if len(pending) >= MIN_CHUNK_CHARS:
                merged.append(pending)
                pending = ""
        elif len(chunk) < MIN_CHUNK_CHARS:
            pending = chunk
        else:
            merged.append(chunk)
    if pending:
        if merged and len(merged[-1]) + len(pending) + 1 <= chunk_size:
            merged[-1] = f"{merged[-1]} {pending}".strip()
        else:
            merged.append(pending)
    return merged


def _overlap_prefix(previous: str, overlap: int) -> str:
    if overlap <= 0 or len(previous) <= overlap:
        return ""
    desired_start = len(previous) - overlap
    sentence_starts = [
        match.end()
        for match in _SENTENCE_BOUNDARY.finditer(previous)
        if desired_start <= match.end() < len(previous)
    ]
    if sentence_starts:
        start = sentence_starts[-1]
    else:
        next_space = re.search(r"\s", previous[desired_start:])
        start = (
            desired_start + next_space.end()
            if next_space
            else desired_start
        )
    return previous[start:].strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size harus lebih besar dari 0")
    if overlap < 0:
        raise ValueError("overlap tidak boleh negatif")

    text = text.strip()
    if not text:
        return []

    segments = _recursive_segments(text, chunk_size)
    chunks: list[str] = []
    current = ""
    for segment in segments:
        candidate = f"{current}{segment}"
        if current and len(candidate) > chunk_size:
            chunks.append(current.strip())
            current = segment
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())

    chunks = _merge_short_chunks(chunks, chunk_size)
    if overlap == 0 or len(chunks) < 2:
        return [chunk for chunk in chunks if chunk]

    with_overlap = [chunks[0]]
    for index in range(1, len(chunks)):
        prefix = _overlap_prefix(chunks[index - 1], overlap)
        with_overlap.append(f"{prefix} {chunks[index]}".strip() if prefix else chunks[index])
    return with_overlap


def _token_count(tokenizer: Any, text: str, add_special_tokens: bool = True) -> int:
    return len(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _actual_overlap_percent(chunks: list[str], overlap: int) -> float:
    total_characters = sum(len(chunk) for chunk in chunks)
    if total_characters == 0:
        return 0.0
    duplicated_characters = sum(
        len(_overlap_prefix(chunks[index - 1], overlap))
        for index in range(1, len(chunks))
    )
    return duplicated_characters / total_characters * 100


def _pdf_texts() -> list[tuple[dict[str, str], str]]:
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as source:
        manifest = list(csv.DictReader(source))
    texts: list[tuple[dict[str, str], str]] = []
    for document in manifest:
        path = CORPUS_DIR / f"{Path(document['nama_file']).stem}.txt"
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].startswith(f"# {document['doc_id']} |"):
            lines = lines[1:]
            if lines and not lines[0].strip():
                lines = lines[1:]
        texts.append((document, "\n".join(lines).strip()))
    return texts


def _experiment(
    pdf_texts: list[tuple[dict[str, str], str]], tokenizer: Any
) -> tuple[list[dict[str, object]], dict[tuple[int, int], list[str]]]:
    rows: list[dict[str, object]] = []
    generated: dict[tuple[int, int], list[str]] = {}
    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            chunks: list[str] = []
            for _, text in pdf_texts:
                chunks.extend(split_text(text, chunk_size, overlap))
            token_counts = [_token_count(tokenizer, chunk) for chunk in chunks]
            character_counts = [len(chunk) for chunk in chunks]
            generated[(chunk_size, overlap)] = chunks
            rows.append(
                {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "jumlah_chunk": len(chunks),
                    "rata2_karakter": (
                        f"{statistics.mean(character_counts):.2f}" if chunks else "0.00"
                    ),
                    "rata2_token": f"{statistics.mean(token_counts):.2f}" if chunks else "0.00",
                    "p95_token": _p95(token_counts),
                    "maks_token": max(token_counts, default=0),
                    "persen_melebihi_batas": (
                        f"{sum(count > MAX_TOKENS for count in token_counts) / len(token_counts) * 100:.2f}"
                        if token_counts
                        else "0.00"
                    ),
                    "persen_overlap_aktual": f"{_actual_overlap_percent(chunks, overlap):.2f}",
                }
            )
    return rows, generated


def _choose_parameters(
    rows: list[dict[str, object]],
) -> tuple[int, int] | None:
    eligible = [
        row
        for row in rows
        if int(row["overlap"]) == (100 if int(row["chunk_size"]) >= 400 else 50)
        and float(str(row["persen_melebihi_batas"])) <= 1
    ]
    if not eligible:
        return None
    selected = max(eligible, key=lambda row: int(row["chunk_size"]))
    chunk_size = int(selected["chunk_size"])
    overlap = 100 if chunk_size >= 400 else 50
    return chunk_size, overlap


def _write_experiment(rows: list[dict[str, object]]) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "chunk_size",
        "overlap",
        "jumlah_chunk",
        "rata2_karakter",
        "rata2_token",
        "p95_token",
        "maks_token",
        "persen_melebihi_batas",
        "persen_overlap_aktual",
    ]
    with (CHUNKS_DIR / "chunking_experiment.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flat_metadata(values: dict[str, object]) -> dict[str, str | int | float | bool]:
    flat: dict[str, str | int | float | bool] = {}
    for key, value in values.items():
        if value is None or value == "":
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            raise TypeError(f"Metadata {key!r} harus berupa nilai skalar, bukan {type(value).__name__}")
    return flat


def _document_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}-c{chunk_index:03d}"


def _document_metadata(
    document: dict[str, str],
    chunk_index: int,
    text: str,
    tokenizer: Any,
) -> dict[str, str | int | float | bool]:
    try:
        year: int | None = int(document["tahun"])
    except (KeyError, ValueError):
        year = None
    return _flat_metadata(
        {
            "source_type": "dokumen",
            "doc_id": document.get("doc_id"),
            "judul": document.get("judul"),
            "jenis": document.get("jenis"),
            "penerbit": document.get("penerbit"),
            "tahun": year,
            "status": document.get("status"),
            "bahasa": document.get("bahasa"),
            "kategori": document.get("kategori"),
            "url_sumber": document.get("url_sumber"),
            "chunk_index": chunk_index,
            "char_count": len(text),
            "token_count": _token_count(tokenizer, text),
        }
    )


def _dataset_metadata(
    record: dict[str, Any],
    chunk_index: int,
    text: str,
    tokenizer: Any,
) -> dict[str, str | int | float | bool]:
    return _flat_metadata(
        {
            "source_type": "dataset",
            "record_type": record.get("record_type"),
            "title": record.get("title"),
            "source_sheet": record.get("source_sheet"),
            "lat": record.get("lat"),
            "lon": record.get("lon"),
            "rating": record.get("rating"),
            "harga_min": record.get("harga_min"),
            "harga_max": record.get("harga_max"),
            "status": record.get("status"),
            "kabupaten": record.get("kabupaten"),
            "chunk_index": chunk_index,
            "char_count": len(text),
            "token_count": _token_count(tokenizer, text),
        }
    )


def _build_final_chunks(
    pdf_texts: list[tuple[dict[str, str], str]],
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
) -> tuple[list[dict[str, object]], dict[str, list[int]]]:
    output: list[dict[str, object]] = []
    token_counts_by_source: dict[str, list[int]] = {}
    for document, text in pdf_texts:
        chunks = split_text(text, chunk_size, overlap)
        document_counts: list[int] = []
        for index, chunk in enumerate(chunks, 1):
            token_count = _token_count(tokenizer, chunk)
            document_counts.append(token_count)
            output.append(
                {
                    "id": _document_chunk_id(document["doc_id"], index),
                    "text": chunk,
                    "metadata": _document_metadata(document, index, chunk, tokenizer),
                }
            )
        token_counts_by_source[document["doc_id"]] = document_counts

    with DATASET_PATH.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record["record_id"])
            text = str(record["text"]).strip()
            parts = (
                split_text(text, chunk_size, overlap)
                if _token_count(tokenizer, text) > MAX_TOKENS
                else [text]
            )
            for index, part in enumerate(parts, 1):
                token_count = _token_count(tokenizer, part)
                output_id = f"R-{record_id}" if len(parts) == 1 else f"R-{record_id}-p{index}"
                output.append(
                    {
                        "id": output_id,
                        "text": part,
                        "metadata": _dataset_metadata(record, index, part, tokenizer),
                    }
                )
                token_counts_by_source.setdefault("dataset", []).append(token_count)
    return output, token_counts_by_source


def _write_final_chunks(chunks: list[dict[str, object]]) -> None:
    with (CHUNKS_DIR / "chunks.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for chunk in chunks:
            destination.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _write_stats(token_counts_by_source: dict[str, list[int]]) -> None:
    with (CHUNKS_DIR / "chunk_stats.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        fields = ["sumber", "jumlah_chunk", "rata2_token", "maks_token", "jumlah_melebihi_batas"]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for source, counts in token_counts_by_source.items():
            writer.writerow(
                {
                    "sumber": source,
                    "jumlah_chunk": len(counts),
                    "rata2_token": f"{statistics.mean(counts):.2f}" if counts else "0.00",
                    "maks_token": max(counts, default=0),
                    "jumlah_melebihi_batas": sum(count > MAX_TOKENS for count in counts),
                }
            )


def _write_histogram(chunks: list[dict[str, object]], tokenizer: Any) -> None:
    import matplotlib.pyplot as plt

    token_counts = [
        int(chunk["metadata"]["token_count"])  # type: ignore[index]
        for chunk in chunks
    ]
    figure, axis = plt.subplots()
    axis.hist(token_counts, bins=30)
    axis.axvline(MAX_TOKENS, color="red", linestyle="--", label=f"MAX_TOKENS={MAX_TOKENS}")
    axis.set_xlabel("Jumlah token per chunk")
    axis.set_ylabel("Frekuensi")
    axis.set_title("Distribusi token chunk final")
    axis.legend()
    figure.tight_layout()
    figure.savefig(CHUNKS_DIR / "token_distribution.png")
    plt.close(figure)


def main() -> None:
    pdf_texts = _pdf_texts()
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    corpus_characters = sum(len(text) for _, text in pdf_texts)
    corpus_tokens = sum(
        _token_count(tokenizer, text, add_special_tokens=False)
        for _, text in pdf_texts
    )
    characters_per_token = corpus_characters / corpus_tokens if corpus_tokens else 0
    print(f"Rasio karakter/token korpus PDF (tanpa token khusus): {characters_per_token:.3f}")

    experiment_rows, _ = _experiment(pdf_texts, tokenizer)
    _write_experiment(experiment_rows)
    parameters = _choose_parameters(experiment_rows)
    if parameters is None:
        raise RuntimeError(
            "Tidak ada kombinasi chunk_size/overlap dengan <=1% chunk dokumen "
            "yang melebihi MAX_TOKENS; parameter final tidak dipaksakan."
        )
    chunk_size, overlap = parameters
    matching = next(
        row
        for row in experiment_rows
        if row["chunk_size"] == chunk_size and row["overlap"] == overlap
    )
    print(
        f"Parameter final: chunk_size={chunk_size}, overlap={overlap}; "
        f"chunk_size terbesar dengan <=1% chunk dokumen melebihi "
        f"{MAX_TOKENS} token; overlap dipilih sesuai aturan "
        f"({'100' if chunk_size >= 400 else '50'}). "
        f"Persen melebihi batas={matching['persen_melebihi_batas']}%."
    )

    final_chunks, counts_by_source = _build_final_chunks(
        pdf_texts, tokenizer, chunk_size, overlap
    )
    document_counts = [
        count
        for source, counts in counts_by_source.items()
        if source != "dataset"
        for count in counts
    ]
    document_excess = (
        sum(count > MAX_TOKENS for count in document_counts)
        / len(document_counts)
        * 100
        if document_counts
        else 0
    )
    if document_excess > 1:
        raise RuntimeError(
            f"Chunk final dokumen melebihi batas pada {document_excess:.2f}% (>1%)."
        )
    ids = [str(chunk["id"]) for chunk in final_chunks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("ID chunk tidak unik.")
    _write_final_chunks(final_chunks)
    _write_stats(counts_by_source)
    _write_histogram(final_chunks, tokenizer)
    print(f"Chunk final ditulis: {len(final_chunks)}")
    print(f"Output: {CHUNKS_DIR}")


if __name__ == "__main__":
    main()
