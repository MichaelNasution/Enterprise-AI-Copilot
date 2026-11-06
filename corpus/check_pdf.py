"""Cek isi dan kualitas teks setiap PDF di corpus/raw."""
import re
from pathlib import Path

from pypdf import PdfReader

RAW_DIR = Path(__file__).resolve().parent / "raw"

for pdf in sorted(RAW_DIR.glob("*.pdf")):
    reader = PdfReader(pdf)
    text = " ".join((page.extract_text() or "") for page in reader.pages)
    words = text.split()
    clean = [w for w in words if re.fullmatch(r"[A-Za-z]{2,}[.,;:]?", w)]
    ratio = len(clean) / max(len(words), 1)
    preview = re.sub(r"\s+", " ", text[:200])
    print(f"\n{pdf.name}")
    print(f"  {len(reader.pages)} halaman | {len(words)} kata | rasio bersih {ratio:.2f}")
    print(f"  awal: {preview}")