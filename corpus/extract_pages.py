"""Ambil rentang halaman dari PDF (nomor halaman penampil PDF, mulai dari 1)."""
import re
import sys

from pypdf import PdfReader, PdfWriter


def main() -> None:
    if len(sys.argv) != 5:
        print("Cara pakai: python extract_pages.py <sumber.pdf> <hal_awal> <hal_akhir> <tujuan.pdf>")
        sys.exit(1)

    source, start, end, target = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    reader = PdfReader(source)
    writer = PdfWriter()
    for index in range(start - 1, end):
        writer.add_page(reader.pages[index])
    with open(target, "wb") as handle:
        writer.write(handle)

    first_page = re.sub(r"\s+", " ", reader.pages[start - 1].extract_text() or "")
    print(f"[OK] {end - start + 1} halaman disimpan ke {target}")
    print(f"Awal halaman {start}: {first_page[:300]}")


if __name__ == "__main__":
    main()