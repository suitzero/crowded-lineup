"""PDF page extraction and scanned image OCR utilities."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    engine: str


def extract_pdf_pages(pdf_path: Path) -> list[ExtractedPage]:
    """Extract per-page text deterministically using `pypdf`."""
    reader = PdfReader(str(pdf_path))
    pages: list[ExtractedPage] = []
    for idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        pages.append(ExtractedPage(page_number=idx, text=txt, engine="pypdf"))
    return pages


def extract_image_ocr(
    image_path: Path,
    sidecar_fallback: str | None = None,
) -> list[ExtractedPage]:
    """Run Tesseract OCR if installed, or explicit test sidecar when binary is absent."""
    tesseract_bin = shutil.which("tesseract")
    if tesseract_bin:
        proc = subprocess.run(
            [tesseract_bin, str(image_path), "stdout"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return [ExtractedPage(page_number=1, text=proc.stdout.strip(), engine="tesseract")]

    if sidecar_fallback:
        return [
            ExtractedPage(
                page_number=1,
                text=sidecar_fallback.strip(),
                engine="sidecar_ocr_fallback",
            )
        ]

    return [ExtractedPage(page_number=1, text="", engine="ocr_unavailable")]
