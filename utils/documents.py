"""Text extraction for uploaded documents (PDF, DOCX, plain text / code).

Uploads are untrusted, so extraction is defensive:
  * DOCX is parsed straight from the ZIP with a cap on the *uncompressed* size
    of ``word/document.xml`` (zip-bomb protection).
  * Everything is done in memory — no temp files to leak on error.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

MAX_DOCX_XML_BYTES = 20 * 1024 * 1024
TEXT_SUFFIXES = {
    "txt", "md", "csv", "json", "py", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h",
    "go", "rs", "rb", "php", "html", "css", "sql", "yaml", "yml", "toml", "sh", "log", "",
}
_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


class UnsupportedDocument(ValueError):
    """Raised for file types or contents we refuse to process."""


def extract_text(filename: str, data: bytes) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == "pdf":
        return _pdf_text(data)
    if suffix == "docx":
        return _docx_text(data)
    if suffix in TEXT_SUFFIXES:
        if b"\x00" in data[:4096]:
            raise UnsupportedDocument("File looks binary, not text.")
        return data.decode("utf-8", errors="replace").strip()
    raise UnsupportedDocument(f"Unsupported file type: .{suffix}")


def _pdf_text(data: bytes) -> str:
    if PdfReader is None:
        raise UnsupportedDocument("PDF support unavailable. Install: pip install pypdf")
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _docx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            info = zf.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise UnsupportedDocument("DOCX content is too large to process.")
            xml = zf.read(info)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise UnsupportedDocument("Not a valid .docx file.") from exc

    root = ET.fromstring(xml)
    paragraphs = []
    for para in root.findall(".//w:p", _W_NS):
        text = "".join(t.text or "" for t in para.findall(".//w:t", _W_NS)).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)
