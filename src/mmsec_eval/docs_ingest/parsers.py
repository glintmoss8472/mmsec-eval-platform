from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def parse_pdf(path: str, max_pages: int = 5) -> str:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"pdf not found: {pdf_path}")
    try:
        from pypdf import PdfReader
    except (ImportError, OSError) as e:  # pragma: no cover
        raise RuntimeError(f"pypdf is required for PDF parsing: {e}") from e

    reader = PdfReader(str(pdf_path))
    texts: list[str] = []
    limit = max(1, int(max_pages))
    for page in reader.pages[:limit]:
        txt = (page.extract_text() or "").strip()
        if txt:
            texts.append(txt)
    out = "\n\n".join(texts).strip()
    if not out:
        raise RuntimeError(f"failed to extract text from PDF: {pdf_path}")
    return out


def _parse_doc_via_word_com(path: str) -> str:
    doc_path = Path(path)
    temp_txt = Path(tempfile.gettempdir()) / f"mmsec_{doc_path.stem}_word.txt"
    ps = rf"""
$ErrorActionPreference='Stop'
$word=$null
$doc=$null
try {{
  $word=New-Object -ComObject Word.Application
  $word.Visible=$false
  $word.DisplayAlerts=0
  $doc=$word.Documents.Open('{str(doc_path)}', $false, $true)
  [System.IO.File]::WriteAllText('{str(temp_txt)}', $doc.Content.Text, [System.Text.Encoding]::UTF8)
  $doc.Close()
  $word.Quit()
  Write-Output 'OK'
}} catch {{
  if($doc){{ $doc.Close() }}
  if($word){{ $word.Quit() }}
  Write-Output 'FAIL'
}}
"""
    p = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if p.returncode == 0 and temp_txt.exists():
        return temp_txt.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def parse_doc(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"doc not found: {p}")

    if p.suffix.lower() == ".docx":
        try:
            import zipfile
            from xml.etree import ElementTree as ET

            with zipfile.ZipFile(p) as zf:
                xml = zf.read("word/document.xml")
            root = ET.fromstring(xml)
            texts = [n.text for n in root.iter() if n.tag.endswith("}t") and n.text]
            out = "\n".join(texts).strip()
            if not out:
                raise RuntimeError(f"docx has no extractable text: {p}")
            return out
        except (OSError, KeyError, RuntimeError, zipfile.BadZipFile, ET.ParseError) as e:
            raise RuntimeError(f"failed to parse docx: {p}: {e}") from e

    text = _parse_doc_via_word_com(str(p))
    if text:
        return text
    raise RuntimeError(f"failed to parse .doc via Word COM: {p}")


def parse_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"text file not found: {p}")
    out = p.read_text(encoding="utf-8", errors="ignore")
    if not out.strip():
        raise RuntimeError(f"text file is empty: {p}")
    return out
