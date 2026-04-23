"""
Document parser for user-uploaded files.

Converts arbitrary documents (.txt, .md, .pdf) into lists of concept
dictionaries that are compatible with the seed data ingestion pipeline.

Two paths:
  1. Structured markdown (has a `# Concept Name` H1 header) → parsed as-is
     using parse_seed_file from loader.py.
  2. Everything else → text is extracted, chunked, and wrapped into plain
     concept nodes whose description holds the raw content.  The keyword-
     based retrieval in graph.py will then surface them as context.
"""

from __future__ import annotations

import re
from pathlib import Path

# Max characters kept in a single concept's description field.
# Keeping this reasonable avoids bloating the Neo4j node store and keeps
# LLM context windows manageable.
_MAX_DESC_CHARS = 1200

# Target chunk size for long documents (characters, not tokens).
_CHUNK_TARGET = 900


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_document(filepath: Path) -> list[dict]:
    """
    Parse an uploaded file into one or more concept dicts.

    Each dict has the shape expected by loader._upsert_concept:
      {
        "name": str,
        "description": str,
        "relationships": [{"type": str, "target": str}],
        "sources": [{"title": str, "url": str, "type": str}],
      }

    Raises ValueError for unsupported extensions.
    Raises ImportError if pypdf is not installed and a .pdf is supplied.
    """
    suffix = filepath.suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf(filepath)
    elif suffix in (".txt", ".md"):
        text = filepath.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Accepted formats: .txt, .md, .pdf"
        )

    # Structured seed-format markdown → delegate to the existing parser so
    # rich relationship and source metadata is preserved.
    if suffix == ".md" and _is_structured_seed(text):
        from knowledge.loader import parse_seed_file
        return [parse_seed_file(filepath)]

    # Plain text / unstructured markdown → extract, chunk, wrap.
    return _text_to_concepts(filepath.stem, text, filepath.name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pdf(filepath: Path) -> str:
    """Extract plain text from all pages of a PDF."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF support.  "
            "Add 'pypdf>=4.0.0' to your dependencies."
        ) from exc

    reader = PdfReader(str(filepath))
    pages = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            pages.append(extracted)
    return "\n\n".join(pages)


def _is_structured_seed(text: str) -> bool:
    """
    Return True if the markdown file looks like a hand-authored seed file,
    i.e. it begins with a top-level `# Concept Name` heading.
    """
    for line in text.lstrip().splitlines()[:10]:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return True
    return False


def _text_to_concepts(stem: str, text: str, original_filename: str) -> list[dict]:
    """Convert raw text into one or more concept dicts."""
    # Normalise whitespace but preserve paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []

    base_name = _human_title(stem)
    source_entry = {
        "title": original_filename,
        "url": "",
        "type": "uploaded_document",
    }

    # Short enough for a single node
    if len(text) <= _MAX_DESC_CHARS:
        return [
            {
                "name": base_name,
                "description": text,
                "relationships": [],
                "sources": [source_entry],
            }
        ]

    # Long document → split into chunks, link back to a root node
    chunks = _chunk_text(text, _CHUNK_TARGET)
    concepts: list[dict] = []

    # Root node holds a short summary (first chunk trimmed)
    root_desc = chunks[0][:_MAX_DESC_CHARS]
    concepts.append(
        {
            "name": base_name,
            "description": root_desc,
            "relationships": [],
            "sources": [source_entry],
        }
    )

    for i, chunk in enumerate(chunks[1:], start=2):
        part_name = f"{base_name} — Part {i}"
        concepts.append(
            {
                "name": part_name,
                "description": chunk[:_MAX_DESC_CHARS],
                "relationships": [
                    {"type": "PART_OF_DOCUMENT", "target": base_name}
                ],
                "sources": [source_entry],
            }
        )

    return concepts


def _chunk_text(text: str, target_size: int) -> list[str]:
    """
    Split *text* into chunks of approximately *target_size* characters,
    preferring to break at paragraph or sentence boundaries.
    """
    # Try paragraph-level splits first
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > target_size and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_len = 0

        # A single paragraph that is too long → split by sentences
        if len(para) > target_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if current_len + len(sent) > target_size and current_parts:
                    chunks.append(" ".join(current_parts))
                    current_parts = []
                    current_len = 0
                current_parts.append(sent)
                current_len += len(sent) + 1
        else:
            current_parts.append(para)
            current_len += len(para) + 2  # +2 for paragraph break

    if current_parts:
        chunks.append(" ".join(current_parts))

    return [c for c in chunks if c.strip()]


def _human_title(stem: str) -> str:
    """Convert a filename stem to a human-readable title."""
    # Strip common extensions that may still be in the stem
    stem = re.sub(r"\.(txt|md|pdf|docx)$", "", stem, flags=re.IGNORECASE)
    # Replace separators with spaces
    stem = stem.replace("_", " ").replace("-", " ")
    # Title-case
    return " ".join(word.capitalize() for word in stem.split())
