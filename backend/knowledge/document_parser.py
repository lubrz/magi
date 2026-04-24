"""
Document parser for user-uploaded files.

Converts arbitrary documents (.txt, .md, .pdf) into lists of concept
dictionaries that are compatible with the seed data ingestion pipeline.

Two paths:
  1. Structured markdown (has a `# Concept Name` H1 header) → parsed as-is
     using parse_seed_file from loader.py.
  2. Everything else → text is extracted, chunked, and passed to an LLM
     to extract POLE+O entities (Person, Organization, Location, Event, Object)
     as detailed in neo4j-labs/create-context-graph.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from agents.llm_providers import BaseLLMProvider

logger = logging.getLogger(__name__)

_MAX_DESC_CHARS = 1200
_CHUNK_TARGET = 3000 # Increased chunk size since LLM will summarize/extract

EXTRACTION_PROMPT = """\
Analyze the following text and extract key entities according to the POLE+O model:
- Person: specific individuals
- Organization: companies, groups, institutions
- Location: physical places, regions
- Event: specific occurrences, historical events
- Object: physical items, specific technologies, products
- Concept: abstract ideas, theories, principles

For each entity, extract:
- name: a clear, concise name
- type: one of [Person, Organization, Location, Event, Object, Concept]
- description: a detailed summary of what the text says about this entity
- relationships: connections to other entities mentioned in the text (use UPPERCASE_SNAKE_CASE for the relationship type)

Respond with ONLY valid JSON matching this schema:
{
  "entities": [
    {
      "name": "Entity Name",
      "type": "Person",
      "description": "...",
      "relationships": [
        {"type": "PART_OF", "target": "Other Entity Name"}
      ]
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def parse_document(filepath: Path, llm: BaseLLMProvider = None) -> list[dict]:
    """
    Parse an uploaded file into one or more concept dicts.
    Uses LLM for POLE+O extraction if provided.
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

    if suffix == ".md" and _is_structured_seed(text):
        from knowledge.loader import parse_seed_file
        return [parse_seed_file(filepath)]

    if not llm:
        logger.warning("No LLM provided to parse_document; falling back to generic chunks.")
        return _text_to_chunks(filepath.stem, text, filepath.name)

    return await _text_to_poleo_entities(filepath.stem, text, filepath.name, llm)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pdf(filepath: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF support.") from exc

    reader = PdfReader(str(filepath))
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n\n".join(pages)

def _is_structured_seed(text: str) -> bool:
    for line in text.lstrip().splitlines()[:10]:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return True
    return False

def _text_to_chunks(stem: str, text: str, original_filename: str) -> list[dict]:
    """Fallback parser if no LLM is available."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    base_name = _human_title(stem)
    source_entry = {"title": original_filename, "url": "", "type": "uploaded_document"}
    
    if len(text) <= _MAX_DESC_CHARS:
        return [{"name": base_name, "description": text, "type": "Concept", "relationships": [], "sources": [source_entry]}]

    chunks = _chunk_text(text, _CHUNK_TARGET)
    concepts = [{"name": base_name, "description": chunks[0][:_MAX_DESC_CHARS], "type": "Concept", "relationships": [], "sources": [source_entry]}]
    
    for i, chunk in enumerate(chunks[1:], start=2):
        concepts.append({
            "name": f"{base_name} — Part {i}",
            "description": chunk[:_MAX_DESC_CHARS],
            "type": "Concept",
            "relationships": [{"type": "PART_OF_DOCUMENT", "target": base_name}],
            "sources": [source_entry],
        })
    return concepts

async def _text_to_poleo_entities(stem: str, text: str, original_filename: str, llm: BaseLLMProvider) -> list[dict]:
    """Extract POLE+O entities using an LLM."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    source_entry = {"title": original_filename, "url": "", "type": "uploaded_document"}
    chunks = _chunk_text(text, _CHUNK_TARGET)
    all_entities = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Extracting POLE+O entities from chunk {i+1}/{len(chunks)}...")
        user_message = f"Text to analyze:\n\n{chunk}"
        try:
            result = await llm.generate_json(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=EXTRACTION_PROMPT,
                temperature=0.2,
            )
            entities = result.get("entities", [])
            for e in entities:
                e["sources"] = [source_entry]
                if "type" not in e:
                    e["type"] = "Concept"
                all_entities.append(e)
        except Exception as exc:
            logger.error(f"Failed to extract entities from chunk {i+1}: {exc}")
            # Fallback for this chunk
            all_entities.append({
                "name": f"{_human_title(stem)} - Chunk {i+1}",
                "type": "Concept",
                "description": chunk[:_MAX_DESC_CHARS],
                "relationships": [],
                "sources": [source_entry]
            })

    # Merge entities with same name
    merged = {}
    for e in all_entities:
        name = e.get("name", "").strip()
        if not name:
            continue
        if name in merged:
            merged[name]["description"] += "\n\n" + e.get("description", "")
            merged[name]["relationships"].extend(e.get("relationships", []))
        else:
            merged[name] = e
            
    return list(merged.values())

def _chunk_text(text: str, target_size: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks, current_parts = [], []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > target_size and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_len = 0

        if len(para) > target_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if current_len + len(sent) > target_size and current_parts:
                    chunks.append(" ".join(current_parts))
                    current_parts, current_len = [], 0
                current_parts.append(sent)
                current_len += len(sent) + 1
        else:
            current_parts.append(para)
            current_len += len(para) + 2

    if current_parts:
        chunks.append(" ".join(current_parts))
    return [c for c in chunks if c.strip()]

def _human_title(stem: str) -> str:
    stem = re.sub(r"\.(txt|md|pdf|docx)$", "", stem, flags=re.IGNORECASE)
    stem = stem.replace("_", " ").replace("-", " ")
    return " ".join(word.capitalize() for word in stem.split())
