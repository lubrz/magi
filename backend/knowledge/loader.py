"""
Data ingestion pipeline — loads markdown seed data into the Neo4j graph.

Each markdown file follows a structured format:
  # Concept Name
  Description paragraph(s)...

  ## Relationships
  - RELATES_TO: Other Concept Name
  - DEPENDS_ON: Another Concept

  ## Sources
  - Title | URL | type
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

# Map directory names to Neo4j labels
DIR_TO_LABEL = {
    "axiom": "AxiomConcept",
    "prism": "PrismConcept",
    "forge": "ForgeConcept",
}

DIR_TO_DOMAIN = {
    "axiom": "science",
    "prism": "culture",
    "forge": "engineering",
}

# Allowlist of valid Neo4j relationship type identifiers
# (must be alphanumeric + underscore, no spaces)
_VALID_REL_TYPE_RE = re.compile(r'^[A-Z][A-Z0-9_]*$')


def _sanitise_rel_type(raw: str) -> str:
    """
    Convert a relationship type string to a valid Neo4j identifier.

    BUG FIX: The original code interpolated rel['type'] directly into Cypher
    f-strings without any validation, which would silently produce invalid
    queries if the markdown contained relationship types with spaces or
    special characters (e.g. "PART OF" instead of "PART_OF").
    """
    # Upper-case and replace spaces/hyphens with underscores
    sanitised = raw.strip().upper().replace(" ", "_").replace("-", "_")
    # Remove any remaining non-identifier characters
    sanitised = re.sub(r'[^A-Z0-9_]', '', sanitised)
    if not sanitised:
        return "RELATES_TO"
    # Must start with a letter
    if not sanitised[0].isalpha():
        sanitised = "REL_" + sanitised
    return sanitised


def parse_seed_file(filepath: Path) -> dict:
    """
    Parse a structured markdown file into a concept dictionary.

    Returns:
        {
            "name": str,
            "description": str,
            "relationships": [{"type": str, "target": str}, ...],
            "sources": [{"title": str, "url": str, "type": str}, ...],
        }
    """
    text = filepath.read_text(encoding="utf-8")
    lines = text.strip().split("\n")

    name = ""
    description_lines = []
    relationships = []
    sources = []
    current_section = "description"

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("# ") and not stripped.startswith("## "):
            name = stripped[2:].strip()
            continue

        if stripped.startswith("## Relationships"):
            current_section = "relationships"
            continue
        elif stripped.startswith("## Sources"):
            current_section = "sources"
            continue
        elif stripped.startswith("## "):
            current_section = "other"
            continue

        if current_section == "description" and stripped:
            description_lines.append(stripped)
        elif current_section == "relationships" and stripped.startswith("- "):
            match = re.match(r"-\s*(\w+):\s*(.+)", stripped)
            if match:
                relationships.append({
                    "type": _sanitise_rel_type(match.group(1)),
                    "target": match.group(2).strip(),
                })
        elif current_section == "sources" and stripped.startswith("- "):
            parts = stripped[2:].split("|")
            source = {"title": parts[0].strip()}
            source["url"] = parts[1].strip() if len(parts) > 1 else ""
            source["type"] = parts[2].strip() if len(parts) > 2 else "document"
            sources.append(source)

    return {
        "name": name,
        "description": " ".join(description_lines),
        "relationships": relationships,
        "sources": sources,
    }


async def load_seed_data(
    driver: AsyncDriver,
    seed_dir: Optional[Path] = None,
) -> dict[str, int]:
    """
    Load all seed data from the seed_data directory into Neo4j.

    Returns a summary of how many concepts were loaded per agent.
    """
    if seed_dir is None:
        seed_dir = Path(__file__).parent / "seed_data"

    if not seed_dir.exists():
        logger.warning(f"Seed data directory not found: {seed_dir}")
        return {}

    stats = {}

    for agent_dir in ["axiom", "prism", "forge"]:
        dir_path = seed_dir / agent_dir
        if not dir_path.exists():
            continue

        label = DIR_TO_LABEL[agent_dir]
        domain = DIR_TO_DOMAIN[agent_dir]
        count = 0

        md_files = sorted(dir_path.glob("*.md"))
        logger.info(f"Loading {len(md_files)} files for {agent_dir} ({label})")

        for filepath in md_files:
            try:
                concept = parse_seed_file(filepath)
                if not concept["name"]:
                    logger.warning(f"Skipping {filepath}: no concept name found")
                    continue
                await _upsert_concept(driver, concept, label, domain)
                count += 1
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")

        stats[agent_dir] = count
        logger.info(f"Loaded {count} concepts for {agent_dir}")

    return stats


async def load_uploaded_document(
    driver: AsyncDriver,
    filepath: Path,
    agent_name: str,
) -> list[str]:
    """
    Parse an uploaded document and ingest it into the agent's knowledge graph.

    Supports .txt, .md (both structured seed format and plain text), and .pdf.
    Returns a list of concept names that were created/updated.
    """
    from knowledge.document_parser import parse_document

    label = DIR_TO_LABEL.get(agent_name.lower())
    domain = DIR_TO_DOMAIN.get(agent_name.lower())
    if not label or not domain:
        raise ValueError(f"Unknown agent '{agent_name}'. Use: axiom, prism, forge")

    concepts = parse_document(filepath)
    if not concepts:
        raise ValueError(f"No content could be extracted from {filepath.name}")

    created: list[str] = []
    for concept in concepts:
        if concept.get("name"):
            await _upsert_concept(driver, concept, label, domain)
            created.append(concept["name"])

    return created


async def load_custom_data(
    driver: AsyncDriver,
    data_dir: Path,
    agent_name: str,
) -> int:
    """Load custom domain data for a specific agent."""
    label = DIR_TO_LABEL.get(agent_name.lower())
    domain = DIR_TO_DOMAIN.get(agent_name.lower())

    if not label or not domain:
        raise ValueError(f"Unknown agent: {agent_name}. Use: axiom, prism, forge")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    count = 0
    for filepath in sorted(data_dir.glob("*.md")):
        try:
            concept = parse_seed_file(filepath)
            if concept["name"]:
                await _upsert_concept(driver, concept, label, domain)
                count += 1
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")

    return count


async def _upsert_concept(
    driver: AsyncDriver,
    concept: dict,
    label: str,
    domain: str,
) -> None:
    """
    Insert or update a concept node with its relationships and sources.

    BUG FIX: Relationship types are now sanitised before being interpolated
    into the Cypher query, preventing silent failures on malformed input.
    """
    async with driver.session() as session:
        await session.run(
            f"""
            MERGE (c:Concept:{label} {{name: $name}})
            SET c.description = $description,
                c.domain = $domain
            WITH c
            MERGE (d:Domain {{name: $domain}})
            MERGE (c)-[:PART_OF]->(d)
            """,
            name=concept["name"],
            description=concept["description"],
            domain=domain,
        )

        for rel in concept.get("relationships", []):
            rel_type = _sanitise_rel_type(rel.get("type", "RELATES_TO"))
            target = rel.get("target", "").strip()
            if not target:
                continue
            # Cypher relationship types cannot be parameterised, so we
            # interpolate the already-sanitised identifier directly.
            await session.run(
                f"""
                MATCH (c:Concept:{label} {{name: $source_name}})
                MERGE (t:Concept {{name: $target_name}})
                ON CREATE SET t.domain = $domain, t.description = ''
                MERGE (c)-[r:{rel_type}]->(t)
                """,
                source_name=concept["name"],
                target_name=target,
                domain=domain,
            )

        for src in concept.get("sources", []):
            title = src.get("title", "").strip()
            if not title:
                continue
            await session.run(
                f"""
                MATCH (c:Concept:{label} {{name: $concept_name}})
                MERGE (s:Source {{title: $title}})
                SET s.url = $url, s.source_type = $type
                MERGE (c)-[:EVIDENCED_BY]->(s)
                """,
                concept_name=concept["name"],
                title=title,
                url=src.get("url", ""),
                type=src.get("type", "document"),
            )
