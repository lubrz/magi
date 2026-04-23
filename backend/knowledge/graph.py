"""
Neo4j Knowledge Graph manager.

Provides retrieval operations for each agent's labeled subgraph,
combining vector search with Cypher traversal (hybrid GraphRAG).
"""

from __future__ import annotations

import logging
from typing import Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from config import Settings, settings
from knowledge.schema import get_all_schema_statements
from models.schemas import GraphContext, GraphSource

logger = logging.getLogger(__name__)

# Map agent labels to their domain
LABEL_TO_DOMAIN = {
    "AxiomConcept": "science",
    "PrismConcept": "culture",
    "ForgeConcept": "engineering",
}


class KnowledgeGraphManager:
    """
    Manages connections to Neo4j and provides retrieval for TRIAD agents.

    Works with both local Docker Neo4j and remote AuraDB — the driver
    auto-detects TLS requirements from the URI scheme.
    """

    def __init__(self, cfg: Optional[Settings] = None):
        self.cfg = cfg or settings
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """Establish connection to Neo4j with improved diagnostics."""
        try:
            logger.info(
                f"Connecting to Neo4j ({self.cfg.neo4j_mode.value}): {self.cfg.neo4j_uri} as {self.cfg.neo4j_user}"
            )
            self._driver = AsyncGraphDatabase.driver(
                self.cfg.neo4j_uri,
                auth=(self.cfg.neo4j_user, self.cfg.neo4j_password),
            )
            # Verify connectivity
            async with self._driver.session() as session:
                result = await session.run("RETURN 1 AS ok")
                record = await result.single()
                if record and record["ok"] == 1:
                    logger.info("Neo4j connection verified.")
                else:
                    logger.error("Neo4j connection failed: Unexpected result.")
        except Exception as e:
            logger.error(f"Neo4j connection error: {e}")
            logger.error(f"Check URI, credentials, and network access to Neo4j.")
            raise

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def init_schema(self) -> None:
        """Create schema constraints, indexes, and seed domain nodes."""
        if not self._driver:
            await self.connect()

        async with self._driver.session() as session:
            for stmt in get_all_schema_statements():
                try:
                    await session.run(stmt)
                except Exception as e:
                    # Some statements may fail if already exists (depending on version)
                    logger.debug(f"Schema statement note: {e}")

        logger.info("Neo4j schema initialised.")

    async def retrieve(
        self,
        question: str,
        label: str,
        top_k: int = 5,
    ) -> GraphContext:
        """
        Retrieve relevant context from an agent's labeled subgraph.

        Uses Cypher full-text + graph traversal. Matches any relationship type
        so DEPENDS_ON, PART_OF_DOCUMENT etc. are all traversed, not just RELATES_TO.
        """
        if not self._driver:
            await self.connect()

        # BUG FIX: The original query used OPTIONAL MATCH (c)-[r:RELATES_TO]-
        # which silently missed DEPENDS_ON, PART_OF_DOCUMENT and any other
        # relationship types present in the seed data. Using [r] matches all.
        #
        # Also moved the label filter into a WHERE clause instead of inline
        # label concatenation so the query is easier to read and safer.
        query = (
            "MATCH (c:Concept)"
            " WHERE $label IN labels(c)"
            "   AND ("
            "     toLower(c.description) CONTAINS toLower($keyword)"
            "     OR toLower(c.name) CONTAINS toLower($keyword)"
            "   )"
            " WITH c ORDER BY c.name LIMIT $top_k"
            " OPTIONAL MATCH (c)-[r]-(related:Concept)"
            " OPTIONAL MATCH (c)-[:EVIDENCED_BY]->(src:Source)"
            " RETURN"
            "   c.name AS concept_name,"
            "   c.description AS concept_description,"
            "   collect(DISTINCT {"
            "     name: related.name,"
            "     rel_type: type(r),"
            "     description: related.description"
            "   }) AS related_concepts,"
            "   collect(DISTINCT {"
            "     title: src.title,"
            "     url: src.url,"
            "     type: src.source_type"
            "   }) AS sources"
        )

        keywords = _extract_keywords(question)

        concepts: list[str] = []
        relationships: list[str] = []
        sources: list[GraphSource] = []
        raw_parts: list[str] = []

        async with self._driver.session() as session:
            for keyword in keywords[:3]:
                result = await session.run(
                    query,
                    label=label,
                    keyword=keyword,
                    top_k=top_k,
                )
                records = await result.data()

                for record in records:
                    name = record.get("concept_name", "")
                    desc = record.get("concept_description", "")
                    if name:
                        concepts.append(name)
                        raw_parts.append(f"**{name}**: {desc}")

                    for rel in record.get("related_concepts", []):
                        rel_name = rel.get("name")
                        if rel_name:
                            rel_str = (
                                f"{name} --[{rel.get('rel_type', 'RELATES_TO')}]--> {rel_name}"
                            )
                            relationships.append(rel_str)
                            if rel.get("description"):
                                raw_parts.append(
                                    f"  Related: **{rel_name}**: {rel['description']}"
                                )

                    for src in record.get("sources", []):
                        if src.get("title"):
                            sources.append(
                                GraphSource(
                                    title=src["title"],
                                    url=src.get("url"),
                                    source_type=src.get("type", "document"),
                                )
                            )

        # Deduplicate while preserving order
        concepts = list(dict.fromkeys(concepts))
        relationships = list(dict.fromkeys(relationships))
        seen_titles: set[str] = set()
        unique_sources: list[GraphSource] = []
        for s in sources:
            if s.title not in seen_titles:
                seen_titles.add(s.title)
                unique_sources.append(s)

        return GraphContext(
            concepts=concepts,
            relationships=relationships,
            sources=unique_sources,
            raw_text="\n".join(raw_parts) if raw_parts else "",
        )

    async def get_stats(self) -> dict:
        """Return graph statistics per agent label."""
        if not self._driver:
            await self.connect()

        stats = {}
        async with self._driver.session() as session:
            for label, domain in LABEL_TO_DOMAIN.items():
                result = await session.run(
                    "MATCH (c:Concept) WHERE $label IN labels(c) RETURN count(c) AS count",
                    label=label,
                )
                record = await result.single()
                concept_count = record["count"] if record else 0

                result2 = await session.run(
                    "MATCH (c:Concept)-[r]-() WHERE $label IN labels(c) RETURN count(r) AS count",
                    label=label,
                )
                record2 = await result2.single()
                rel_count = record2["count"] if record2 else 0

                stats[domain] = {
                    "concepts": concept_count,
                    "relationships": rel_count,
                }

            result3 = await session.run("MATCH (s:Source) RETURN count(s) AS count")
            record3 = await result3.single()
            stats["total_sources"] = record3["count"] if record3 else 0

        return stats

    async def health_check(self) -> str:
        """Check Neo4j connectivity."""
        try:
            if not self._driver:
                await self.connect()
            async with self._driver.session() as session:
                result = await session.run("RETURN 1")
                await result.single()
            return "connected"
        except Exception as e:
            return f"error: {str(e)}"


def _extract_keywords(question: str) -> list[str]:
    """Extract significant keywords from a question for graph search."""
    stop_words = {
        "a", "an", "the", "is", "it", "of", "to", "in", "for", "on", "with",
        "by", "at", "or", "and", "be", "we", "do", "if", "as", "no", "not",
        "are", "was", "were", "been", "has", "have", "had", "will", "would",
        "could", "should", "may", "might", "can", "this", "that", "these",
        "those", "what", "which", "who", "whom", "how", "why", "when", "where",
        "there", "their", "they", "them", "its", "our", "your", "his", "her",
        "my", "me", "i", "you", "he", "she", "than", "more", "most", "very",
        "just", "about", "also", "into", "from", "up", "out", "all", "some",
    }
    words = (
        question.lower()
        .replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .split()
    )
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords
