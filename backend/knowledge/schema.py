"""
Neo4j schema definitions — Cypher statements to initialise the graph.
"""

from config import settings

# Constraints ensure uniqueness and speed up lookups
CONSTRAINTS = [
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT source_title IF NOT EXISTS FOR (s:Source) REQUIRE s.title IS UNIQUE",
    "CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
]


def _vector_index_statement() -> str:
    """
    Build the vector index DDL with the configured embedding dimensions.

    Dimension must match the embedding provider:
      - OpenAI text-embedding-3-small → 1536
      - Ollama nomic-embed-text       → 768
    """
    dims = settings.embedding_dimensions
    return (
        "CREATE VECTOR INDEX concept_embedding IF NOT EXISTS "
        "FOR (c:Concept) "
        "ON (c.embedding) "
        "OPTIONS {indexConfig: {"
        f"  `vector.dimensions`: {dims},"
        "  `vector.similarity_function`: 'cosine'"
        "}}"
    )


# Initial domain nodes
SEED_DOMAINS = """
MERGE (d1:Domain {name: 'science'})
MERGE (d2:Domain {name: 'culture'})
MERGE (d3:Domain {name: 'engineering'})
"""

# Agent-specific label indexes
AGENT_INDEXES = [
    "CREATE INDEX axiom_idx IF NOT EXISTS FOR (c:AxiomConcept) ON (c.name)",
    "CREATE INDEX prism_idx IF NOT EXISTS FOR (c:PrismConcept) ON (c.name)",
    "CREATE INDEX forge_idx IF NOT EXISTS FOR (c:ForgeConcept) ON (c.name)",
]


def get_all_schema_statements() -> list[str]:
    """Return all Cypher statements needed to initialise the schema."""
    stmts = []
    stmts.extend(CONSTRAINTS)
    stmts.append("DROP INDEX concept_embedding IF EXISTS")
    stmts.append(_vector_index_statement())
    stmts.append(SEED_DOMAINS)
    stmts.extend(AGENT_INDEXES)
    return stmts
