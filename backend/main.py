"""
TRIAD — FastAPI application entry point.

Provides:
  - WebSocket endpoint for real-time deliberation streaming
  - REST endpoints for health, graph stats, user votes, and document upload
  - CORS configuration for frontend
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents.axiom import AxiomAgent
from agents.forge import ForgeAgent
from agents.llm_providers import create_provider
from agents.prism import PrismAgent
from config import settings
from knowledge.graph import KnowledgeGraphManager
from knowledge.loader import load_seed_data, load_uploaded_document, ensure_seed_data_loaded
from models.schemas import (
    AgentName,
    AskRequest,
    DeliberationResult,
    GraphStatsResponse,
    HealthResponse,
    UserVoteRequest,
    WSEvent,
    WSEventType,
)
from orchestrator.deliberation import Deliberation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("triad")

# ---------------------------------------------------------------------------
# Supported upload extensions
# ---------------------------------------------------------------------------
_ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
_AGENT_NAMES = {"axiom", "prism", "forge"}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
graph_manager: Optional[KnowledgeGraphManager] = None
agents: dict[AgentName, object] = {}
active_deliberations: dict[str, DeliberationResult] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global graph_manager, agents

    graph_manager = KnowledgeGraphManager(settings)
    try:
        await graph_manager.connect()
        await graph_manager.init_schema()

        # Check for seed data control via env or argument
        seed_env = os.environ.get("SEED_DATA", str(getattr(settings, "seed_data", "false"))).lower()
        seed_arg = os.environ.get("SEED_ARG", "").lower()  # Optionally set by process launcher
        seed_requested = seed_env in ("1", "true", "yes") or seed_arg in ("1", "true", "yes")

        if seed_requested:
            logger.info("SEED_DATA requested — ensuring seed data is loaded...")
            await ensure_seed_data_loaded(
                graph_manager._driver,
                embedder=graph_manager.embedder,
            )
            logger.info("Seed data ensured.")
        else:
            stats = await graph_manager.get_stats()
            total = sum(
                d.get("concepts", 0) for d in stats.values() if isinstance(d, dict)
            )
            if total == 0:
                logger.info("Graph is empty — loading seed data...")
                await load_seed_data(
                    graph_manager._driver,
                    embedder=graph_manager.embedder,
                )
                logger.info("Seed data loaded.")
            else:
                logger.info(f"Graph already populated ({total} concepts).")

    except Exception as e:
        logger.warning(f"Neo4j connection failed (will retry on request): {e}")

    from agents.arbiter import ArbiterAgent
    for agent_name, AgentClass in [
        ("axiom", AxiomAgent),
        ("prism", PrismAgent),
        ("forge", ForgeAgent),
        ("arbiter", ArbiterAgent),
    ]:
        cfg = settings.get_agent_config(agent_name)
        provider = create_provider(
            provider=cfg.provider,
            model=cfg.model,
            api_key=cfg.api_key,
            settings=settings,
        )
        agent = AgentClass(llm=provider, graph_manager=graph_manager)
        agents[agent.name] = agent
        logger.info(
            f"Agent {agent_name.upper()} initialised: "
            f"{cfg.provider.value}/{cfg.model}"
        )

    yield

    if graph_manager:
        await graph_manager.close()
    logger.info("TRIAD shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TRIAD",
    description="Multi-Agent Consensus System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    neo4j_status = "not_configured"
    graph_stats: dict[str, int] = {}

    if graph_manager:
        neo4j_status = await graph_manager.health_check()
        try:
            raw_stats = await graph_manager.get_stats()
            graph_stats = {
                k: v.get("concepts", 0) if isinstance(v, dict) else v
                for k, v in raw_stats.items()
            }
        except Exception:
            pass

    agent_status: dict[str, str] = {}
    for name, agent in agents.items():
        agent_status[name.value] = f"{agent.llm.model} ({type(agent.llm).__name__})"

    return HealthResponse(
        status="ok",
        neo4j=neo4j_status,
        agents=agent_status,
        graph_stats=graph_stats,
    )


@app.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats():
    """Knowledge graph statistics."""
    if not graph_manager:
        return GraphStatsResponse()

    raw = await graph_manager.get_stats()
    per_agent: dict[str, dict[str, int]] = {}
    total_concepts = 0
    total_rels = 0

    for key, val in raw.items():
        if isinstance(val, dict):
            per_agent[key] = val
            total_concepts += val.get("concepts", 0)
            total_rels += val.get("relationships", 0)

    return GraphStatsResponse(
        total_concepts=total_concepts,
        total_relationships=total_rels,
        total_sources=raw.get("total_sources", 0),
        per_agent=per_agent,
    )


@app.post("/ask")
async def ask_sync(request: AskRequest):
    """Submit a question and receive the complete deliberation result (non-streaming)."""
    max_rounds = request.max_rounds or settings.max_rounds

    deliberation = Deliberation(
        agents=agents,
        arbiter=agents[AgentName.ARBITER],
        max_rounds=max_rounds,
        consensus_threshold=settings.consensus_threshold,
        consensus_min_votes=settings.consensus_min_votes,
    )

    result = await deliberation.run(question=request.question)
    active_deliberations[result.id] = result
    return result.model_dump(mode="json")


@app.post("/vote")
async def user_vote(request: UserVoteRequest):
    """User selects an agent's position during deadlock."""
    delib = active_deliberations.get(request.deliberation_id)
    if not delib:
        return JSONResponse(
            status_code=404,
            content={"error": "Deliberation not found"},
        )

    if delib.rounds:
        last_round = delib.rounds[-1]
        selected = next(
            (p for p in last_round.positions if p.agent == request.selected_agent),
            None,
        )
        if selected:
            delib.final_answer = selected.position
            delib.selected_by = "user"
            delib.status = "user_decided"
            return delib.model_dump(mode="json")

    return JSONResponse(
        status_code=400,
        content={"error": "Selected agent position not found"},
    )


@app.post("/upload/{agent}")
async def upload_document(agent: str, file: UploadFile = File(...)):
    """
    Upload a document (.txt, .md, or .pdf) to an agent's knowledge base.

    The document is parsed, chunked if necessary, and its content is
    immediately ingested into the agent's Neo4j subgraph so the agent
    can reference it in future deliberations.

    Path parameters:
      agent — one of: axiom, prism, forge

    Returns:
      {
        "status": "ok",
        "agent": "axiom",
        "filename": "my_paper.pdf",
        "concepts_created": 3,
        "concept_names": ["My Paper", "My Paper — Part 2", ...],
        "verified": true
      }
    """
    agent_lower = agent.lower()
    if agent_lower not in _AGENT_NAMES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown agent '{agent}'. Valid: axiom, prism, forge"},
        )

    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No filename provided"})

    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Unsupported file type '{ext}'. "
                    f"Accepted: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
                )
            },
        )

    if not graph_manager:
        return JSONResponse(
            status_code=503,
            content={"error": "Knowledge graph is not available"},
        )

    # Sanitise the original filename (strip path traversal, special chars)
    safe_name = re.sub(r"[^\w\-_\.]", "_", Path(file.filename).name)

    # Save to the agent's seed_data directory so it persists across restarts
    seed_dir = Path(__file__).parent / "knowledge" / "seed_data" / agent_lower
    seed_dir.mkdir(parents=True, exist_ok=True)
    dest_path = seed_dir / safe_name

    # Write file to disk
    try:
        contents = await file.read()
        dest_path.write_bytes(contents)
    except Exception as e:
        logger.error(f"File write failed: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to save file: {e}"})

    # Parse and ingest into Neo4j (with embeddings)
    try:
        created_names = await load_uploaded_document(
            graph_manager._driver,
            dest_path,
            agent_lower,
            embedder=graph_manager.embedder,
        )
        logger.info(
            f"Uploaded '{safe_name}' to {agent_lower}: "
            f"{len(created_names)} concept(s) created"
        )

        # Verification: attempt to retrieve the first created concept
        verified = False
        if created_names:
            try:
                from knowledge.loader import DIR_TO_LABEL
                label = DIR_TO_LABEL[agent_lower]
                async with graph_manager._driver.session() as session:
                    result = await session.run(
                        "MATCH (c:Concept) WHERE c.name = $name AND $label IN labels(c) "
                        "RETURN c.name AS name, c.embedding IS NOT NULL AS has_embedding",
                        name=created_names[0],
                        label=label,
                    )
                    record = await result.single()
                    verified = record is not None
                    if record:
                        has_emb = record.get("has_embedding", False)
                        logger.info(
                            f"Upload verified: '{created_names[0]}' exists in graph "
                            f"(embedding: {has_emb})"
                        )
            except Exception as ve:
                logger.warning(f"Upload verification query failed: {ve}")

        return {
            "status": "ok",
            "agent": agent_lower,
            "filename": safe_name,
            "concepts_created": len(created_names),
            "concept_names": created_names,
            "verified": verified,
        }
    except Exception as e:
        logger.error(f"Ingestion failed for '{safe_name}': {e}")
        # Remove the file so a retry doesn't leave partial state
        dest_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Ingestion failed: {e}"},
        )


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for real-time deliberation.

    Client sends: {"question": "...", "max_rounds": 3}
    Server streams: WSEvent objects as JSON
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await ws.receive_text()
            payload = json.loads(data)

            question = payload.get("question", "").strip()
            if not question:
                await ws.send_json(
                    WSEvent(
                        type=WSEventType.ERROR,
                        data={"error": "No question provided"},
                    ).model_dump(mode="json")
                )
                continue

            max_rounds = payload.get("max_rounds", settings.max_rounds)

            event_queue: asyncio.Queue[WSEvent] = asyncio.Queue()

            def queue_event(event: WSEvent):
                event_queue.put_nowait(event)

            deliberation = Deliberation(
                agents=agents,
                arbiter=agents[AgentName.ARBITER],
                max_rounds=max_rounds,
                consensus_threshold=settings.consensus_threshold,
                consensus_min_votes=settings.consensus_min_votes,
            )

            async def run_deliberation():
                return await deliberation.run(
                    question=question,
                    on_event=queue_event,
                )

            delib_task = asyncio.create_task(run_deliberation())

            while not delib_task.done():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    await ws.send_json(event.model_dump(mode="json"))
                except asyncio.TimeoutError:
                    continue

            while not event_queue.empty():
                event = event_queue.get_nowait()
                await ws.send_json(event.model_dump(mode="json"))

            result = await delib_task
            active_deliberations[result.id] = result

            await ws.send_json(
                WSEvent(
                    type=WSEventType.FINAL_ANSWER,
                    data=result.model_dump(mode="json"),
                ).model_dump(mode="json")
            )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json(
                WSEvent(
                    type=WSEventType.ERROR,
                    data={"error": str(e)},
                ).model_dump(mode="json")
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
