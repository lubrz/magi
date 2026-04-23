"""
TRIAD — FastAPI application entry point.

Provides:
  - WebSocket endpoint for real-time deliberation streaming
  - REST endpoints for health, graph stats, and user votes
  - CORS configuration for frontend
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents.axiom import AxiomAgent
from agents.forge import ForgeAgent
from agents.llm_providers import create_provider
from agents.prism import PrismAgent
from config import settings
from knowledge.graph import KnowledgeGraphManager
from knowledge.loader import load_seed_data
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
# Global state
# ---------------------------------------------------------------------------
graph_manager: Optional[KnowledgeGraphManager] = None
agents: dict[AgentName, object] = {}
# Store active deliberations for user-vote resolution
active_deliberations: dict[str, DeliberationResult] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global graph_manager, agents

    # --- Connect to Neo4j ---
    graph_manager = KnowledgeGraphManager(settings)
    try:
        await graph_manager.connect()
        await graph_manager.init_schema()

        # Load seed data if graph is empty
        stats = await graph_manager.get_stats()
        total = sum(d.get("concepts", 0) for d in stats.values() if isinstance(d, dict))
        if total == 0:
            logger.info("Graph is empty — loading seed data...")
            await load_seed_data(graph_manager._driver)
            logger.info("Seed data loaded.")
        else:
            logger.info(f"Graph already populated ({total} concepts).")

    except Exception as e:
        logger.warning(f"Neo4j connection failed (will retry on request): {e}")

    # --- Create agents ---
    for agent_name, AgentClass in [
        ("axiom", AxiomAgent),
        ("prism", PrismAgent),
        ("forge", ForgeAgent),
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

    # --- Shutdown ---
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
    graph_stats = {}

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

    agent_status = {}
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
    per_agent = {}
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
    """
    Submit a question and wait for the complete deliberation result.
    (Non-streaming — useful for CLI and API integrations.)
    """
    max_rounds = request.max_rounds or settings.max_rounds

    deliberation = Deliberation(
        agents=agents,
        max_rounds=max_rounds,
        consensus_threshold=settings.consensus_threshold,
        consensus_min_votes=settings.consensus_min_votes,
    )

    result = await deliberation.run(question=request.question)

    # Store for potential user vote
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

    # Find the selected agent's latest position
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

            # Create event callback that sends to WebSocket
            async def send_event(event: WSEvent):
                try:
                    await ws.send_json(event.model_dump(mode="json"))
                except Exception:
                    pass

            # We need a sync wrapper for the async callback
            event_queue: asyncio.Queue[WSEvent] = asyncio.Queue()

            def queue_event(event: WSEvent):
                event_queue.put_nowait(event)

            # Run deliberation and stream events
            deliberation = Deliberation(
                agents=agents,
                max_rounds=max_rounds,
                consensus_threshold=settings.consensus_threshold,
                consensus_min_votes=settings.consensus_min_votes,
            )

            # Run deliberation in background, stream events from queue
            async def run_deliberation():
                return await deliberation.run(
                    question=question,
                    on_event=queue_event,
                )

            delib_task = asyncio.create_task(run_deliberation())

            # Stream events while deliberation runs
            while not delib_task.done():
                try:
                    event = await asyncio.wait_for(
                        event_queue.get(), timeout=0.1
                    )
                    await ws.send_json(event.model_dump(mode="json"))
                except asyncio.TimeoutError:
                    continue

            # Drain remaining events
            while not event_queue.empty():
                event = event_queue.get_nowait()
                await ws.send_json(event.model_dump(mode="json"))

            # Send final result
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
