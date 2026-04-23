# TRIAD — Multi-Agent Consensus System

TRIAD is a multi-agent deliberation architecture inspired by a famous sci-fi computing system from the 90s. 

Three AI agents with distinct simulated cognitive perspectives (Analytical, Creative, Pragmatic) collaborate to produce grounded, high-quality answers.

Each agent is grounded in a dedicated **Neo4j context graph** related to their domain (e.g., space exploration physics for Axiom, cultural history for Prism, and engineering trade-offs for Forge).

## Architecture

- **AXIOM (Analytical)**: Reasons through logic, evidence, and systematic analysis.
- **PRISM (Creative)**: Reasons through patterns, connections, and lateral thinking.
- **FORGE (Pragmatic)**: Reasons through feasibility, risk management, and practical constraints.

The agents undergo multiple rounds of deliberation, critiquing each other's positions until a consensus (Unanimous or Majority) is reached. If they cannot agree after the maximum rounds, the system enters a **Deadlock**, allowing the user to select the most appropriate position.

## Tech Stack

- **Backend**: Python 3.12, FastAPI, WebSocket, Pydantic, official `neo4j-graphrag`.
- **Frontend**: Vite, Vanilla JS/CSS (Retro Sci-Fi Terminal UI).
- **Graph**: Neo4j 5 (Local Docker or AuraDB Cloud).
- **CLI**: Typer + Rich.
- **LLMs**: OpenAI, Anthropic, Google Gemini, or local Ollama.

## Setup & Usage

### 1. Prerequisites
- Docker & Docker Compose
- Make (optional)
- API Keys (OpenAI, Anthropic, or Gemini) or local Ollama

### 2. Configuration
```bash
make setup
# Edit .env and add your API keys
```

### 3. Launch

**Option A: Local Neo4j + Hosted LLMs (Default - Requires API keys)**
```bash
make up
```

**Option B: Neo4j AuraDB + Hosted LLMs**
```bash
# Set NEO4J_MODE=aura and connection details in .env
make up-aura
```

**Option C: Fully Local (Ollama + local Neo4j)**
```bash
make up-local
```

### 4. Access
- **Web UI**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8080](http://localhost:8080)
- **Neo4j Browser**: [http://localhost:7474](http://localhost:7474) (Local mode only)

### 5. CLI Usage
```bash
make cli-install
triad ask "Is a crewed Mars mission feasible by 2035?"
```

## Seed Data
The system comes with a "Space Exploration" test domain including ~15 concepts across the three agents' domains. This data is automatically loaded on first launch.
