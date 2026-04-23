.PHONY: setup up down stop logs build clean cli-install up-aura up-local up-aura-local

# --- Setup ---
setup:
	cp .env.example .env
	@echo "Created .env - please configure your API keys"

# --- Docker Control ---
up:
	docker compose up -d --build

up-aura:
	docker compose -f docker-compose.yml -f docker-compose.aura.yml up -d --build

up-local:
	docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build

up-aura-local:
	docker compose -f docker-compose.yml -f docker-compose.aura.yml -f docker-compose.local.yml up -d --build

up-all-local:
	docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build

stop:
	docker compose stop

down:
	docker compose down -v

logs:
	docker compose logs -f

build:
	docker compose build

# --- Backend Dev ---
dev-backend:
	cd backend && pip install -e . && uvicorn main:app --reload

# --- Frontend Dev ---
dev-frontend:
	cd frontend && npm install && npm run dev

# --- CLI ---
cli-install:
	cd cli && pip install -e .
	@echo "CLI installed. Run 'triad --help' to start."

# --- Cleanup ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf backend/.pytest_cache
	rm -rf frontend/dist
	rm -rf frontend/node_modules
