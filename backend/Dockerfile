FROM python:3.12-slim

WORKDIR /app

# Install hatchling so it's available as the build backend
RUN pip install --no-cache-dir hatchling

# --- Dependency layer (cached separately from source changes) ---
# Copy only the manifest first.  Because hatchling now has an explicit
# [tool.hatch.build.targets.wheel] section, `pip install --no-deps .`
# is not needed here — we just install the declared dependencies directly.
# This layer is only rebuilt when pyproject.toml changes.
COPY pyproject.toml .

# Install every dependency listed in [project.dependencies].
# `--no-build-isolation` lets pip reuse the hatchling already installed above.
# We extract the dep list with Python's stdlib tomllib (available in 3.11+)
# so we don't need a separate requirements.txt to maintain.
RUN python3 -c "\
import tomllib; \
f=open('pyproject.toml','rb'); \
deps=tomllib.load(f)['project']['dependencies']; \
f.close(); \
print('\n'.join(deps))" > /tmp/deps.txt \
 && pip install --no-cache-dir -r /tmp/deps.txt \
 && rm /tmp/deps.txt

# --- Source layer ---
# Copy the rest of the source.  Changes here don't invalidate the dep layer.
COPY . .

# Install the project itself (registers the packages into site-packages
# so Python can import agents, knowledge, models, orchestrator, etc.).
RUN pip install --no-cache-dir --no-build-isolation --no-deps .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]