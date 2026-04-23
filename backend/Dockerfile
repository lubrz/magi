FROM python:3.12-slim

WORKDIR /app

# Install build tools and hatchling
RUN pip install --no-cache-dir hatchling

# Copy everything
COPY . .

# Install dependencies and the project
RUN pip install --no-cache-dir .

# Expose port
EXPOSE 8080

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
