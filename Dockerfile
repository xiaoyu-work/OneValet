FROM python:3.12-slim

# System deps for asyncpg and git (needed for momex git dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[all]"

# Copy application code
COPY onevalet/ onevalet/

EXPOSE 8000

CMD ["python", "-m", "onevalet", "--host", "0.0.0.0"]
