FROM python:3.12-slim

# System deps for asyncpg and git (needed for momex git dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[all]"

# Copy application code
COPY koa/ koa/

# Copy migration files
COPY alembic.ini ./
COPY migrations/ migrations/

# Create non-root user
RUN groupadd -r koa --gid=1000 && useradd -r -g koa --uid=1000 koa
RUN chown -R koa:koa /app

USER koa

EXPOSE 8000

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "alembic upgrade head && python -m koa --host 0.0.0.0"]
