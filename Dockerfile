FROM python:3.12-slim AS base

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build deps first so cache survives code changes
COPY pyproject.toml README.md LICENSE NOTICE /app/
COPY genome/__init__.py /app/genome/__init__.py
RUN pip install --no-deps --target=/tmp/deps setuptools wheel \
 && pip install --upgrade pip

# Copy the rest of the source
COPY genome/ /app/genome/

# Install with server + postgres extras (common deployment profile)
RUN pip install ".[fastapi,postgres]"

# Pre-download the default embedding model so container starts fast
# Comment out if you want to use a different model / save image size.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Non-root user. /data is created and owned HERE, in the image: a fresh named
# volume inherits the image directory's ownership, whereas a mount point Docker
# has to create itself is root-owned and uid 1001 cannot open the SQLite file.
# (Bind mounts and cloud volumes keep their own ownership; chown them yourself.)
RUN useradd --create-home --uid 1001 genome \
 && mkdir -p /data \
 && chown -R genome:genome /app /data
USER genome

ENV GENOME_STORAGE=/data/memory.db \
    GENOME_HOST=0.0.0.0 \
    GENOME_PORT=8080

VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://localhost:${GENOME_PORT}/health" || exit 1

CMD ["python", "-m", "genome.server"]
