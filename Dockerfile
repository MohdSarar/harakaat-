FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." 2>/dev/null || pip install --no-cache-dir \
    torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e "."

# Application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run API
CMD ["python", "scripts/serve.py", "--port", "8000", "--checkpoint", "checkpoints/best.pt"]
