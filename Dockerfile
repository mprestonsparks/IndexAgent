# IndexAgent API Server Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    nodejs \
    npm \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js tools
RUN npm install -g @anthropic-ai/claude-code@0.2.64
RUN npm install -g markdownlint-cli

# Copy application code
COPY main.py .
COPY main_api.py .
COPY src/ /app/src/
COPY indexagent/ /app/indexagent/
COPY tests/ /app/tests/
COPY scripts/ /app/scripts/

# Create necessary directories
RUN mkdir -p /app/logs /app/reports /app/docs/auto /tmp/worktrees

# Expose port 8081 for IndexAgent API
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Run the IndexAgent API application
CMD ["python", "main_api.py"]