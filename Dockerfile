# Minimal Dockerfile for IndexAgent
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Copy the source code and test directories
COPY src /app/src
COPY tests /app/tests

# Copy the entire scripts directory, including nested subdirectories
COPY scripts/ /app/scripts/

# Install necessary Python dependencies (pytest and pytest-cov)
RUN pip install pytest pytest-cov
# Install Node.js and npm for Claude CLI and markdownlint
RUN apt-get update && apt-get install -y curl nodejs npm jq && rm -rf /var/lib/apt/lists/*
# Install Claude Code CLI and markdownlint
RUN npm install -g @anthropic-ai/claude-code@0.2.64
RUN npm install -g markdownlint-cli

# Keep the container running for now (e.g., for debugging)
CMD ["tail", "-f", "/dev/null"]