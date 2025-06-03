#!/bin/bash

# Post-create script for IndexAgent Dev Container
# This script runs once after the container is created

set -e

echo "🚀 Setting up IndexAgent development environment..."

# Ensure we're in the correct directory
cd /workspaces/IndexAgent

# Install Python dependencies if requirements files exist
if [ -f "requirements-dev.txt" ]; then
    echo "📦 Installing development dependencies..."
    pip install --user -r requirements-dev.txt
fi

if [ -f "requirements.txt" ]; then
    echo "📦 Installing production dependencies..."
    pip install --user -r requirements.txt
fi

# Install the project in development mode if setup.py or pyproject.toml exists
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "📦 Installing project in development mode..."
    pip install --user -e .
fi

# Set up pre-commit hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
fi

# Create useful directories
echo "📁 Creating development directories..."
mkdir -p ~/.local/bin
mkdir -p ~/.cache/pip
mkdir -p ~/.cache/mypy
mkdir -p ~/.cache/ruff

# Set up git configuration if not already configured
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Git user not configured. You may want to run:"
    echo "   git config --global user.name 'Your Name'"
    echo "   git config --global user.email 'your.email@example.com'"
fi

# Create a development environment file template
if [ ! -f ".env.dev" ]; then
    echo "📝 Creating development environment template..."
    cat > .env.dev << 'EOF'
# Development environment variables for IndexAgent
# Copy this to .env and customize as needed

# Docker platform (auto-detected)
DOCKER_DEFAULT_PLATFORM=linux/amd64

# Port configurations
ZOEKT_UI_PORT=6070
SOURCEBOT_UI_PORT=3000
INDEXAGENT_API_PORT=8080

# Claude configuration
CLAUDE_MODEL=claude-3-opus-2025-05-02

# Development flags
DEBUG=true
INDEXAGENT_DEV_CONTAINER=true

# Paths (adjust as needed)
# HOME_REPOS_PATH=${HOME}/repos
EOF
fi

# Set up shell completion for common tools
echo "🔧 Setting up shell completions..."

# Docker completion
if command -v docker &> /dev/null; then
    docker completion bash > ~/.local/share/bash-completion/completions/docker 2>/dev/null || true
fi

# Make completion
if command -v make &> /dev/null; then
    complete -W "\`grep -oE '^[a-zA-Z0-9_.-]+:([^=]|$)' Makefile 2>/dev/null | sed 's/[^a-zA-Z0-9_.-]*$//'\`" make 2>/dev/null || true
fi

# Create useful development scripts
echo "📝 Creating development helper scripts..."

# Script to quickly run tests
cat > ~/.local/bin/run-tests << 'EOF'
#!/bin/bash
# Quick test runner for IndexAgent

set -e

echo "🧪 Running IndexAgent tests..."

# Run with coverage if pytest-cov is available
if python -c "import pytest_cov" 2>/dev/null; then
    python -m pytest tests/ --cov=indexagent --cov-report=term-missing --cov-report=xml
else
    python -m pytest tests/
fi
EOF

# Script to run linting and formatting
cat > ~/.local/bin/lint-code << 'EOF'
#!/bin/bash
# Code quality checks for IndexAgent

set -e

echo "🔍 Running code quality checks..."

# Format with black
if command -v black &> /dev/null; then
    echo "🎨 Formatting with black..."
    black --line-length=100 .
fi

# Lint with ruff
if command -v ruff &> /dev/null; then
    echo "🔍 Linting with ruff..."
    ruff check .
fi

# Type check with mypy
if command -v mypy &> /dev/null; then
    echo "🔍 Type checking with mypy..."
    mypy indexagent/ || true
fi

echo "✅ Code quality checks complete!"
EOF

# Script to start the full stack
cat > ~/.local/bin/start-stack << 'EOF'
#!/bin/bash
# Start the IndexAgent Docker stack

set -e

echo "🐳 Starting IndexAgent Docker stack..."

if [ -f "config/docker-compose.yml" ]; then
    cd config
    docker-compose up -d
    echo "✅ Stack started! Services available at:"
    echo "   - Zoekt UI: http://localhost:6070"
    echo "   - Sourcebot UI: http://localhost:3000"
    echo "   - IndexAgent API: http://localhost:8080"
else
    echo "❌ docker-compose.yml not found in config directory"
    exit 1
fi
EOF

# Make scripts executable
chmod +x ~/.local/bin/run-tests
chmod +x ~/.local/bin/lint-code
chmod +x ~/.local/bin/start-stack

# Verify Docker access
echo "🐳 Verifying Docker access..."
if docker version &> /dev/null; then
    echo "✅ Docker is accessible"
else
    echo "⚠️  Docker may not be accessible. This is normal if Docker is not running on the host."
fi

# Display helpful information
echo ""
echo "🎉 IndexAgent development environment setup complete!"
echo ""
echo "📚 Helpful commands:"
echo "   run-tests     - Run the test suite with coverage"
echo "   lint-code     - Run code formatting and linting"
echo "   start-stack   - Start the full Docker stack"
echo ""
echo "🔧 Development tools installed:"
echo "   - Python $(python --version | cut -d' ' -f2)"
echo "   - Docker $(docker --version | cut -d' ' -f3 | cut -d',' -f1)"
echo "   - Node.js $(node --version)"
echo "   - Claude CLI $(claude --version 2>/dev/null || echo 'installed')"
echo ""
echo "📁 Useful directories:"
echo "   - Project: /workspaces/IndexAgent"
echo "   - Repos: /repos"
echo "   - Local bin: ~/.local/bin"
echo ""
echo "Happy coding! 🚀"