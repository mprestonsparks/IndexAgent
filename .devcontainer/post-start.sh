#!/bin/bash

# Post-start script for IndexAgent Dev Container
# This script runs every time the container starts

set -e

echo "🔄 Starting IndexAgent development session..."

# Ensure we're in the correct directory
cd /workspaces/IndexAgent

# Update PATH to include local bin
export PATH="$HOME/.local/bin:$PATH"

# Check if Docker daemon is accessible
if docker version &> /dev/null; then
    echo "✅ Docker daemon is accessible"
    
    # Show running containers if any
    RUNNING_CONTAINERS=$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | tail -n +2)
    if [ -n "$RUNNING_CONTAINERS" ]; then
        echo "🐳 Running containers:"
        echo "$RUNNING_CONTAINERS"
    fi
else
    echo "⚠️  Docker daemon not accessible. Start Docker on your host system if needed."
fi

# Check if the project dependencies are up to date
if [ -f "requirements-dev.txt" ]; then
    echo "🔍 Checking Python dependencies..."
    
    # Check if all requirements are satisfied
    if ! pip check &> /dev/null; then
        echo "⚠️  Some Python dependencies may need updating. Run 'pip install -r requirements-dev.txt' if needed."
    fi
fi

# Display project status
echo ""
echo "📊 Project Status:"
echo "   - Working directory: $(pwd)"
echo "   - Python version: $(python --version)"
echo "   - Git branch: $(git branch --show-current 2>/dev/null || echo 'not a git repository')"
echo "   - Git status: $(git status --porcelain 2>/dev/null | wc -l) modified files"

# Show available make targets if Makefile exists
if [ -f "Makefile" ]; then
    echo ""
    echo "🔧 Available make targets:"
    grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' Makefile 2>/dev/null | \
        awk 'BEGIN {FS = ":.*?## "}; {printf "   %-15s %s\n", $1, $2}' || \
        grep -E '^[a-zA-Z0-9_-]+:' Makefile 2>/dev/null | \
        head -10 | sed 's/:.*$//' | sed 's/^/   - /' || true
fi

# Show helpful reminders
echo ""
echo "💡 Quick reminders:"
echo "   - Use 'run-tests' to run the test suite"
echo "   - Use 'lint-code' to format and lint your code"
echo "   - Use 'start-stack' to start the Docker services"
echo "   - Ports 6070, 3000, and 8080 are forwarded to your host"

# Check for common development files and suggest next steps
echo ""
echo "🎯 Suggested next steps:"

if [ ! -f ".env" ] && [ -f ".env.dev" ]; then
    echo "   - Copy .env.dev to .env and customize for your environment"
fi

if [ -f "requirements-dev.txt" ] && [ ! -d ".git/hooks" ]; then
    echo "   - Set up pre-commit hooks: pre-commit install"
fi

if [ -d "tests" ] && [ "$(find tests -name '*.py' | wc -l)" -gt 0 ]; then
    echo "   - Run tests to verify everything works: run-tests"
fi

if [ -f "config/docker-compose.yml" ]; then
    echo "   - Start the full stack: start-stack"
fi

echo ""
echo "🚀 Ready for development!"