#!/bin/bash

# Test script for IndexAgent Dev Container
# This script verifies that all components are working correctly

set -e

echo "🧪 Testing IndexAgent Dev Container Setup..."
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing $test_name... "
    
    if eval "$test_command" &> /dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
    fi
}

# Function to run a test with output
run_test_with_output() {
    local test_name="$1"
    local test_command="$2"
    
    echo "Testing $test_name..."
    
    if eval "$test_command"; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
    fi
    echo ""
}

echo "🐍 Python Environment Tests"
echo "----------------------------"

# Test Python installation
run_test "Python 3.11 installation" "python --version | grep -q '3.11'"

# Test pip installation
run_test "pip installation" "pip --version"

# Test Python path
run_test "Python path configuration" "python -c 'import sys; print(sys.path)' | grep -q '/workspaces/IndexAgent'"

# Test development dependencies
echo ""
echo "📦 Development Dependencies Tests"
echo "---------------------------------"

run_test "black formatter" "black --version"
run_test "ruff linter" "ruff --version"
run_test "mypy type checker" "mypy --version"
run_test "pytest test runner" "pytest --version"
run_test "coverage tool" "coverage --version"
run_test "invoke task runner" "invoke --version"

echo ""
echo "🔧 System Tools Tests"
echo "---------------------"

run_test "git installation" "git --version"
run_test "make installation" "make --version"
run_test "curl installation" "curl --version"
run_test "jq installation" "jq --version"

echo ""
echo "🐳 Docker Tests"
echo "---------------"

run_test "Docker CLI installation" "docker --version"
run_test "Docker Compose installation" "docker-compose --version"

# Test Docker daemon access (may fail if Docker not running)
echo -n "Testing Docker daemon access... "
if docker version &> /dev/null; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((TESTS_PASSED++))
    
    # If Docker is accessible, test container operations
    run_test "Docker container operations" "docker run --rm hello-world"
else
    echo -e "${YELLOW}⚠ SKIP (Docker daemon not accessible)${NC}"
fi

echo ""
echo "🌐 Node.js Tools Tests"
echo "----------------------"

run_test "Node.js installation" "node --version"
run_test "npm installation" "npm --version"
run_test "Claude CLI installation" "claude --version || test -f /usr/local/bin/claude"
run_test "markdownlint installation" "markdownlint --version"

echo ""
echo "📁 File System Tests"
echo "--------------------"

run_test "Workspace directory" "test -d /workspaces/IndexAgent"
run_test "Repos directory" "test -d /repos"
run_test "Local bin directory" "test -d ~/.local/bin"

# Test helper scripts
run_test "run-tests script" "test -x ~/.local/bin/run-tests"
run_test "lint-code script" "test -x ~/.local/bin/lint-code"
run_test "start-stack script" "test -x ~/.local/bin/start-stack"

echo ""
echo "🔍 Project Structure Tests"
echo "--------------------------"

# Test if we're in the right directory
run_test "Working directory" "test '$(pwd)' = '/workspaces/IndexAgent'"

# Test project files
run_test "pyproject.toml exists" "test -f pyproject.toml"
run_test "requirements-dev.txt exists" "test -f requirements-dev.txt"
run_test "Makefile exists" "test -f Makefile"

# Test source directories
run_test "indexagent package" "test -d indexagent"
run_test "tests directory" "test -d tests"
run_test "docs directory" "test -d docs"

echo ""
echo "⚙️ Configuration Tests"
echo "----------------------"

# Test git configuration
run_test "Git line ending config" "git config core.eol | grep -q lf"
run_test "Git autocrlf config" "git config core.autocrlf | grep -q input"

# Test Python configuration
run_test "PYTHONPATH environment" "echo \$PYTHONPATH | grep -q '/workspaces/IndexAgent'"

echo ""
echo "🧪 Python Package Tests"
echo "-----------------------"

# Test if we can import key packages
run_test "pytest import" "python -c 'import pytest'"
run_test "coverage import" "python -c 'import coverage'"
run_test "black import" "python -c 'import black'"

# Test if project packages can be imported
if [ -d "indexagent" ] && [ -f "indexagent/__init__.py" ]; then
    run_test "indexagent package import" "python -c 'import indexagent'"
fi

echo ""
echo "🔧 VSCode Integration Tests"
echo "---------------------------"

# Test if we're in a Dev Container
run_test "Dev Container environment" "test '\$INDEXAGENT_DEV_CONTAINER' = 'true'"

# Test workspace folder
run_test "VSCode workspace folder" "test '\$PWD' = '/workspaces/IndexAgent'"

echo ""
echo "📊 Test Results Summary"
echo "======================="

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))

echo "Total tests run: $TOTAL_TESTS"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"

if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
    echo ""
    echo -e "${YELLOW}Some tests failed. This may be normal if:${NC}"
    echo "- Docker daemon is not running on the host"
    echo "- Some optional dependencies are not installed"
    echo "- Network connectivity issues"
    echo ""
    echo "Check the failed tests and ensure critical functionality works."
    exit 1
else
    echo -e "Tests failed: ${GREEN}0${NC}"
    echo ""
    echo -e "${GREEN}🎉 All tests passed! Your Dev Container is ready for development.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run 'run-tests' to execute the project test suite"
    echo "2. Run 'lint-code' to check code quality"
    echo "3. Run 'start-stack' to start the Docker services"
    echo ""
    echo "Happy coding! 🚀"
fi