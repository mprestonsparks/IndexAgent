#!/usr/bin/env bash
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PARALLEL_LIMIT=10
DAG_ID="parallel_maintenance_poc"

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "info")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "success")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "warning")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        "error")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
    esac
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Pre-flight checks
print_status "info" "Starting pre-flight checks..."

# Check if Airflow is installed
if ! command_exists airflow; then
    print_status "error" "Airflow is not installed or not in PATH"
    exit 1
fi
print_status "success" "Airflow is installed"

# Check Airflow version
AIRFLOW_VERSION=$(airflow version 2>/dev/null || echo "unknown")
print_status "info" "Airflow version: $AIRFLOW_VERSION"

# Check if DAG file exists
DAG_FILE="$PROJECT_ROOT/dags/${DAG_ID}.py"
if [ ! -f "$DAG_FILE" ]; then
    print_status "error" "DAG file not found: $DAG_FILE"
    exit 1
fi
print_status "success" "DAG file found: $DAG_FILE"

# Check if Airflow home is set
if [ -z "${AIRFLOW_HOME:-}" ]; then
    print_status "warning" "AIRFLOW_HOME not set, using default"
    export AIRFLOW_HOME="$HOME/airflow"
fi
print_status "info" "AIRFLOW_HOME: $AIRFLOW_HOME"

# Check if database is initialized
if ! airflow db check 2>/dev/null; then
    print_status "error" "Airflow database is not initialized. Run 'airflow db init' first."
    exit 1
fi
print_status "success" "Airflow database is initialized"

# List available DAGs to verify our DAG is registered
print_status "info" "Checking if DAG is registered..."
if ! airflow dags list 2>/dev/null | grep -q "$DAG_ID"; then
    print_status "warning" "DAG '$DAG_ID' not found in registered DAGs"
    print_status "info" "Attempting to trigger DAG parsing..."
    airflow dags list >/dev/null 2>&1 || true
    sleep 2
fi

# Run the smoke test
print_status "info" "Starting smoke test for DAG: $DAG_ID"
print_status "info" "Parallel limit: $PARALLEL_LIMIT"
print_status "info" "Execution date: $(date +%Y-%m-%d)"

# Create a temporary file to capture output
TEMP_LOG=$(mktemp)
trap "rm -f $TEMP_LOG" EXIT

# Run the DAG test
if airflow dags test "$DAG_ID" "$(date +%Y-%m-%d)" 2>&1 | tee "$TEMP_LOG"; then
    print_status "success" "Smoke run completed successfully"
    
    # Check for any errors in the output
    if grep -qi "error\|exception\|failed" "$TEMP_LOG"; then
        print_status "warning" "Smoke run completed but found potential issues in output"
        print_status "info" "Please review the output above for any errors"
    fi
else
    print_status "error" "Smoke run failed"
    print_status "info" "Check the output above for error details"
    exit 1
fi

# Post-run checks
print_status "info" "Performing post-run checks..."

# Check if shard_status.db was created/updated
SHARD_DB="$PROJECT_ROOT/shard_status.db"
if [ -f "$SHARD_DB" ]; then
    print_status "success" "Shard status database found: $SHARD_DB"
    
    # Get database size
    DB_SIZE=$(du -h "$SHARD_DB" | cut -f1)
    print_status "info" "Database size: $DB_SIZE"
    
    # Check last modification time
    if [ "$(uname)" = "Darwin" ]; then
        LAST_MOD=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$SHARD_DB")
    else
        LAST_MOD=$(stat -c "%y" "$SHARD_DB" | cut -d'.' -f1)
    fi
    print_status "info" "Last modified: $LAST_MOD"
else
    print_status "warning" "Shard status database not found"
fi

# Summary
echo ""
print_status "success" "=== Smoke Test Summary ==="
print_status "info" "DAG: $DAG_ID"
print_status "info" "Execution Date: $(date +%Y-%m-%d)"
print_status "info" "Parallel Limit: $PARALLEL_LIMIT"
print_status "info" "Status: COMPLETED"
echo ""

exit 0