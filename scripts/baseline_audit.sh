#!/bin/bash

set -e

# Usage: baseline_audit.sh <repo_path> [--diagnostic=(true|false)]
# Default: --diagnostic=true

# Parse arguments
REPO_PATH=""
DIAGNOSTIC=true

for arg in "$@"; do
    case $arg in
        --diagnostic=*)
            DIAGNOSTIC="${arg#*=}"
            shift
            ;;
        *)
            if [[ -z "$REPO_PATH" ]]; then
                REPO_PATH="$arg"
            else
                echo "Unknown argument: $arg"
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$REPO_PATH" ]]; then
    echo "[ERROR] Usage: $0 <repo_path> [--diagnostic=(true|false)]"
    exit 1
fi

if [[ ! -d "$REPO_PATH" ]]; then
    echo "[ERROR] Provided repo_path '$REPO_PATH' is not a directory."
    exit 1
fi

REPO_NAME=$(basename "$REPO_PATH")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/baseline_issues/$REPO_NAME/$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "[INFO] Baseline audit for repo: $REPO_NAME"
echo "[INFO] Logs will be saved to: $LOG_DIR"

# Helper to check command existence
function check_cmd() {
    command -v "$1" >/dev/null 2>&1
}

# Linter: ruff or flake8
LINTER=""
if check_cmd ruff; then
    LINTER="ruff"
elif check_cmd flake8; then
    LINTER="flake8"
else
    echo "[ERROR] Neither 'ruff' nor 'flake8' is installed. Please install one."
    exit 1
fi

# License checker
if ! check_cmd license-check; then
    echo "[ERROR] 'license-check' is not installed. Please install it."
    exit 1
fi

# Pytest
if ! check_cmd pytest; then
    echo "[ERROR] 'pytest' is not installed. Please install it."
    exit 1
fi

# Run diagnostics if enabled
if [[ "$DIAGNOSTIC" == "true" ]]; then
    echo "[INFO] Running $LINTER..."
    $LINTER "$REPO_PATH" > "$LOG_DIR/linter_report.txt" 2>&1 || true
    echo "[INFO] Linter report saved to $LOG_DIR/linter_report.txt"

    echo "[INFO] Running license-check..."
    license-check "$REPO_PATH" > "$LOG_DIR/license_report.txt" 2>&1 || true
    echo "[INFO] License report saved to $LOG_DIR/license_report.txt"

    echo "[INFO] Running pytest..."
    pytest --quiet "$REPO_PATH" > "$LOG_DIR/pytest_report.txt" 2>&1 || true
    echo "[INFO] Pytest report saved to $LOG_DIR/pytest_report.txt"
else
    echo "[INFO] --diagnostic=false, skipping checks."
fi

echo "[INFO] Baseline audit complete for $REPO_NAME."