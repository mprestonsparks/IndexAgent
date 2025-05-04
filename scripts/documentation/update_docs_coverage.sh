#!/usr/bin/env bash
set -euo pipefail

# Run documentation scan to generate undoc.json
python3 scripts/documentation/find_undocumented.py

# Count total Python modules
TOTAL=$(find src -type f -name '*.py' | wc -l | tr -d ' ')
# Count undocumented modules
UNDOC=$(jq length undoc.json)

# Compute coverage percentage with two decimal places
COVERAGE=$(awk "BEGIN {printf \"%.2f\", ($TOTAL - $UNDOC)/$TOTAL * 100}")

# Write Prometheus gauge file
echo "docs_coverage $COVERAGE" > docs_coverage.prom