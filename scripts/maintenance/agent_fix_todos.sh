#!/usr/bin/env bash
# Fixes TODOs found by Zoekt via Sourcebot API

set -euo pipefail
API="http://localhost:3000/api/internal/search"
tmp=$(mktemp)

# Create a new branch for automated TODO fixes
REPO_ROOT=$(git rev-parse --show-toplevel)
TIMESTAMP=$(date +%Y%m%d%H%M%S)
BRANCH="indexagent/todo-fixes-$TIMESTAMP"
cd "$REPO_ROOT"
git checkout -b "$BRANCH"

curl -s "$API?q=TODO&num=1000&case=yes" >"$tmp" || {
  echo "Search API unreachable" >&2; exit 1; }

jq -c '.Results[]' "$tmp" | while read -r hit; do
  FILE=$(jq -r '.File' <<<"$hit")
  PREVIEW=$(jq -r '.Preview' <<<"$hit")

  # Read API key from Docker secret and set environment variable
  export CLAUDE_API_KEY=$(cat /run/secrets/claude_api_key)

  claude -m "Resolve the TODO below while preserving functionality:
File: $FILE
Snippet:
$PREVIEW
Return only the patch." \
  --file "$FILE" --apply --dir "$(git rev-parse --show-toplevel)" \
  || echo "CLAUDE_ERROR|$FILE" >> .ai_errors.log
done

# Commit and push changes
cd "$REPO_ROOT"
git add .
git commit -m "Automated TODO fixes by IndexAgent"
git push -u origin "$BRANCH"