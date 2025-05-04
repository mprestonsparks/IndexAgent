#!/usr/bin/env bash
# Fixes TODOs found by Zoekt via Sourcebot API

set -euo pipefail
API="http://localhost:3000/api/internal/search"
tmp=$(mktemp)

curl -s "$API?q=TODO&num=1000&case=yes" >"$tmp" || {
  echo "Search API unreachable" >&2; exit 1; }

jq -c '.Results[]' "$tmp" | while read -r hit; do
  FILE=$(jq -r '.File' <<<"$hit")
  PREVIEW=$(jq -r '.Preview' <<<"$hit")

  claude -m "Resolve the TODO below while preserving functionality:
File: $FILE
Snippet:
$PREVIEW
Return only the patch." \
  --file "$FILE" --apply --dir "$(git rev-parse --show-toplevel)" \
  || echo "CLAUDE_ERROR|$FILE" >> .ai_errors.log
done