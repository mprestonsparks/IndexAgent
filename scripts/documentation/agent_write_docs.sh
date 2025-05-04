#!/bin/bash

# Define the path to the undoc.json file
UNDOC_FILE="/app/reports/undoc.json"

# Define the output directory for generated documentation
OUTPUT_DIR="/app/docs/auto"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR" || { echo "Error: Could not create output directory $OUTPUT_DIR"; exit 1; }

# Check if undoc.json exists
if [ ! -f "$UNDOC_FILE" ]; then
  echo "Error: $UNDOC_FILE not found."
  exit 1
fi

# Read and parse undoc.json to get the list of undocumented files
# Use jq to extract the 'file' value from each object in the JSON array
UNDOC_FILES=$(jq -r '.[].file' "$UNDOC_FILE")

# Check if jq command was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to parse $UNDOC_FILE with jq. Is jq installed and the JSON valid?"
  exit 1
fi

# Loop through each undocumented file
echo "$UNDOC_FILES" | while IFS= read -r file; do
  echo "Processing file: $file"

  # Extract the base module name from the file path (remove directory and .py extension)
  module=$(basename "$file" .py)

  # Define the output markdown file path
  output_md="$OUTPUT_DIR/$module.md"

  # Generate documentation using the claude command
  echo "Generating documentation for module: $module"

  # Read API key from Docker secret and set environment variable
  # Read API key from Docker secret and set environment variable
  export CLAUDE_API_KEY=$(cat /run/secrets/claude_api_key)

  # Export CLAUDE_MODEL environment variable (should be set on host)
  export CLAUDE_MODEL="$CLAUDE_MODEL"

  claude -p "Write a markdown page for module $module: overview, API table, usage examples." > "$output_md"

  # Check if the claude command was successful
  if [ $? -ne 0 ]; then
    echo "Warning: Failed to generate documentation for $file. Skipping linting."
    continue # Skip linting and move to the next file
  fi

  # Lint the generated markdown file
  echo "Linting generated documentation: $output_md"
  markdownlint -f "$output_md"

  # Check if the markdownlint command was successful
  if [ $? -ne 0 ]; then
    echo "Warning: markdownlint failed for $output_md. Please check the file manually."
    # Continue processing other files even if linting fails for one
  fi

done

echo "Documentation generation and linting process completed."
exit 0