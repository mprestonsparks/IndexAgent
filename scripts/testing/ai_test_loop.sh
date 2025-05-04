#!/bin/bash

# Ensure the script is executable
chmod +x scripts/run_cov.py

# Run the coverage script
python3 scripts/run_cov.py

# Check if the coverage report was generated
if [ -d "coverage_html_report" ]; then
    echo "Coverage report generated successfully."
else
    echo "Failed to generate coverage report."
    exit 1
fi

# Additional automation tasks can be added here
# For example, sending notifications or cleaning up temporary files

echo "Test automation completed."