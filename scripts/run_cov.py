"""
Script to run pytest with coverage analysis.
"""

import pytest
import os

# Define the path to the tests directory and the source directory for coverage
TEST_DIR = "tests/"
SOURCE_DIR = "src/"

# Check if the test directory exists
if not os.path.isdir(TEST_DIR):
    print(f"Error: Test directory '{TEST_DIR}' not found.")
    exit(1)

# Check if the source directory exists (optional, but good practice)
if not os.path.isdir(SOURCE_DIR):
    print(f"Warning: Source directory '{SOURCE_DIR}' not found. Coverage report may be empty.")
    # Continue anyway, as tests might still run

# Run pytest with coverage
# --cov=src: Specifies the source directory to measure coverage for
# tests/: Specifies the directory containing the tests to run
print(f"Running tests in '{TEST_DIR}' with coverage for '{SOURCE_DIR}'...")
exit_code = pytest.main([f"--cov={SOURCE_DIR}", TEST_DIR])

# Exit with the same exit code as pytest
exit(exit_code)