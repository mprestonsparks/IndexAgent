print("Script find_undocumented.py started.")
import os
import ast
import json

# Define the directory to scan
SCAN_DIRECTORY = "src/"
# Define the minimum length for a docstring (excluding whitespace)
DOCSTRING_THRESHOLD = 20
# Define the output file name
OUTPUT_FILE = "undoc.json"

def is_docstring_short(docstring):
    """Checks if a docstring is shorter than the defined threshold."""
    return len(docstring.strip()) < DOCSTRING_THRESHOLD

def find_undocumented_files(directory):
    """
    Scans a directory for Python files with missing or short module docstrings.

    Args:
        directory (str): The path to the directory to scan.

    Returns:
        list: A list of file paths (relative to the scanned directory)
              that are missing or have short docstrings.
    """
    undocumented_files = []

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Process only Python files
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                # Get the path relative to the scanned directory
                relative_filepath = os.path.relpath(filepath, directory)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Parse the file content into an AST
                    tree = ast.parse(content)

                    # Check for a module-level docstring
                    docstring = ast.get_docstring(tree)

                    # If docstring is missing or short, add to the list
                    if docstring is None or is_docstring_short(docstring):
                        undocumented_files.append(relative_filepath)

                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
                    # Optionally, you could add files that cause errors to a separate list

    return undocumented_files

if __name__ == "__main__":
    # Find undocumented files
    undocumented = find_undocumented_files(SCAN_DIRECTORY)

    # Write the list to a JSON file
    # Ensure the output directory exists if it's not the current directory
    output_path = os.path.join(".", OUTPUT_FILE) # Write to current working directory
    print(f"Attempting to write JSON to {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(undocumented, f, indent=4)
        print(f"Successfully wrote JSON to {output_path}")
        print(f"Found {len(undocumented)} undocumented files. List written to {output_path}")
    except Exception as e:
        print(f"Error writing output file {output_path}: {e}")