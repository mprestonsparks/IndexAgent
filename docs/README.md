# IndexAgent Documentation

## Running the Baseline Audit Locally

The Baseline Audit script helps you assess code quality, licensing, and test coverage for your repository. Follow the instructions below to run the audit locally.

### Prerequisites

Ensure the following tools are installed and available in your system PATH:

- **Python** (version 3.7 or higher recommended)
- **ruff** or **flake8** (for Python linting)
- **license-check** (for license compliance checks)
- **pytest** (for running Python tests)

You can install the Python tools using pip:

```sh
pip install ruff flake8 pytest
```

Install `license-check` as per its documentation (e.g., via pip or your package manager).

### Usage

Run the Baseline Audit script from the root of the repository:

```sh
bash scripts/baseline_audit.sh [REPO_PATH] [OPTIONS]
```

#### Arguments

- `REPO_PATH`: Path to the target repository to audit (required).
- `[OPTIONS]`: Additional options passed to the script (see script help for details).

#### Example

```sh
bash scripts/baseline_audit.sh ./my-repo
```

### Output

Audit results and logs are saved to:

```
logs/baseline_issues/<repo>/<timestamp>/
```

- `<repo>`: Name of the audited repository.
- `<timestamp>`: Time when the audit was run (format: YYYYMMDD_HHMMSS).

Check this directory for detailed reports and logs after the script completes.

### Troubleshooting

- **Missing Tools**:  
  If you see errors about missing commands (e.g., `ruff: command not found`), ensure all prerequisites are installed and available in your PATH.

- **Permissions**:  
  If you encounter permission errors, try running the script with elevated privileges or adjust file permissions as needed.

- **Python Version**:  
  Ensure you are using a compatible Python version (3.7+).

- **Other Issues**:  
  Review the output logs in the `logs/baseline_issues/<repo>/<timestamp>/` directory for more details on failures.

For further assistance, consult the script's comments or reach out to the project maintainers.