# Quality Assurance Tasks

This directory contains invoke tasks for running quality checks on the codebase.

## Setup

First, install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Available Tasks

### Individual Quality Checks

- **Black (Code Formatting)**
  ```bash
  invoke quality.black-check    # Check formatting
  invoke quality.black          # Auto-format code
  ```

- **Ruff (Linting)**
  ```bash
  invoke quality.ruff-check     # Check for linting issues
  invoke quality.ruff           # Auto-fix issues
  ```

- **MyPy (Type Checking)**
  ```bash
  invoke quality.mypy           # Run type checker
  ```

- **Testing**
  ```bash
  invoke quality.pytest-unit        # Run unit tests only
  invoke quality.pytest-integration # Run integration tests only
  invoke quality.pytest-all         # Run all tests with coverage check (≥90%)
  ```

### Combined Checks

- **Run All Quality Checks**
  ```bash
  invoke quality.all            # Run all checks (recommended before committing)
  ```

- **Coverage Report**
  ```bash
  invoke quality.coverage-report    # Generate HTML coverage report
  ```

## Quality Gates

The following quality gates are enforced:

1. **Black**: All code must be formatted according to Black's style
2. **Ruff**: Zero linting warnings/errors
3. **MyPy**: Zero type checking errors
4. **Coverage**: ≥90% code coverage

## CI/CD Integration

These tasks are designed to work in CI/CD environments. Each task will:
- Exit with code 0 on success
- Exit with code 1 on failure
- Provide clear, colored output (when supported)

## Configuration

Quality tool configurations are defined in:
- `pyproject.toml`: Black, Ruff, MyPy, and Coverage settings
- `invoke.yaml`: Invoke task runner configuration