# Python Development Infrastructure Analysis

This document analyzes the Python environment setup, testing infrastructure, and development workflows across the three repositories: **infra**, **airflow-hub**, and **IndexAgent**.

## 1. Python Environment Analysis

### IndexAgent Repository

#### Dependency Management
- **Method**: Requirements files (no Poetry/pip-tools)
- **Files**: 
  - [`requirements-dev.txt`](requirements-dev.txt) - Development dependencies only
  - No production requirements.txt found
- **Key Dependencies**:
  ```
  pytest>=7.4.0
  pytest-cov>=4.1.0
  pytest-airflow>=0.2.0
  coverage>=7.3.0
  black>=23.7.0
  ruff>=0.1.0
  mypy>=1.5.0
  invoke>=2.2.0
  pytest-asyncio>=0.21.0
  pytest-mock>=3.11.0
  ```

#### Python Version
- **Version**: Python 3.11 (from [`Dockerfile`](Dockerfile:2))
- **Type Checking**: Configured for Python 3.10 in [`pyproject.toml`](pyproject.toml:24)
- **Black Target**: Python 3.10 and 3.11 ([`pyproject.toml`](pyproject.toml:16))

#### Code Quality Tools
- **Formatter**: Black (line-length: 100)
- **Linter**: Ruff with extensive rule selection
- **Type Checker**: MyPy with strict settings:
  - `warn_return_any = true`
  - `warn_unused_configs = true`
  - `disallow_untyped_defs = true`

### airflow-hub Repository

#### Dependency Management
- **Method**: pip with requirements files and pip-tools
- **Files**:
  - [`requirements.txt`](../airflow-hub/requirements.txt) - Production dependencies
  - [`requirements/requirements-dev.txt`](../airflow-hub/requirements/requirements-dev.txt) - Generated with pip-compile
  - [`requirements/requirements-dev.in`](../airflow-hub/requirements/requirements-dev.in) - Source for dev requirements
- **Key Dependencies**:
  ```
  apache-airflow==3.0.0
  apache-airflow-providers-snowflake==6.3.1
  apache-airflow-providers-docker==4.3.1
  pandas==2.1.4
  numpy==1.26.4
  scikit-learn==1.4.2
  pytest==7.3.1 (in main requirements)
  flake8==7.2.0 (in dev requirements)
  ```

#### Python Version
- **Version**: Python 3.11 (from [`Dockerfile.test`](../airflow-hub/Dockerfile.test:3))
- **Constraints**: Uses Airflow 3.0.0 constraints for Python 3.11

#### Code Quality Tools
- **Linter**: flake8 (from dev requirements)
- **No formatter or type checker configured**

### infra Repository

#### Dependency Management
- **No Python dependency files found**
- **Purpose**: Infrastructure orchestration only
- **Scripts**: Shell scripts only ([`scripts/check-ports.sh`](../infra/scripts/check-ports.sh))

## 2. Testing Infrastructure Analysis

### IndexAgent Repository

#### Test Structure
```
tests/
├── test_claude_cli.py
├── test_devcontainer_integration.py
├── test_example.py
├── test_worktree_manager.py
└── integration/
    ├── __init__.py
    ├── test_high_load_20.py
    └── test_parallel_dag.py
```

#### Testing Framework
- **Framework**: pytest with plugins
- **Configuration**: [`pytest.ini`](pytest.ini)
  ```ini
  [pytest]
  addopts = --maxfail=1 --disable-warnings -q -p no:pytest_asyncio -p no:asyncio
  ```
- **Coverage**: Configured in [`pyproject.toml`](pyproject.toml:1-12)
  - Branch coverage enabled
  - Source: `indexagent` module
  - Exclusions for pragma, repr, assertions

#### Test Execution
- **Coverage Script**: [`scripts/run_cov.py`](scripts/run_cov.py) - Python wrapper for pytest
- **Task Runner**: Invoke with [`tasks/quality.py`](tasks/quality.py)
  - Unit tests: `invoke pytest-unit`
  - Integration tests: `invoke pytest-integration`
  - All tests with 90% coverage requirement: `invoke pytest-all`

### airflow-hub Repository

#### Test Structure
```
tests/
├── __init__.py
├── test_dag_validation.py
├── test_market_analysis_dag.py
├── dags/
├── plugins/
│   ├── __init__.py
│   ├── project_analytics/
│   │   ├── __init__.py
│   │   └── test_data_quality_operator.py
│   └── project_trading/
│       ├── __init__.py
│       └── test_ibkr_data_operator.py
└── scripts/
    └── test_extract_ibkr_data_cli.py
```

#### Testing Framework
- **Framework**: pytest
- **Configuration**: [`pytest.ini`](../airflow-hub/pytest.ini)
  - Logging enabled
  - Deprecation warnings filtered
  - Environment variables removed (handled by Dockerfile)
- **Test Types**:
  - DAG validation tests
  - Plugin/operator tests
  - Script tests

#### Test Execution
- **Docker-based**: [`Dockerfile.test`](../airflow-hub/Dockerfile.test)
- **Default command**: `pytest -v -s tests/`

### infra Repository

#### Testing Infrastructure
- **No Python tests found**
- **Purpose**: Infrastructure validation only

## 3. Script Organization Analysis

### IndexAgent Repository

#### Script Structure
```
scripts/
├── README.md
├── agent_fix_todos.sh
├── ai_test_loop.sh
├── check_system.py
├── run_cov.py
├── smoke_run.sh
├── test_sanity_checks.sh
├── documentation/
│   ├── agent_write_docs.sh
│   ├── find_undocumented.py
│   └── update_docs_coverage.sh
├── maintenance/
│   ├── agent_fix_todos.sh
│   └── baseline_audit.sh
└── testing/
    ├── ai_test_loop.sh
    └── run_cov.py
```

#### Naming Conventions
- Shell scripts: `snake_case.sh`
- Python scripts: `snake_case.py`
- AI-assisted scripts: `agent_*.sh` or `ai_*.sh`

#### Script Categories
- **Testing**: Coverage, test loops, sanity checks
- **Documentation**: Finding undocumented code, coverage updates
- **Maintenance**: TODO fixes, baseline audits
- **System**: Environment checks

### airflow-hub Repository

#### Script Structure
```
scripts/
├── bootstrap_airflow.py
└── inventory_rename.py
```

#### Script Purpose
- **bootstrap_airflow.py**: Airflow initialization
- **inventory_rename.py**: Data management utility

### infra Repository

#### Script Structure
```
scripts/
└── check-ports.sh
```

#### Script Purpose
- **check-ports.sh**: Port availability validation

## 4. Development Workflow Analysis

### IndexAgent Repository

#### Workflow Tools
1. **Task Runner**: Invoke
   - Configuration: [`invoke.yaml`](invoke.yaml)
   - Tasks: [`tasks/quality.py`](tasks/quality.py)
   - Commands:
     - `invoke black` - Format code
     - `invoke ruff` - Fix linting issues
     - `invoke mypy` - Type check
     - `invoke all` - Run all quality checks

2. **Makefile**: Docker and service management
   - `make up` - Start services
   - `make down` - Stop services
   - `make find-undocumented` - Documentation check

3. **Coverage Requirements**:
   - Enforced 90% minimum coverage
   - HTML reports: `invoke coverage-report`

#### Development Process
1. Code formatting with Black
2. Linting with Ruff
3. Type checking with MyPy
4. Unit and integration testing
5. Coverage verification

### airflow-hub Repository

#### Workflow Tools
1. **No task runner configured**
2. **Docker-based testing**: 
   - Build: `docker build -f Dockerfile.test -t airflow-test .`
   - Run: `docker run airflow-test`

3. **pip-tools for dependency management**:
   - Compile: `pip-compile requirements/requirements-dev.in`

#### Development Process
1. DAG validation testing
2. Plugin/operator testing
3. Docker-based test execution

### infra Repository

#### Workflow Tools
1. **Makefile**: Service orchestration
   - `make check-ports` - Validate ports
   - `make up` - Start all services

2. **Multi-repo coordination**:
   - Orchestrates docker-compose files from multiple repos
   - Environment variable management via `.env`

## 5. CI/CD Pipeline Analysis

### Current State
- **No CI/CD configurations found** in any repository
- No `.github/workflows/`, `.gitlab-ci.yml`, or other CI files
- No pre-commit hooks configured

### Recommendations for CI/CD

#### GitHub Actions Workflow
```yaml
name: Python CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    
    - name: Run quality checks
      run: |
        invoke all
```

## 6. Integration Points for Agent-Evolution Code

### Recommended Structure
```
agent-evolution/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── agent.py
│   ├── evolution.py
│   └── fitness.py
├── operators/
│   ├── __init__.py
│   ├── mutation.py
│   ├── crossover.py
│   └── selection.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
└── tests/
    ├── __init__.py
    ├── test_agent.py
    ├── test_evolution.py
    └── integration/
        └── test_full_evolution.py
```

### Integration Approach

1. **In IndexAgent**: Add as a new module under `indexagent/`
2. **In airflow-hub**: Create as an Airflow plugin for DAG-based evolution
3. **Testing**: Follow existing pytest patterns with coverage requirements

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install agent-evolution in development mode
pip install -e .

# Run tests with coverage
invoke pytest-all
```

## 7. Key Findings and Recommendations

### Strengths
1. **IndexAgent**: Well-structured with comprehensive tooling
2. **Type safety**: MyPy configuration in IndexAgent
3. **Test organization**: Clear separation of unit/integration tests
4. **Task automation**: Invoke tasks for common operations

### Areas for Improvement
1. **Standardize Python versions**: All repos should use 3.11
2. **Add pre-commit hooks**: Automate quality checks
3. **Implement CI/CD**: GitHub Actions for all repos
4. **Unify dependency management**: Consider Poetry or pip-tools across all repos
5. **Add production requirements**: IndexAgent lacks production dependencies
6. **Enhance airflow-hub tooling**: Add formatter, type checker, task runner

### Immediate Actions
1. Create `.pre-commit-config.yaml` for automated checks
2. Add GitHub Actions workflows
3. Standardize development dependencies across repos
4. Document Python version requirements
5. Create unified development setup guide