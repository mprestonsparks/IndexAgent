# IndexAgent

![CI](https://github.com/mprestonsparks/IndexAgent/workflows/CI/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/mprestonsparks/IndexAgent/badge.svg?branch=main)](https://coveralls.io/github/mprestonsparks/IndexAgent?branch=main)

**IndexAgent** is a Docker-first, Free and Open Source Software (FOSS) stack for self-hosted, large-scale code search and indexing. Designed for privacy, extensibility, and developer autonomy, IndexAgent empowers organizations and individuals to run their own code search infrastructure—**no vendor lock-in, no cloud dependencies**. IndexAgent can integrate with Apache Airflow for orchestrating maintenance workflows but does not include any DAG definitions within this repository.

## CI/CD Process

This project uses GitHub Actions for continuous integration and deployment:
- **Matrix Testing**: Automated tests run on Python 3.10 and 3.11
- **Quality Gates**: Code must pass black formatting, ruff linting, mypy type checking, and maintain ≥90% test coverage
- **Integration Tests**: Comprehensive integration testing suite runs separately from unit tests
- **Dependency Management**: Automated dependency updates via Dependabot
- **Efficient Caching**: Dependencies are cached for faster CI runs

---

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Further Documentation](#further-documentation)
- [License](#license)

## Introduction

IndexAgent uniquely integrates [Sourcebot](https://github.com/sourcebot-dev/sourcebot) for code discovery and syncing, [Zoekt](https://github.com/sourcegraph/zoekt) for high-performance indexing and search, and the [Claude Code CLI](https://www.npmjs.com/package/@anthropic-ai/claude-cli) for automated maintenance tasks. Built with a Docker-first approach, all components run in containers for consistent deployment across environments. As a self-hosted FOSS platform, it provides a privacy-focused, extensible code infrastructure—empowering full autonomy without vendor lock-in.

Routine maintenance workflows (TODO cleanup, documentation generation, coverage loops) are orchestrated via [Apache Airflow 3.0](https://airflow.apache.org/docs/apache-airflow/stable/index.html) DAGs in the companion [airflow-hub](https://github.com/mprestonsparks/airflow-hub) repository. Sourcebot ensures code is current, Zoekt powers sub-100 ms search across mounted repositories, and the Claude Code CLI applies intelligent patches—together realizing an autonomous, scalable maintenance ecosystem.

## Multi-Repository Integration

IndexAgent operates as part of a comprehensive multi-repository development workspace, providing code indexing and search capabilities across multiple projects:

### Integrated Services

- **IndexAgent** (Port 8081): Core code indexing and search API
- **Airflow-hub** (Port 8080): Workflow orchestration and DAG management
- **Market-analysis** (Port 8000): Financial data analysis and trading signals
- **Infra**: Infrastructure orchestration and deployment automation

### Cross-Service Communication

IndexAgent provides REST API endpoints that integrate seamlessly with other services:

```bash
# Health check endpoint
curl http://localhost:8081/health

# Search across indexed repositories
curl -X POST http://localhost:8081/search \
  -H "Content-Type: application/json" \
  -d '{"query": "function analyze", "repositories": ["market-analysis", "airflow-hub"]}'

# Repository indexing status
curl http://localhost:8081/repositories/status
```

### Shared Infrastructure

The multi-repository workspace provides:

- **Shared PostgreSQL Database**: Dedicated `indexagent` schema for metadata storage
- **Vault Integration**: Secure API key and credential management
- **Volume Mounting**: Unified access to `/repos`, `/data`, `/logs`, and `/secrets`
- **Service Discovery**: Automatic service registration and health monitoring

### Airflow Integration

IndexAgent works seamlessly with Preston's [airflow-hub](https://github.com/mprestonsparks/airflow-hub) to schedule maintenance DAGs but can also run standalone or integrate with any Airflow 3.0 deployment.

### Automated Maintenance Scripts

- [scripts/agent_fix_todos.sh](scripts/agent_fix_todos.sh): Phase 2 – nightly TODO cleanup via the Claude Code CLI.  
- [scripts/run_cov.py](scripts/run_cov.py) & [scripts/ai_test_loop.sh](scripts/ai_test_loop.sh): Phase 3 – continuous test and coverage loops.  
- [scripts/documentation/find_undocumented.py](scripts/documentation/find_undocumented.py): Phase 4 – detect missing or short docstrings.  
- [scripts/documentation/agent_write_docs.sh](scripts/documentation/agent_write_docs.sh): Phase 4 – generate markdown documentation.  
- [scripts/documentation/update_docs_coverage.sh](scripts/documentation/update_docs_coverage.sh): Phase 4 – compute and expose the `docs_coverage` metric.

### Claude Code CLI Integration

- **Planned usage:** wrap `claude -m` calls for code maintenance tasks (TODO fixes, doc generation, coverage loops) in the project directory (`--dir $(pwd)`), offline and interactive.  
- **Current scripts:**  
  - `scripts/agent_fix_todos.sh`: invokes Claude CLI to apply TODO patches.  
  - `scripts/documentation/agent_write_docs.sh`: invokes Claude CLI to generate markdown docs.  
- **Future enhancements:**  
  - CI integration to automate CLI tasks within build pipelines.  
  - Enhanced error handling, metrics, and logging for monitoring.

### Example Full System Flow

1. `make up` starts Docker containers for Sourcebot, Zoekt, and the Claude Code CLI service.  
2. Scheduled Airflow DAGs in [airflow-hub](https://github.com/mprestonsparks/airflow-hub) call wrapper scripts via SSHOperator or HTTP.  
3. Wrapper scripts run maintenance tasks, apply patches, and push commits/PRs back to source repositories.

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/)
- [GNU Make](https://www.gnu.org/software/make/) (optional)

### Development Environment Options

#### Option 1: Multi-Repository Dev Container (Recommended)

For integrated development across all repositories, use the multi-repository Dev Container workspace:

1. **Prerequisites:**
   - [VSCode](https://code.visualstudio.com/) with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/)

2. **Setup:**
   ```bash
   # Ensure all repositories are cloned as siblings
   ~/Documents/gitRepos/
     ├── airflow-hub/
     ├── IndexAgent/
     ├── market-analysis/
     └── infra/
   
   # Open the parent directory in VSCode
   code ~/Documents/gitRepos
   
   # Select "Reopen in Container" and choose the workspace configuration
   ```

3. **Benefits:**
   - Integrated development environment with all repositories
   - Shared PostgreSQL database with dedicated `indexagent` schema
   - Vault integration for secure credential management
   - Port 8081 for IndexAgent API access
   - Cross-repository communication and testing
   - Docker-in-Docker capabilities for infrastructure testing

#### Option 2: Standalone Dev Container

For IndexAgent-only development:

1. **Quick Start:**
   ```bash
   git clone https://github.com/your-org/IndexAgent.git
   cd IndexAgent
   code .
   # VSCode will prompt to "Reopen in Container" - click it!
   ```

2. **Benefits:**
   - Isolated IndexAgent development environment
   - Pre-configured Python 3.11, Docker, and all development tools
   - Automatic code formatting, linting, and testing setup
   - Port forwarding for web services (6070, 3000, 8081)

For detailed setup instructions, see [Dev Container Documentation](docs/devcontainer-setup.md).

#### Option 3: Local Development

For traditional local development setup:

### Docker Architecture

IndexAgent follows a Docker-first approach with three primary containers:

1. **zoekt-indexserver**: Handles code indexing and search capabilities
2. **sourcebot**: Manages code discovery and repository syncing
3. **indexagent**: Runs maintenance scripts and Claude Code CLI operations

All containers mount the same code repositories from `$HOME/repos` on the host, ensuring consistent access across the stack. For detailed Docker implementation information, see our [Docker documentation](docs/docker.md).

### Setup & Usage

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-org/IndexAgent.git
   cd IndexAgent
   ```

2. **Mount your code:**
   ```sh
   mkdir -p $HOME/repos
   # Place or clone your application repositories under $HOME/repos
   ```

3. **Manage secrets:**
   - For production, use [Docker secrets](https://docs.docker.com/engine/swarm/secrets/).  
   - For local development, use a `.env` file. See [`docker-compose.yml`](docker-compose.yml) for supported variables.

4. **Start the stack:**
   ```sh
   docker compose up -d
   ```
   Or:
   ```sh
   make up
   ```

5. **Stop the stack:**
   ```sh
   docker compose down
   ```
   Or:
   ```sh
   make down
   ```

6. **Access & advanced usage:**
   - Open [http://localhost:6070](http://localhost:6070) to explore the Zoekt web UI.
   - Access IndexAgent API at [http://localhost:8081](http://localhost:8081)
   - For API access and advanced configuration, see the [`docs/`](./docs/) directory.

## Further Documentation

See the [`docs/`](./docs/) directory for architecture diagrams, ADRs, and advanced configuration options.

### Development Environment

- **[Dev Container Setup](docs/devcontainer-setup.md)**: Complete guide for VSCode Dev Containers with cross-platform support
- **[Docker Documentation](docs/docker.md)**: Docker implementation details and container architecture

### Project Documentation

- **[Architecture](docs/architecture.md)**: System architecture and component interactions
- **[Runbook](docs/runbook.md)**: Operational procedures and maintenance tasks

## License

IndexAgent is released by [Preston Sparks](https://github.com/mprestonsparks) under a **GNU GPL** license. See the [LICENSE](LICENSE) file for details.