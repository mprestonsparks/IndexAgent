# IndexAgent

**IndexAgent** is a Free and Open Source Software (FOSS) stack for self-hosted, large-scale code search and indexing. Designed for privacy, extensibility, and developer autonomy, IndexAgent empowers organizations and individuals to run their own code search infrastructure—**no vendor lock-in, no cloud dependencies**. IndexAgent is designed to be used in Airflow DAGs, but doesn't store any DAG files within the repository.

---

## Table of Contents

- [Introduction](#introduction)
- [Use Case & Integration](#use-case--integration)
- [Airflow Integration](#airflow-integration)
  - [Automated Maintenance Scripts](#automated-maintenance-scripts)
  - [Claude Code CLI Integration](#claude-code-cli-integration)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)

## Introduction

IndexAgent uniquely integrates Sourcebot for code discovery and syncing, Zoekt for high-performance indexing and search, and the Claude Code CLI for automated maintenance tasks. As a Free and Open Source Software (FOSS) stack, it provides a self-hosted, privacy-focused, and extensible platform—empowering organizations and individuals with full control over their code infrastructure without vendor lock-in or cloud dependencies.

Planned use of the Claude Code CLI includes driving nightly TODO cleanup, automated documentation generation, and continuous test coverage loops, creating a closed-loop maintenance pipeline. Workflows are orchestrated via Apache Airflow DAGs, Sourcebot keeps repositories up-to-date, Zoekt powers large-scale search, and the Claude Code CLI applies maintenance patches—together realizing an autonomous, scalable code maintenance ecosystem.

## Features

- **Self-hosted:** All components run locally or on your own infrastructure.
- **Extensible:** Built to support additional source integrations and indexing strategies.
- **FOSS:** Only uses open source libraries.

## Stack Components

- [**Sourcebot**](https://github.com/sourcebot-dev/sourcebot): Handles source code discovery, repository syncing, and metadata extraction. Keeps your codebase up-to-date within the IndexAgent stack.
- [**Zoekt**](https://github.com/sourcegraph/zoekt): A fast, scalable code search engine. Zoekt indexes your repositories and provides powerful search capabilities across all mounted code.

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/)
- [GNU Make](https://www.gnu.org/software/make/) (optional, for convenience)

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-org/IndexAgent.git
   cd IndexAgent
   ```

2. **Mount your code repositories:**
   - By default, IndexAgent expects your code to be available at `$HOME/repos` on the host.
   - You can change this path in [`docker-compose.yml`](docker-compose.yml) if needed.

3. **Start the stack:**
   - Using Docker Compose:
     ```sh
     docker compose up -d
     ```
   - Or, using the Makefile:
     ```sh
     make up
     ```

4. **Stop the stack:**
   - With Docker Compose:
     ```sh
     docker compose down
     ```
   - Or, with Makefile:
     ```sh
     make down
     ```

## Use Case & Integration

### Airflow Integration

IndexAgent is designed to integrate with Preston Sparks’s [airflow-hub](https://github.com/mprestonsparks/airflow-hub) repository for orchestrating maintenance DAGs. The system can also be adapted to run stand-alone or with a user’s own [Airflow 3.0](https://airflow.apache.org/docs/apache-airflow/stable/index.html) setup.

Example end-to-end flow:

1. IndexAgent containers ([Sourcebot](https://github.com/sourcebot-dev/sourcebot), [Zoekt](https://github.com/sourcegraph/zoekt), and [Claude Code CLI](https://www.npmjs.com/package/@anthropic-ai/claude-cli)) process a mounted repository.
2. Airflow (via SSHOperator or HTTP) in [airflow-hub](https://github.com/mprestonsparks/airflow-hub) schedules nightly tasks that call wrapper scripts in the IndexAgent containers.

### Automated Maintenance Scripts

- [**scripts/agent_fix_todos.sh**](scripts/agent_fix_todos.sh): Phase 2 – nightly TODO cleanup via Claude CLI.
- [**scripts/run_cov.py**](scripts/run_cov.py) & [**scripts/ai_test_loop.sh**](scripts/ai_test_loop.sh): Phase 3 – test and coverage loop.
- [**scripts/documentation/find_undocumented.py**](scripts/documentation/find_undocumented.py): Phase 4 – scan for missing/short docstrings.
- [**scripts/documentation/agent_write_docs.sh**](scripts/documentation/agent_write_docs.sh): Phase 4 – generate markdown docs.
- [**scripts/documentation/update_docs_coverage.sh**](scripts/documentation/update_docs_coverage.sh): Phase 4 – compute and expose docs_coverage metric.

_Note: `docs_coverage.prom` is produced for Prometheus scraping._

### Claude Code CLI Integration

A brief overview of planned usage:

- Wrapping calls to the Claude Code CLI for code maintenance tasks (TODO fixes, doc generation, etc.).
- Running locally with no external network (`--dir $(pwd)`), interactive CLI agent.

Current implementation summary:

- `scripts/agent_fix_todos.sh` invokes `claude -m` to apply patches to TODOs.
- `scripts/documentation/agent_write_docs.sh` invokes `claude -m` to generate markdown docs.

Future planned enhancements:

- CI integration to automate CLI tasks within build pipelines.
- Error-handling metrics and logging for monitoring and troubleshooting.

### Example Full System Flow

- `make up` starts [Zoekt](https://github.com/sourcegraph/zoekt), [Sourcebot](https://github.com/sourcebot-dev/sourcebot), and exposes the [Claude Code CLI](https://www.npmjs.com/package/@anthropic-ai/claude-cli) endpoint.
- Airflow DAGs in [airflow-hub](https://github.com/mprestonsparks/airflow-hub) call wrapper scripts such as [scripts/agent_fix_todos.sh](scripts/agent_fix_todos.sh), [scripts/documentation/agent_write_docs.sh](scripts/documentation/agent_write_docs.sh), and [scripts/documentation/update_docs_coverage.sh](scripts/documentation/update_docs_coverage.sh) to fix TODOs, refresh docs, and generate metrics.
- Each task pushes commits and PRs back to the source repositories.

## Mounting Code & Secret Management

- **Mounting Repositories:**  
  The stack expects your code to be available at `$HOME/repos` on the host. You can change this path in `docker-compose.yml` if needed.

- **Secrets:**
  - For production, use [Docker secrets](https://docs.docker.com/engine/swarm/secrets/) to manage sensitive information.
  - For local development, you may use a `.env` file in the project root. See `docker-compose.yml` for supported environment variables.

## Further Documentation

See the [`docs/`](./docs/) directory for detailed documentation, architecture diagrams, and advanced configuration options.

## Usage

Once the stack is running and your repositories are mounted, you can access Zoekt’s web interface (by default at [http://localhost:6070](http://localhost:6070)) to perform code searches.

**Example:**
- Open your browser and go to [http://localhost:6070](http://localhost:6070)
- Enter a search query (e.g., `func main`) to search across all indexed repositories.

For advanced usage and API access, see the documentation in [`docs/`](./docs/).

---

## License

IndexAgent is released by [Preston Sparks](https://github.com/mprestonsparks) under a **GNU GPL** license. See [`LICENSE`](LICENSE) for details.