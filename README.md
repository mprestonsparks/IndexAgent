# IndexAgent

**IndexAgent** is a Free and Open Source Software (FOSS) stack for self-hosted, large-scale code search and indexing. Designed for privacy, extensibility, and developer autonomy, IndexAgent empowers organizations and individuals to run their own code search infrastructure—**no vendor lock-in, no cloud dependencies, and no Airflow DAGs in this repository**.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Stack Components](#stack-components)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage Example](#usage-example)
- [Mounting Code & Secret Management](#mounting-code--secret-management)
- [Further Documentation](#further-documentation)
- [License](#license)

---

## Overview

IndexAgent provides a robust, extensible platform for searching and indexing source code at scale. All components are self-hosted, giving you full control over your code search infrastructure and data.

## Features

- **Self-hosted:** All components run locally or on your own infrastructure.
- **No Airflow DAGs:** This repository does not include any Airflow DAGs or orchestration logic.
- **Extensible:** Built to support additional source integrations and indexing strategies.
- **FOSS:** Licensed for open collaboration and community-driven improvement.

## Stack Components

- **Sourcebot:**  
  Handles source code discovery, repository syncing, and metadata extraction. Keeps your codebase up-to-date within the IndexAgent stack.

- **Zoekt:**  
  A fast, scalable code search engine. Zoekt indexes your repositories and provides powerful search capabilities across all mounted code.

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
   - You can change this path in `docker-compose.yml` if needed.

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

## Usage Example

Once the stack is running and your repositories are mounted, you can access Zoekt’s web interface (by default at [http://localhost:6070](http://localhost:6070)) to perform code searches.

**Example:**
- Open your browser and go to [http://localhost:6070](http://localhost:6070)
- Enter a search query (e.g., `func main`) to search across all indexed repositories.

For advanced usage and API access, see the documentation in [`docs/`](./docs/).

## Mounting Code & Secret Management

- **Mounting Repositories:**  
  The stack expects your code to be available at `$HOME/repos` on the host. You can change this path in `docker-compose.yml` if needed.

- **Secrets:**  
  - For production, use [Docker secrets](https://docs.docker.com/engine/swarm/secrets/) to manage sensitive information.
  - For local development, you may use a `.env` file in the project root. See `docker-compose.yml` for supported environment variables.

## Further Documentation

See the [`docs/`](./docs/) directory for detailed documentation, architecture diagrams, and advanced configuration options.

---

## License

IndexAgent is released under an open source license. See [`docs/`](./docs/) for details.