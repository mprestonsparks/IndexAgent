# IndexAgent Overview

## Introduction

**IndexAgent** is a Free and Open Source Software (FOSS) stack for self-hosted, large-scale code search and indexing. It is designed to give organizations and individuals full control over their code search infrastructure, prioritizing privacy, extensibility, and developer autonomy. With IndexAgent, you can run your own code search platform—free from vendor lock-in, cloud dependencies, and external orchestration frameworks.

## What Does IndexAgent Do?

IndexAgent enables you to search and index source code across multiple repositories at scale. All components are self-hosted, ensuring your code and metadata remain under your control. The platform is built for extensibility, allowing you to integrate additional source types and customize indexing strategies as your needs evolve.

## Architecture and Main Components

The IndexAgent stack is composed of modular components that work together to provide a seamless code search experience:

### 1. Sourcebot

- **Role:** Handles source code discovery, repository synchronization, and metadata extraction.
- **Functionality:** Keeps your codebase up-to-date within the IndexAgent stack by regularly syncing repositories and extracting relevant metadata for indexing.

### 2. Zoekt

- **Role:** High-performance code search engine.
- **Functionality:** Indexes all mounted repositories and provides fast, scalable search capabilities across your entire codebase. Zoekt powers the web interface and API for code search queries.

### 3. Orchestration & Integration

- **Philosophy:** IndexAgent intentionally avoids complex orchestration frameworks (such as Airflow) within this repository, focusing on simplicity and transparency.
- **Integration:** Components communicate via well-defined interfaces, and the stack is managed using Docker Compose for ease of deployment and maintenance.

## Workflow

The typical workflow for using IndexAgent is as follows:

1. **Repository Mounting:**  
   - You make your code repositories available to the stack (by default at `$HOME/repos` on the host system).
   - The path can be customized in `docker-compose.yml`.

2. **Source Discovery & Sync:**  
   - Sourcebot detects and syncs repositories, ensuring the latest code and metadata are available for indexing.

3. **Indexing:**  
   - Zoekt indexes the synchronized repositories, building efficient search indices.

4. **Code Search:**  
   - Users access Zoekt’s web interface (default: [http://localhost:6070](http://localhost:6070)) to perform fast, full-text code searches across all indexed repositories.
   - Advanced users and developers can interact with the search API for automation or integration with other tools.

## Extensibility

- **Pluggable Integrations:** IndexAgent is designed to support additional source integrations and custom indexing strategies.
- **Open Source:** Contributions are welcome, and the project encourages community-driven improvements.

## Who Is This For?

- **Developers** who need fast, private, large-scale code search.
- **Organizations** seeking to maintain control over their code search infrastructure.
- **Anyone** who values privacy, autonomy, and extensibility in their development workflows.

## Further Reading

- For setup instructions, usage examples, and advanced configuration, see the [README.md](../README.md) and other documents in the [`docs/`](./) directory.
- For details on mounting code and managing secrets, refer to the relevant sections in the main README.

---