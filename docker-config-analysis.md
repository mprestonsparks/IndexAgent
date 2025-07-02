# Docker and Configuration Infrastructure Analysis

## Executive Summary

This analysis examines the Docker setup and configuration management across three repositories in the gitRepos directory: infra, airflow-hub, and IndexAgent. The infrastructure uses Docker Compose for orchestration, with the infra repository serving as the central orchestration point that references services from the other repositories.

## 1. Docker Infrastructure Analysis

### 1.1 Repository Structure and Relationships

#### **infra Repository** (Central Orchestrator)
- **Location**: `../infra/docker-compose.yml`
- **Role**: Main orchestration point for all services
- **Key Files**:
  - `docker-compose.yml`: Central compose file
  - `.env.example`: Environment variable template
  - `Makefile`: Build and deployment automation
  - `scripts/check-ports.sh`: Port availability checking

#### **airflow-hub Repository** (Workflow Management)
- **Location**: `../airflow-hub/`
- **Role**: Apache Airflow services and DAG management
- **Key Files**:
  - `docker-compose.yml`: Airflow-specific services
  - `docker/Dockerfile.airflow`: Custom Airflow image
  - `docker/Dockerfile.airflow-test`: Test environment image
  - Multiple project-specific Dockerfiles

#### **IndexAgent Repository** (Agent Services)
- **Location**: `./` (current directory)
- **Role**: Index agent services
- **Key Files**:
  - `Dockerfile`: Minimal Python 3.11 based image
  - `config/docker-compose.yml`: Alternative compose configuration

### 1.2 Docker Compose Architecture

#### Central Orchestration (infra/docker-compose.yml)
```yaml
version: "3.8"

services:
  airflow-service:
    user: "${AIRFLOW_UID:-50000}:0"
    build:
      context: ../airflow-hub
      dockerfile: docker/Dockerfile.airflow
    container_name: airflow-service
    platform: "${DOCKER_DEFAULT_PLATFORM}"
    ports:
      - "${AIRFLOW_PORT}:8080"
    env_file:
      - .env
    volumes:
      - ../airflow-hub/dags:/opt/airflow/dags
      - ../airflow-hub/plugins:/opt/airflow/plugins
      - ../airflow-hub/logs:/opt/airflow/logs
    restart: unless-stopped

  indexagent:
    build:
      context: ../IndexAgent
    container_name: indexagent
    platform: "${DOCKER_DEFAULT_PLATFORM}"
    ports:
      - "${INDEXAGENT_API_PORT}:8080"
    env_file:
      - .env
    volumes:
      - /Users/preston/repos:/repos
      - ../IndexAgent/reports:/app/reports
      - ../IndexAgent/docs/auto:/app/docs/auto
    restart: unless-stopped

  zoekt-indexserver:
    image: sourcegraph/zoekt-indexserver:latest
    container_name: zoekt-indexserver
    platform: "${DOCKER_DEFAULT_PLATFORM}"
    ports:
      - "${ZOEKT_UI_PORT}:6070"
    volumes:
      - /Users/preston/repos:/repos
    restart: unless-stopped

  sourcebot:
    image: ghcr.io/sourcebot-dev/sourcebot:latest
    container_name: sourcebot
    platform: "${DOCKER_DEFAULT_PLATFORM}"
    ports:
      - "${SOURCEBOT_UI_PORT}:3000"
    volumes:
      - /Users/preston/repos:/repos
    restart: unless-stopped
```

### 1.3 Service Definitions and Naming

#### Service Naming Convention
- **Pattern**: Lowercase with hyphens for multi-word names
- **Examples**:
  - `airflow-service`
  - `indexagent`
  - `zoekt-indexserver`
  - `sourcebot`
  - `airflow-webserver`
  - `airflow-scheduler`
  - `postgres`
  - `vault`

#### Container Naming
- Explicit container names are set for most services
- Matches service names for consistency

### 1.4 Base Images Analysis

#### Common Base Images Used:
1. **Apache Airflow**: `apache/airflow:3.0.0`
2. **Python Services**: `python:3.11-slim`
3. **PostgreSQL**: Built from context (custom image)
4. **HashiCorp Vault**: `hashicorp/vault:1.12.0` (referenced in docs)
5. **External Services**:
   - `sourcegraph/zoekt-indexserver:latest`
   - `ghcr.io/sourcebot-dev/sourcebot:latest`

### 1.5 Inter-Service Communication

#### Network Configuration
- Services communicate using Docker's default bridge network
- Service names are used as hostnames for internal communication
- Examples:
  - Airflow connects to PostgreSQL at `postgres:5432`
  - Services connect to Vault at `vault:8200`
  - IndexAgent endpoint: `http://indexagent-container/cleanup-todos`

#### Port Mappings
All ports are configurable via environment variables:
- `AIRFLOW_PORT`: Airflow web UI
- `INDEXAGENT_API_PORT`: IndexAgent API
- `ZOEKT_UI_PORT`: Zoekt search UI
- `SOURCEBOT_UI_PORT`: Sourcebot UI
- `VAULT_PORT`: HashiCorp Vault

## 2. Configuration Management Analysis

### 2.1 Configuration File Formats

#### Primary Formats Used:
1. **YAML**: Docker Compose files, Airflow DAG configurations
2. **JSON**: Not prominently used in infrastructure
3. **Python**: Airflow configuration, hooks, operators
4. **Environment Files**: `.env` files for runtime configuration

### 2.2 Environment Variable Management

#### Pattern: Centralized .env File
Location: `../infra/.env` (referenced but not committed)

Template (`../infra/.env.example`):
```
AIRFLOW_PORT=
VAULT_PORT=
INDEXAGENT_API_PORT=
ZOEKT_UI_PORT=
SOURCEBOT_UI_PORT=
MARKET_API_PORT=
DOCKER_DEFAULT_PLATFORM=
```

#### Environment Variable Propagation:
1. Services use `env_file: - .env` directive
2. Build-time args passed via `ARG` in Dockerfiles
3. Runtime environment variables set in compose files

### 2.3 Configuration Storage Patterns

#### Airflow Configuration:
- **Location**: `../airflow-hub/airflow.cfg`
- **Environment Overrides**: `AIRFLOW__SECTION__KEY` pattern
- **Example**: `AIRFLOW__CORE__EXECUTOR: LocalExecutor`

#### Project-Specific Configurations:
- Stored in respective project directories
- Mounted as volumes into containers
- Example: `../airflow-hub/config.yaml`

### 2.4 Sensitive Configuration Management

#### Current State:
1. **Development**: Using `.env` files and hardcoded dev tokens
2. **Secrets in Compose**: Some sensitive data in docker-compose.yml
3. **External .env References**: `../../market-analysis/.env`

#### Planned Architecture (from secrets_management_architecture.md):
- **HashiCorp Vault Integration**: Centralized secrets management
- **Path Structure**:
  ```
  secret/
  ├── airflow/
  │   ├── connections/
  │   └── variables/
  └── projects/
      ├── trading/
      ├── analytics/
      └── market_analysis/
  ```

### 2.5 Configuration Inheritance and Overrides

#### Docker Compose Override Pattern:
```yaml
x-airflow-common:
  &airflow-common
  environment:
    &airflow-common-env
    # Base configuration

services:
  airflow-webserver:
    <<: *airflow-common
    environment:
      <<: *airflow-common-env
      # Service-specific overrides
```

## 3. Environment Variables and Secrets

### 3.1 Environment Variable Categories

#### Infrastructure Variables:
- `DOCKER_DEFAULT_PLATFORM`: Platform specification
- `AIRFLOW_UID`: User ID for Airflow containers
- Port configurations (`*_PORT` variables)

#### Application Variables:
- `AIRFLOW_VAR_*`: Airflow-specific variables
- `BINANCE_API_KEY`, `BINANCE_SECRET_KEY`: External API credentials
- `CLAUDE_MODEL`: AI model configuration

### 3.2 Secret Management Patterns

#### Current Implementation:
1. **Docker Secrets** (IndexAgent):
   ```yaml
   secrets:
     claude_api_key:
       file: /Users/preston/.secrets/claude_api_key.secret
   ```

2. **Environment Variables**: Direct injection via env_file

3. **Vault Backend Configuration** (Airflow):
   ```yaml
   AIRFLOW__SECRETS__BACKEND: "airflow.providers.hashicorp.secrets.vault.VaultBackend"
   AIRFLOW__SECRETS__BACKEND_KWARGS: '{"connections_path": "airflow/connections", ...}'
   ```

### 3.3 Configuration Management Tools

#### Currently in Use:
1. **Docker Compose**: Primary orchestration
2. **Make**: Build automation (`../infra/Makefile`)
3. **Shell Scripts**: Port checking, initialization

#### Planned/Documented:
1. **HashiCorp Vault**: Centralized secrets management
2. **Airflow Connections**: Database-backed connection management

## 4. Recommendations for Agent-Evolution Integration

### 4.1 Docker Compose Modifications

#### Add New Service to infra/docker-compose.yml:
```yaml
services:
  agent-evolution:
    build:
      context: ../agent-evolution
      dockerfile: Dockerfile
    container_name: agent-evolution
    platform: "${DOCKER_DEFAULT_PLATFORM}"
    ports:
      - "${AGENT_EVOLUTION_PORT}:8080"
    env_file:
      - .env
    volumes:
      - ../agent-evolution/data:/app/data
      - ../agent-evolution/logs:/app/logs
    depends_on:
      - indexagent
      - airflow-service
    restart: unless-stopped
```

### 4.2 Environment Variable Additions

Add to `../infra/.env.example`:
```
AGENT_EVOLUTION_PORT=8090
AGENT_EVOLUTION_API_KEY=
AGENT_EVOLUTION_LOG_LEVEL=INFO
```

### 4.3 Configuration File Structure

Recommended structure for agent-evolution:
```
agent-evolution/
├── Dockerfile
├── config/
│   ├── default.yaml
│   ├── development.yaml
│   └── production.yaml
├── docker-compose.override.yml  # For local development
└── .env.example
```

### 4.4 Integration with Existing Services

1. **Airflow Integration**:
   - Create DAGs for agent-evolution workflows
   - Use Airflow connections for API credentials

2. **IndexAgent Communication**:
   - Use service names for internal communication
   - Example: `http://indexagent:8080/api/endpoint`

3. **Shared Volumes**:
   - Mount shared data directories if needed
   - Use consistent path patterns

### 4.5 Security Considerations

1. **Secrets Management**:
   - Integrate with planned Vault implementation
   - Use Docker secrets for sensitive files
   - Avoid hardcoding credentials

2. **Network Isolation**:
   - Consider custom Docker networks for service groups
   - Implement proper firewall rules

3. **Access Control**:
   - Use proper file permissions in volumes
   - Implement API authentication between services

## 5. Key Findings and Patterns

### 5.1 Strengths
1. **Centralized Orchestration**: infra repository serves as single entry point
2. **Consistent Naming**: Clear service and container naming conventions
3. **Flexible Configuration**: Environment variables for all key settings
4. **Service Modularity**: Each repository maintains its own Dockerfile

### 5.2 Areas for Improvement
1. **Secrets Management**: Currently relies on .env files, Vault integration planned but not implemented
2. **Hard-coded Paths**: Some absolute paths (e.g., `/Users/preston/repos`)
3. **Missing Production Config**: Development-focused, production configs need work
4. **Documentation**: Configuration patterns not consistently documented

### 5.3 Configuration Patterns Discovered
1. **Build Context References**: Services built from sibling directories
2. **Volume Mounting**: Consistent patterns for logs, data, and code
3. **Port Management**: All ports externalized to environment variables
4. **Service Dependencies**: Proper health checks and dependency declarations

## Conclusion

The Docker and configuration infrastructure follows a hub-and-spoke model with the infra repository at the center. While the current setup is functional for development, the planned HashiCorp Vault integration will significantly improve security and secrets management. New services like agent-evolution can be easily integrated by following the established patterns for service definition, configuration management, and inter-service communication.