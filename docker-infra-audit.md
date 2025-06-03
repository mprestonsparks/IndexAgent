# Repository: `infra`

### 1. Inspected Paths
- docker-compose.yml
- .env
- README.md

### 2. Docker-Compose & Env Config
| File Path           | Snippet (lines X–Y)         | Purpose                              |
|---------------------|-----------------------------|--------------------------------------|
| docker-compose.yml  | 9–10:<br>ports:<br>  - "8080:8080" | Airflow UI: host 8080 → container 8080 |
| docker-compose.yml  | 23–24:<br>ports:<br>  - "8081:8080" | IndexAgent API: host 8081 → container 8080 |
| docker-compose.yml  | 36–37:<br>ports:<br>  - "6070:6070" | Zoekt UI: host 6070 → container 6070 |
| docker-compose.yml  | 45–46:<br>ports:<br>  - "3000:3000" | Sourcebot UI: host 3000 → container 3000 |
| .env                | _No relevant code found_     | No port or platform env vars present |

### 3. Platform Declarations
| File Path           | Snippet                     | Notes                                |
|---------------------|-----------------------------|--------------------------------------|
| docker-compose.yml  | _No relevant code found_    | No `platform:` fields present        |
| .env                | _No relevant code found_    | No `DOCKER_DEFAULT_PLATFORM` present |
| README.md           | _No relevant code found_    | No platform config documented        |

### 4. Port-Check Scripts & Hooks
| File Path           | Snippet                     | Description                          |
|---------------------|-----------------------------|--------------------------------------|
| docker-compose.yml  | _No relevant code found_    | No scripts or hooks present          |
| .env                | _No relevant code found_    | No scripts or hooks present          |
| README.md           | _No relevant code found_    | No scripts or hooks present          |

### 5. Observations
- All port mappings in `docker-compose.yml` are hard-coded and unique; no duplicates.
- No `platform:` fields are set for any service in `docker-compose.yml`.
- No use of `DOCKER_DEFAULT_PLATFORM` in `.env` or elsewhere.
- No port or platform settings are controlled by environment variables.
- No scripts, Makefiles, or CI hooks for port-checking or Docker Compose orchestration are present in this repo.
- All volume mounts use `${HOME}` from `.env`, but ports are not parameterized.
- Documentation in `README.md` matches the hard-coded port assignments in `docker-compose.yml`.

---

# Repository: `airflow-hub`

### 1. Inspected Paths
- docker-compose.yml
- Dockerfile.test
- docker/Dockerfile.airflow
- docker/Dockerfile.airflow-test
- docker/project_specific/Dockerfile.project_analytics
- docker/project_specific/Dockerfile.project_trading
- scripts/ (all files)

### 2. Docker-Compose & Env Config
| File Path         | Snippet (lines X–Y)         | Purpose                                 |
|-------------------|-----------------------------|-----------------------------------------|
| docker-compose.yml | 52–53:<br>```yaml<br>ports:<br>  - "8200:8200"<br>``` | Exposes Vault service on host:8200 to container:8200 |
| docker-compose.yml | 64–65:<br>```yaml<br>ports:<br>  - "8080:8080"<br>``` | Exposes Airflow webserver on host:8080 to container:8080 |
| docker-compose.yml | 67, 85, 105:<br>```yaml<br>env_file:<br>  - ../market-analysis/.env<br>``` | Loads environment variables from external .env file (outside repo) for Airflow services |
| docker-compose.yml | 6–24, 68–73, 86–91, 106–117 | Environment variables for Airflow and Vault, but no port env-var indirection |

### 3. Platform Declarations
| File Path         | Snippet        | Notes                                   |
|-------------------|---------------|-----------------------------------------|
| _No relevant code found_ |               | No `platform:` fields or `DOCKER_DEFAULT_PLATFORM` usage in any Compose, Dockerfile, or script |

### 4. Port-Check Scripts & Hooks
| File Path         | Snippet        | Description                             |
|-------------------|---------------|-----------------------------------------|
| _No relevant code found_ |               | No scripts, Makefiles, or CI hooks reference `docker compose`, `docker-compose`, or port-checking commands like `lsof` |

### 5. Observations
- Only `vault` and `airflow-webserver` services expose ports, both with hard-coded values (`8200`, `8080`).
- No `platform:` fields are set in any Compose service or Dockerfile.
- No use of `DOCKER_DEFAULT_PLATFORM` anywhere in the repo.
- No port-checking or Compose-invoking scripts/hooks found.
- All port mappings are hard-coded; no environment variable indirection for ports.
- No duplicated port or platform settings.
- Environment variables are used for secrets and Airflow config, but not for port or platform configuration.
- The referenced `.env` file for secrets is outside the repo and not auditable here.

---

# Repository: `IndexAgent`

### 1. Inspected Paths
- config/docker-compose.yml
- Makefile
- Dockerfile
- scripts/ (all scripts and subdirectories)
- config/alembic.ini

### 2. Docker-Compose & Env Config
| File Path                 | Snippet (lines X–Y)                             | Purpose                              |
|---------------------------|-------------------------------------------------|--------------------------------------|
| config/docker-compose.yml | 7–9                                             | Host-to-container port mapping for zoekt-indexserver (`"6070:6070"`) |
| config/docker-compose.yml | 18–19                                           | Host-to-container port mapping for sourcebot (`"3000:3000"`)         |
| config/docker-compose.yml | 30–31                                           | Host-to-container port mapping for indexagent (`"8080:8080"`)        |

### 3. Platform Declarations
| File Path                 | Snippet                                        | Notes                                |
|---------------------------|------------------------------------------------|--------------------------------------|
| _No relevant code found_  |                                                | No `platform:` or `DOCKER_DEFAULT_PLATFORM` found in Compose, Makefile, Dockerfile, or scripts |

### 4. Port-Check Scripts & Hooks
| File Path                 | Snippet                                        | Description                          |
|---------------------------|------------------------------------------------|--------------------------------------|
| _No relevant code found_  |                                                | No scripts, Makefile targets, or hooks reference `docker compose`, `docker-compose`, or port-checking commands like `lsof` |

### 5. Observations
- All port mappings in `config/docker-compose.yml` are hard-coded (`6070:6070`, `3000:3000`, `8080:8080`).
- No use of environment variables for port configuration in Compose or Dockerfile.
- No `platform:` fields in Compose services; no `DOCKER_DEFAULT_PLATFORM` usage in Makefile, Dockerfile, or scripts.
- No port-checking or Docker Compose invocation scripts/hooks found.
- No duplicated port settings or platform declarations.
- No inconsistent environment-variable usage for ports or platform.

---

# Repository: `market-analysis`

### 1. Inspected Paths
- docker-compose.yml
- Dockerfile
- docker/Dockerfile.airflow
- .env.example
- scripts/cleanup.py

### 2. Docker-Compose & Env Config
| File Path           | Snippet (lines X–Y)         | Purpose                              |
|---------------------|-----------------------------|--------------------------------------|
| docker-compose.yml  | 6–7:<br>  ports:<br>    - "8000:8000" | Host-to-container port mapping for API service |
| docker-compose.yml  | 9–10:<br>  - API_PORT=8000<br>  - API_HOST=0.0.0.0 | API service environment variables    |
| Dockerfile          | 36–40:<br>ENV API_PORT=8000<br>ENV API_HOST=0.0.0.0<br>EXPOSE 8000 | API port and host config, port exposure |
| .env.example        | 5–6:<br>API_HOST=0.0.0.0<br>API_PORT=8000 | Example environment variables for API port/host |
| .env.example        | 10:<br>REDIS_PORT=6379      | Example environment variable for Redis port    |
| .env.example        | 14:<br>IBKR_PORT=           | Example environment variable for IBKR port    |

### 3. Platform Declarations
| File Path                 | Snippet                                        | Notes                                |
|---------------------------|------------------------------------------------|--------------------------------------|
| _No relevant code found_  |                                                | No `platform:` or `DOCKER_DEFAULT_PLATFORM` found in any inspected file |

### 4. Port-Check Scripts & Hooks
| File Path                 | Snippet                                        | Description                          |
|---------------------------|------------------------------------------------|--------------------------------------|
| _No relevant code found_  |                                                | No scripts, Makefiles, or hooks reference Docker Compose or port-checking commands |

### 5. Observations
- The only host-to-container port mapping is `"8000:8000"` for the API service in `docker-compose.yml`.
- The API port is hard-coded as 8000 in multiple places: `docker-compose.yml`, `Dockerfile`, and `.env.example`.
- No `platform:` field is set in `docker-compose.yml` or Dockerfiles.
- No usage of `DOCKER_DEFAULT_PLATFORM` or platform-related environment variables.
- No scripts or hooks automate Docker Compose or port-checking.
- Environment variable usage for ports is consistent, but the port is also hard-coded in Compose and Dockerfile.
- No duplicated port settings, but port values are repeated across files.

---

# Review Required Code

> ## infra
> - [`docker-compose.yml`](../infra/docker-compose.yml:4–18): Add `platform:` field to `airflow` service.
> - [`docker-compose.yml`](../infra/docker-compose.yml:19–32): Add `platform:` field to `indexagent` service.
> - [`docker-compose.yml`](../infra/docker-compose.yml:33–41): Add `platform:` field to `zoekt-indexserver` service.
> - [`docker-compose.yml`](../infra/docker-compose.yml:42–49): Add `platform:` field to `sourcebot` service.
>
> ## airflow-hub
> - [`docker-compose.yml`](../airflow-hub/docker-compose.yml:52–53): Hard-coded port mapping for Vault service.
> - [`docker-compose.yml`](../airflow-hub/docker-compose.yml:64–65): Hard-coded port mapping for Airflow webserver.
> - [`docker-compose.yml`](../airflow-hub/docker-compose.yml:all services): Add `platform:` fields as needed for multi-platform support.
>
> ## IndexAgent
> - [`config/docker-compose.yml`](config/docker-compose.yml:7-9): Hard-coded port mapping for zoekt-indexserver.
> - [`config/docker-compose.yml`](config/docker-compose.yml:18-19): Hard-coded port mapping for sourcebot.
> - [`config/docker-compose.yml`](config/docker-compose.yml:30-31): Hard-coded port mapping for indexagent.
> - [`config/docker-compose.yml`](config/docker-compose.yml:all services): Add `platform:` fields as needed for multi-platform support.
>
> ## market-analysis
> - [`docker-compose.yml`](../market-analysis/docker-compose.yml:6–7): Hard-coded port mapping for API service.
> - [`docker-compose.yml`](../market-analysis/docker-compose.yml:all services): Add `platform:` field for multi-platform support.
> - [`Dockerfile`](../market-analysis/Dockerfile:36–40): API port is hard-coded and exposed.
> - [`.env.example`](../market-analysis/.env.example:5–6): API port and host are set as environment variables, but not referenced dynamically in Compose or Dockerfile.