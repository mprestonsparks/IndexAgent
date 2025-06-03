# Help Request: Resolving Persistent Docker Infrastructure Issues

I have encountered persistent issues while trying to bring up the full Docker infrastructure stack. Despite applying the recommended edits and trying various solutions, I am still unable to get all containers running correctly.

## Problem Summary

The `docker compose up` command consistently fails with errors related to the `build` configuration or missing directories. The specific error messages vary, but they generally indicate that Docker Compose is unable to resolve the build context or access required files.

## Steps Taken

I have taken the following steps to try to resolve the issue:

1. Verified that all required files (docker-compose.yml, Dockerfiles, .env) exist in the correct locations.
2. Corrected the `build` contexts and Dockerfile paths in the docker-compose.yml files.
3. Specified absolute paths to the Dockerfiles.
4. Tried running the `docker compose up` command with an explicit project directory (`--project-directory .`).
5. Inspected the interpreted Docker Compose configuration using `docker compose config`.
6. Replaced `${HOME}` with the explicit path `/Users/preston` in the `volumes` for the specified services.

## Specific Questions and Answers

### Question 1: Why does `docker compose up` error-out with “build context … not found” and how should the build section be written?

**Expected Answer Structure:**

```
services:
  <service-name>:
    build:
      context: <absolute-or-relative-path>
      dockerfile: <path/from/context/Dockerfile>
```

**Answer:**

> In every repo be sure the `context:` points at the *directory containing the Dockerfile* (or its parent) and that `dockerfile:` is **relative to that context**.
> Example for airflow-hub, assuming repo layout
> `airflow-hub/docker/Dockerfile.airflow`
>
> ```
> services:
>   airflow-webserver:
>     build:
>       context: ./docker
>       dockerfile: Dockerfile.airflow
> ```
>
> Common mistakes:
>
> * Context set to `.` in infra when the Dockerfile lives in another repo.
> * `dockerfile:` given an absolute path (Compose rejects this).
>   Use `docker compose config | grep -A3 build` to check the resolved path.

---

### Question 2: Which environment variables can break the build context and how should they be set?

**Expected Answer Structure:**

1. `<VAR_NAME>` – *reason & correct value*
2. …

**Answer:**

1. **COMPOSE\_PROJECT\_NAME** – If overridden it changes volume/ network names and can make cached images look missing; set to a short, unique string or leave unset.
2. **DOCKER\_DEFAULT\_PLATFORM** – Must match the `platform:` you declared (`linux/arm64` vs `linux/amd64`). Export it in `.env` *and* your shell:

   ```bash
   export DOCKER_DEFAULT_PLATFORM=linux/arm64
   ```
3. **AIRFLOW\_UID** – Needed only if you set `user: "${AIRFLOW_UID}:0"` in the Airflow service. Export once:

   ```bash
   export AIRFLOW_UID=$(id -u)
   ```
4. **HOME** – If `${HOME}` appears in volume paths but resolves differently inside Compose (rare on macOS). Prefer the literal path or `${USER_HOME}` you set yourself.

---

### Question 3: What is an alternative way to bring up the multi-repo stack if Compose still complains?

**Expected Answer Structure:**
*Step-by-step list.*

**Answer:**

1. **Pre-build each image** from its own repo to avoid cross-context issues:

   ```bash
   cd airflow-hub && docker build -t local/airflow-web:dev -f docker/Dockerfile.airflow docker
   cd ../IndexAgent    && docker build -t local/indexagent:dev      .
   cd ../market-analysis && docker build -t local/market-api:dev    .
   ```
2. **Replace the build: section** in every compose file with `image: local/<name>:dev`.
3. From the **infra** repo run the single compose command:

   ```bash
   docker compose \
     -f docker-compose.yml \
     -f ../airflow-hub/docker-compose.yml \
     -f ../IndexAgent/config/docker-compose.yml \
     -f ../market-analysis/docker-compose.yml \
     up -d
   ```
4. This isolates build problems (they’ll fail fast during `docker build`) and Compose only has to run containers.

---

### Question 4: Why does the port checker still report “Port 8080 in use” after I set `AIRFLOW_PORT=8082`, and how do I fix it?

**Expected Answer Structure:**
*Bullet list of possible causes with one-line remedies.*

**Answer:**

* **Old container still bound to 8080** – run `docker ps | grep 8080` and `docker stop <id>`
* **.env not re-sourced** – after editing `.env`, execute `source .env` *in the same shell* before running the port checker.
* **Variable shadowed** – ensure you didn’t also export `AIRFLOW_PORT=8080` in your shell startup files (`echo $AIRFLOW_PORT`).
* **Compose override file hard-codes 8080** – search: `grep -R "8080:8080" infra airflow-hub` and replace with `${AIRFLOW_PORT}:8080`.
* **Health-check hitting 8080 internally** – internal container port stays 8080; only host side changes. That doesn’t affect the port checker, but make sure you changed the *host* port, not container port.

---

#### Try this quick diagnostic bundle

```bash
# stop anything still mapped to 8080
docker ps --filter "publish=8080" -q | xargs -r docker stop

# verify no listener
lsof -i :8080 || echo "Port 8080 free"

# ensure env is current
grep AIRFLOW_PORT infra/.env
echo "Runtime AIRFLOW_PORT=$AIRFLOW_PORT"
```

If `AIRFLOW_PORT` shows 8082 everywhere and `lsof` is silent, rerun the stack—8080 conflicts should be gone.

---

*Once you’ve applied these answers, re-run `docker compose config` to validate paths, rebuild, and start the stack. If problems persist, paste the exact new error text so we can iterate.*