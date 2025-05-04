# RooCode Implementation Instructions for **IndexAgent**

---

## 0  Project Context and Overview

You are implementing **IndexAgent**—a fully FOSS, self‑hosted platform that:

1. **Indexes & searches** every project repository in < 50 ms via *Zoekt* (back‑end) and *Sourcebot* (UI).
2. **Automates code hygiene**—TODO removal, coverage boosts, documentation refresh—using a local **Claude Code CLI** REPL agent.
3. **Orchestrates maintenance tasks** through **Apache Airflow** DAGs that live in the existing **`airflow-hub`** repository.

---

## 1  Repository Purposes & Top‑Level Workflow

| Repository                                                                            | Purpose                                                                                                                                                        | New / Existing |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| **`airflow-hub`**                                                                     | *Central repo* for **all Airflow DAGs**—including the new code‑maintenance, coverage, and doc‑refresh DAGs                                                     | **Existing**   |
| `market-analysis`, `trade-manager`, `trade-discovery`, `trade-dashboard`, `git-books` | Application/service repos that will be indexed and auto‑maintained                                                                                             | Existing       |
| **`IndexAgent`**                                                                      | Contains **Docker‑compose**, Zoekt + Sourcebot config, setup scripts, Claude wrapper scripts, Makefile, and infrastructure docs. **No DAG code** resides here. | **New**        |

### How the repos interact

```
              ┌────────────────────┐
              │  IndexAgent repo   │
              │  (compose/scripts) │
              └───────┬────────────┘
                      │ mounts $HOME/repos
  index/search via    │
      Zoekt API       ▼
 ┌─────────────┐  Reads/Writes  ┌────────────────┐
 │ Sourcebot & │<──────────────>│  Application   │
 │  Zoekt      │                │   Repos        │
 └──────┬──────┘                └────────────────┘
        │ REST results
        ▼
   Claude CLI wrappers
        └───────────────┐
   Airflow DAG calls    │
          (lives in     ▼
       airflow-hub) ┌──────────┐  Commits/PRs  ┌────────────┐
                    │ airflow  │──────────────▶│  Git host  │
                    └──────────┘               └────────────┘
```

1. **IndexAgent** stack runs Sourcebot + Zoekt containers and Claude wrapper scripts.
2. Airflow (deployed from `airflow-hub`) calls the wrapper scripts through network‑visible endpoints or SSH exec.
3. Scripts read/write application repos mounted into the IndexAgent containers and push branches/PRs.

---

## 2  Background Knowledge Transfer

### Current system

* Application repos listed above, cloned locally under `$HOME/repos`.
* `airflow-hub` already bootstrapped (minimal DAGs).
* Search & maintenance tasks are manual.

### Domain snapshot

```
Repo ─┬─> File ──> TODO
      ├─> TestFile
      ├─> DocPage
      └─> ModuleMeta (coverage, doc-status)

ZoektIndex <─ refresh ─ Airflow DAG (airflow-hub)
ClaudeAgent ── acts‑on ─ SearchResult
Commit/PR ──> Git platform
```

---

## 3  Development Environment

| Item              | Setting                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| IDE               | Windsurf (VS Code) + RooCode                                                                      |
| OS                | Windows 11 (WSL 2) & macOS 14 (M1 Pro)                                                            |
| Date baseline     | **May 2 2025**                                                                                    |
| Languages         | Python 3.11, Bash, Make                                                                           |
| Container runtime | Docker 25+, docker‑compose v2                                                                     |
| Airflow install   | Managed inside `airflow-hub` (LocalExecutor)                                                      |
| Repo layout       | All application repos cloned into `$HOME/repos` which is mount‑shared with IndexAgent containers. |

---

## 4  Absolute Requirements (Hard Constraints)

1. **Tech stack**

   * Zoekt `main` (Apache‑2.0)
   * Sourcebot `ghcr.io/sourcebot-dev/sourcebot:latest` (MIT)
   * Claude Code CLI `@anthropic-ai/claude-cli` beta
   * Apache Airflow 3.0 (in `airflow-hub`)
   * pytest 8 + pytest‑cov

2. **Licensing** — All runtime components FOSS & self‑hosted.

3. **Security** — Agent runs with `--dir $(pwd)` & no external network; Sourcebot UI behind Caddy basic‑auth.

4. **Performance** — Search latency < 100 ms for 2 M files; nightly index < 15 min.

5. **Coverage gate** — Overall coverage must stay ≥ 80 %.

---

## 5  Incremental Implementation Road‑map

### Phase 1  Foundation MVP (IndexAgent)

*Goal*: One‑command stack up, search verified.

Tasks (all inside **`IndexAgent`**)

1. Create repo skeleton:

   ```
   IndexAgent/
     docker-compose.yml
     Makefile
     scripts/
     docs/
   ```
2. `docker-compose.yml` runs `sourcebot` + `zoekt-indexer`.
3. `Makefile` targets: `make up | down | index | status`.
4. Provide README for setup.

*DAG or Airflow changes occur **only** in `airflow-hub`, but Phase 1 needs none yet.*

### Phase 2  Automated TODO Cleanup

*Goal*: Nightly Claude‑driven TODO fixes via Airflow.

Tasks

1. Add `scripts/agent_fix_todos.sh` (IndexAgent).
2. In **`airflow-hub`**, create DAG `nightly_todo_cleanup.py` that SSH‑execs wrapper or hits exposed endpoint on IndexAgent container.
3. Test branch creation & PR flow.

### Phase 3  AI Test & Coverage Loop

*Goal*: Coverage ≥ 80 %.

Tasks

1. Add `scripts/run_cov.py` & `scripts/ai_test_loop.sh` (IndexAgent).
2. DAG `nightly_test_improve.py` committed to **`airflow-hub`**.
3. CI rule in each app repo verifying gate.

### Phase 4  Doc Refresh (detailed below)

All wrapper scripts in **IndexAgent**; DAG resides in **`airflow-hub`**.

---

## 6  IndexAgent Repository (contents)

```
IndexAgent/
├ docker-compose.yml
├ caddy/Caddyfile
├ Makefile
├ scripts/
│   ├ agent_fix_todos.sh
│   ├ run_cov.py
│   ├ ai_test_loop.sh
│   ├ find_undocumented.py
│   └ agent_write_docs.sh
└ docs/
    └ adr/
```

*No Airflow code here; DAG python files stay in `airflow-hub/dags/`.*

---

## 7  Claude Code CLI Integration (Expanded)

### 7.1  Setup

```bash
npm i -g @anthropic-ai/claude-cli
export CLAUDE_API_KEY=sk-ant-xxx
export CLAUDE_MODEL=claude-3-opus-2025-05-02
```

### 7.2  Wrapper script example (`scripts/agent_fix_todos.sh`)

```bash
#!/usr/bin/env bash
# Fixes TODOs found by Zoekt via Sourcebot API

set -euo pipefail
API="http://localhost:3000/api/internal/search"
tmp=$(mktemp)

curl -s "$API?q=TODO&num=1000&case=yes" >"$tmp" || {
  echo "Search API unreachable" >&2; exit 1; }

jq -c '.Results[]' "$tmp" | while read -r hit; do
  FILE=$(jq -r '.File' <<<"$hit")
  PREVIEW=$(jq -r '.Preview' <<<"$hit")

  claude -m "Resolve the TODO below while preserving functionality:
File: $FILE
Snippet:
$PREVIEW
Return only the patch." \
  --file "$FILE" --apply --dir "$(git rev-parse --show-toplevel)" \
  || echo "CLAUDE_ERROR|$FILE" >> .ai_errors.log
done
```

**Error handling patterns**

* 429/5xx → retry 4× exponentially.
* Patch apply fail → diff saved to `.ai_failed_patches/`.
* Log `CLAUDE_ERROR|file` lines to allow DAG to aggregate failures.

---

## 8  Enhanced Error Recovery Matrix

| Scenario            | Auto‑fix                                                             | Human Step                            |
| ------------------- | -------------------------------------------------------------------- | ------------------------------------- |
| Zoekt shard missing | Remove shard, full re‑index (`make index`)                           | If >2 nights fail, run fsck, open ADR |
| Claude rate limit   | Back‑off 30 s × 4; queue job                                         | Rotate key or reschedule              |
| Test flake          | Retry 3×; tag flaky in pytest‑mark                                   | Refactor flaky test next sprint       |
| Git merge conflict  | Rebase branch, rerun agent; if still conflict label `[needs-review]` | Manual merge                          |
| Doc build fail      | Run `markdownlint --fix`; regenerate                                 | Assign to docs maintainer             |

---

## 9  Phase 4  Doc Refresh (Complete Plan)

| Objective | Generate markdown for undocumented modules |

### Tasks

1. **Undocumented scan** (`scripts/find_undocumented.py`, IndexAgent)

   * Use `ast` to inspect Python packages; flag modules with missing or very short docstrings.
   * Output JSON list `undoc.json`.

2. **Doc generation** (`scripts/agent_write_docs.sh`, IndexAgent)

   ```bash
   claude -m "Write a markdown page for module {{module}}: overview, API table, usage examples." \
          --outfile "docs/auto/{{module}}.md"
   markdownlint -f "docs/auto/{{module}}.md"
   ```

3. **DAG** (`nightly_doc_refresh.py`, in `airflow-hub`)

   ```
   reindex -> scan_undoc -> generate_docs
   ```

4. **Metrics**

   * Maintain Prometheus gauge `docs_coverage`.
   * Success when ≥ 90 % or nightly delta ≥ 5 newly documented modules until saturated.

### Success Criteria

* [ ] Markdown lints clean.
* [ ] `docs_coverage` trending positive.

---

## 10  ADR Template

```
# ADR-[NNN]: [Short decision title]

## Status
Proposed | Accepted | Superseded | Deprecated

## Context
Why we needed a decision.

## Decision
What we chose and why.

## Alternatives Considered
* Option A
* Option B
* Do nothing

## Consequences
### Positive
* ...

### Negative
* ...

## References
Links, issues, benchmarks.
```

---

## 11  Phase Transition Checklist

Before flagging a phase **complete**, RooCode must confirm:

1. **IndexAgent** stack (`make up`) healthy.
2. Corresponding **Airflow DAG** in `airflow-hub` runs green.
3. Search latency sampled (< 100 ms).
4. Coverage ≥ 80 % (Phase 3+).
5. Docs coverage metric updated (Phase 4).
6. CI green across affected repos.
7. ADRs added for new design choices.
8. README & CHANGELOG updated in each touched repo.
9. `docs/deviations.md` updated (if any).

---

## 12  Self‑Verification Questions

*(unchanged – see previous version section 10)*

---

**Phase 1 Implementation Complete. Awaiting feedback before proceeding to Phase 2.**
