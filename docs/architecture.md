# Parallel Processing System Architecture

## Overview

The Parallel Processing System is designed to execute maintenance tasks across multiple Git worktrees in parallel, leveraging Apache Airflow for orchestration and SQLite for state management. The system maximizes throughput while preventing resource contention and maintaining system stability.

## System Components

### Core Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Apache Airflow                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Parallel Maintenance DAG                        │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │   │
│  │  │ Task 1  │  │ Task 2  │  │ Task 3  │  │ Task N  │       │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │   │
│  └───────┼────────────┼────────────┼────────────┼─────────────┘   │
└──────────┼────────────┼────────────┼────────────┼─────────────────┘
           │            │            │            │
           ▼            ▼            ▼            ▼
    ┌──────────────────────────────────────────────────┐
    │              ShardCoordinator                     │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
    │  │   Shard    │  │   Shard    │  │   Shard    │ │
    │  │ Assignment │  │  Tracking  │  │  Cleanup   │ │
    │  └────────────┘  └────────────┘  └────────────┘ │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │              WorktreeManager                      │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
    │  │   Create   │  │   Manage   │  │   Remove   │ │
    │  │  Worktree  │  │  Worktree  │  │  Worktree  │ │
    │  └────────────┘  └────────────┘  └────────────┘ │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │              ClaudeRunner                         │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
    │  │    TODO    │  │    Code    │  │   Report   │ │
    │  │  Scanner   │  │  Analyzer  │  │ Generator  │ │
    │  └────────────┘  └────────────┘  └────────────┘ │
    └──────────────────────────────────────────────────┘
```

### Component Descriptions

#### 1. **Apache Airflow**
- **Role**: Orchestration engine
- **Responsibilities**:
  - DAG scheduling and execution
  - Task dependency management
  - Resource allocation
  - Monitoring and alerting
- **Key Features**:
  - Parallel task execution
  - Retry logic
  - Task state management
  - Web UI for monitoring

#### 2. **Parallel Maintenance DAG**
- **Role**: Workflow definition
- **Responsibilities**:
  - Define task dependencies
  - Set parallelism limits
  - Configure retry policies
  - Handle task failures
- **Configuration**:
  ```python
  max_active_tasks = 10  # Parallel execution limit
  retries = 2           # Retry failed tasks
  retry_delay = 300     # 5 minutes between retries
  ```

#### 3. **ShardCoordinator**
- **Role**: Task distribution and tracking
- **Responsibilities**:
  - Assign shards to workers
  - Track shard execution status
  - Prevent duplicate processing
  - Handle failed shards
- **Key Methods**:
  - `assign_shard()`: Atomically assign available shard
  - `update_status()`: Update shard execution status
  - `cleanup_stale()`: Remove abandoned shards

#### 4. **WorktreeManager**
- **Role**: Git worktree lifecycle management
- **Responsibilities**:
  - Create isolated worktrees
  - Manage worktree paths
  - Clean up after task completion
  - Handle worktree conflicts
- **Key Methods**:
  - `create_worktree()`: Create new worktree
  - `remove_worktree()`: Safe worktree removal
  - `cleanup_all()`: Bulk cleanup operation

#### 5. **ClaudeRunner**
- **Role**: Task execution engine
- **Responsibilities**:
  - Execute maintenance tasks
  - Process TODO comments
  - Generate reports
  - Handle task-specific logic
- **Integration Points**:
  - Reads from worktree
  - Writes results to database
  - Communicates with external APIs

## Data Flow

```
1. DAG Trigger
   │
   ▼
2. Task Creation (N parallel tasks)
   │
   ▼
3. Shard Assignment (ShardCoordinator)
   │
   ├─► Check available shards
   ├─► Atomically assign shard
   └─► Update status to "running"
   │
   ▼
4. Worktree Creation (WorktreeManager)
   │
   ├─► Generate unique path
   ├─► Create Git worktree
   └─► Return worktree path
   │
   ▼
5. Task Execution (ClaudeRunner)
   │
   ├─► Process assigned shard
   ├─► Execute maintenance logic
   └─► Generate results
   │
   ▼
6. Status Update (ShardCoordinator)
   │
   ├─► Mark shard complete/failed
   ├─► Record execution time
   └─► Log any errors
   │
   ▼
7. Cleanup (WorktreeManager)
   │
   ├─► Remove worktree
   └─► Free resources
```

## Database Schema

### shard_status Table

```sql
CREATE TABLE shard_status (
    shard_id TEXT PRIMARY KEY,      -- Unique identifier for shard
    status TEXT NOT NULL,           -- pending|running|completed|failed
    started_at TIMESTAMP,           -- Task start time
    completed_at TIMESTAMP,         -- Task completion time
    updated_at TIMESTAMP            -- Last update time
        DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,             -- Error details if failed
    worker_id TEXT,                 -- Airflow worker identifier
    retry_count INTEGER DEFAULT 0,  -- Number of retry attempts
    
    -- Indexes for performance
    CREATE INDEX idx_status ON shard_status(status);
    CREATE INDEX idx_updated_at ON shard_status(updated_at);
);
```

### Status Transitions

```
┌─────────┐     assign      ┌─────────┐
│ pending ├────────────────►│ running │
└─────────┘                 └────┬────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                success      failure      timeout
                    │            │            │
                    ▼            ▼            ▼
              ┌──────────┐ ┌─────────┐ ┌─────────┐
              │completed │ │ failed  │ │ stale   │
              └──────────┘ └─────────┘ └─────────┘
```

## Concurrency Model

### Parallel Execution Strategy

1. **Resource Pooling**:
   - Fixed pool of worker slots (PARALLEL_LIMIT)
   - Dynamic task assignment
   - Load balancing across workers

2. **Isolation Mechanisms**:
   - Separate Git worktrees per task
   - Independent file systems
   - No shared mutable state

3. **Synchronization Points**:
   - Database transactions for shard assignment
   - Atomic status updates
   - Distributed locking via database

### Concurrency Controls

```python
# Airflow DAG configuration
dag = DAG(
    'parallel_maintenance_poc',
    max_active_tasks=10,        # Global parallelism limit
    max_active_runs=1,          # Single DAG run at a time
    concurrency=10,             # Task concurrency limit
)

# Database-level controls
WITH locked_shard AS (
    UPDATE shard_status 
    SET status = 'running', 
        worker_id = :worker_id,
        started_at = CURRENT_TIMESTAMP
    WHERE shard_id = (
        SELECT shard_id 
        FROM shard_status 
        WHERE status = 'pending'
        LIMIT 1
    )
    RETURNING shard_id
)
SELECT * FROM locked_shard;
```

## Performance Considerations

### Bottlenecks and Mitigations

1. **Git Operations**:
   - **Bottleneck**: Worktree creation/deletion
   - **Mitigation**: Pre-create worktree pool, async cleanup

2. **Database Contention**:
   - **Bottleneck**: Concurrent shard assignments
   - **Mitigation**: Row-level locking, connection pooling

3. **File System I/O**:
   - **Bottleneck**: Multiple worktrees on same disk
   - **Mitigation**: SSD storage, tmpfs for temporary files

4. **Memory Usage**:
   - **Bottleneck**: Large repositories in memory
   - **Mitigation**: Shallow clones, sparse checkouts

### Performance Metrics

| Metric | Target | Monitoring |
|--------|--------|------------|
| Task Throughput | 100 tasks/hour | Airflow metrics |
| Worktree Creation | < 5 seconds | Application logs |
| Shard Assignment | < 100ms | Database query time |
| Memory per Task | < 1GB | System monitoring |
| Disk I/O | < 50MB/s sustained | iostat |

### Optimization Strategies

1. **Batch Processing**:
   ```python
   # Process multiple items per shard
   ITEMS_PER_SHARD = 10
   ```

2. **Connection Pooling**:
   ```python
   # Reuse database connections
   engine = create_engine(
       'sqlite:///shard_status.db',
       pool_size=20,
       max_overflow=0
   )
   ```

3. **Lazy Loading**:
   ```python
   # Only load required files
   git sparse-checkout set "src/" "tests/"
   ```

4. **Caching**:
   ```python
   # Cache frequently accessed data
   @lru_cache(maxsize=1000)
   def get_file_content(path):
       return Path(path).read_text()
   ```

## Failure Handling

### Failure Scenarios

1. **Task Failure**:
   - Automatic retry with exponential backoff
   - Error logging to database
   - Alerting via Airflow

2. **Worker Crash**:
   - Timeout detection (1 hour default)
   - Automatic shard reassignment
   - Worktree cleanup on restart

3. **Database Corruption**:
   - Regular backups
   - Integrity checks
   - Rebuild from Git history

4. **Resource Exhaustion**:
   - Resource limits per task
   - Automatic scaling down
   - Graceful degradation

### Recovery Procedures

```bash
# 1. Identify stuck shards
sqlite3 shard_status.db "
SELECT * FROM shard_status 
WHERE status = 'running' 
AND datetime(started_at) < datetime('now', '-1 hour');"

# 2. Reset stuck shards
sqlite3 shard_status.db "
UPDATE shard_status 
SET status = 'pending', worker_id = NULL 
WHERE status = 'running' 
AND datetime(started_at) < datetime('now', '-1 hour');"

# 3. Clean up orphaned worktrees
git worktree prune
find /tmp -name "worktree_*" -mtime +1 -exec rm -rf {} \;

# 4. Restart failed DAG run
airflow dags trigger parallel_maintenance_poc
```

## Security Considerations

1. **Access Control**:
   - Airflow RBAC for DAG access
   - File system permissions for worktrees
   - Database access restrictions

2. **Data Isolation**:
   - Separate worktrees prevent cross-contamination
   - Temporary directories with restricted permissions
   - No sensitive data in logs

3. **Resource Limits**:
   - CPU/memory limits per task
   - Disk quota enforcement
   - Network access restrictions

## Monitoring and Observability

### Key Metrics

1. **System Health**:
   - Active worktree count
   - Database connection pool usage
   - Disk space utilization

2. **Task Performance**:
   - Task execution time
   - Success/failure rates
   - Retry frequency

3. **Resource Usage**:
   - CPU utilization per worker
   - Memory consumption
   - I/O wait time

### Monitoring Stack

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Airflow    │────►│  Prometheus  │────►│   Grafana     │
│  Metrics    │     │              │     │  Dashboards   │
└─────────────┘     └──────────────┘     └───────────────┘
       │                    │                      │
       │                    │                      │
       ▼                    ▼                      ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ Application │     │   Database   │     │    Alerts     │
│    Logs     │     │   Metrics    │     │  (PagerDuty)  │
└─────────────┘     └──────────────┘     └───────────────┘
```

## Future Enhancements

1. **Distributed Execution**:
   - Kubernetes executor for Airflow
   - Multi-node worker pools
   - Cross-region replication

2. **Advanced Scheduling**:
   - Priority-based shard assignment
   - Predictive resource allocation
   - Dynamic parallelism adjustment

3. **Enhanced Monitoring**:
   - Real-time task progress tracking
   - Predictive failure detection
   - Automated remediation

4. **Performance Optimizations**:
   - Worktree pooling
   - Incremental processing
   - Result caching