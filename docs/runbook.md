# Parallel Processing System Runbook

## Overview

The Parallel Processing System is an Apache Airflow-based solution designed to efficiently process maintenance tasks across multiple Git worktrees in parallel. It uses a DAG (Directed Acyclic Graph) to orchestrate parallel execution of tasks while managing system resources and preventing conflicts.

Key components:
- **WorktreeManager**: Manages Git worktree lifecycle (creation, cleanup)
- **ShardCoordinator**: Coordinates parallel task execution and tracks progress
- **ClaudeRunner**: Executes AI-powered maintenance tasks
- **Shard Status Database**: SQLite database tracking task execution state

## Prerequisites

Before operating this system, ensure you have:

1. **Required Software**:
   - Git ≥ 2.20 (for worktree support)
   - Python ≥ 3.10
   - Apache Airflow (latest stable version)
   - SQLite3 (for database inspection)

2. **Access Requirements**:
   - Read/write access to the repository
   - Permissions to create/delete Git worktrees
   - Access to Airflow web UI and CLI
   - Database read permissions

3. **Environment Setup**:
   - `AIRFLOW_HOME` environment variable set
   - Airflow database initialized (`airflow db init`)
   - DAGs folder configured to include project's `dags/` directory

## How to Run smoke_run.sh

The smoke run script performs a test execution of the parallel maintenance DAG to verify system functionality.

### Step-by-Step Instructions

1. **Navigate to the project root**:
   ```bash
   cd /path/to/IndexAgent
   ```

2. **Run the smoke test**:
   ```bash
   ./scripts/smoke_run.sh
   ```

3. **Expected Output**:
   - Pre-flight checks (Airflow installation, DAG registration)
   - DAG execution with parallel limit of 10
   - Post-run verification of shard database
   - Success/failure status with colored output

### Interpreting Results

- **Green [SUCCESS]**: Component verified successfully
- **Blue [INFO]**: Informational message
- **Yellow [WARNING]**: Non-critical issue, review recommended
- **Red [ERROR]**: Critical failure, action required

### Common Issues

1. **"Airflow is not installed"**: Install Airflow or update PATH
2. **"DAG file not found"**: Ensure you're in the correct directory
3. **"Database is not initialized"**: Run `airflow db init`

## How to Inspect shard_status.db

The shard status database tracks the execution state of parallel tasks.

### Database Location

```
<project_root>/shard_status.db
```

### Useful SQL Queries

1. **View all shard statuses**:
   ```sql
   sqlite3 shard_status.db "SELECT * FROM shard_status ORDER BY updated_at DESC;"
   ```

2. **Check currently running shards**:
   ```sql
   sqlite3 shard_status.db "SELECT shard_id, status, started_at FROM shard_status WHERE status = 'running';"
   ```

3. **Find failed shards**:
   ```sql
   sqlite3 shard_status.db "SELECT shard_id, error_message, updated_at FROM shard_status WHERE status = 'failed';"
   ```

4. **Get execution statistics**:
   ```sql
   sqlite3 shard_status.db "
   SELECT 
     status, 
     COUNT(*) as count,
     AVG(JULIANDAY(completed_at) - JULIANDAY(started_at)) * 24 * 60 as avg_duration_minutes
   FROM shard_status 
   GROUP BY status;"
   ```

5. **Check for stale locks**:
   ```sql
   sqlite3 shard_status.db "
   SELECT shard_id, started_at, status 
   FROM shard_status 
   WHERE status = 'running' 
   AND datetime(started_at) < datetime('now', '-1 hour');"
   ```

### Database Schema

```sql
CREATE TABLE shard_status (
    shard_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    worker_id TEXT
);
```

## How to Manually Prune Stale Worktrees

Stale worktrees may accumulate if tasks fail unexpectedly or the system is interrupted.

### When This Might Be Needed

- After system crashes or unexpected shutdowns
- When disk space is running low
- If Git operations become slow
- After failed DAG runs that didn't clean up properly

### Identifying Stale Worktrees

1. **List all worktrees**:
   ```bash
   git worktree list
   ```

2. **Find worktrees older than 24 hours**:
   ```bash
   find .git/worktrees -name "gitdir" -mtime +1 | while read f; do
     dirname "$f" | xargs basename
   done
   ```

3. **Check for orphaned worktree directories**:
   ```bash
   # List worktree directories not tracked by Git
   ls -la /tmp/worktree_* 2>/dev/null | grep -v "$(git worktree list | awk '{print $1}')"
   ```

### Safe Cleanup Procedures

1. **Automated cleanup (recommended)**:
   ```bash
   # Prune worktrees marked for deletion
   git worktree prune -v
   
   # Force prune if needed (use with caution)
   git worktree prune -v --expire=now
   ```

2. **Manual cleanup for specific worktree**:
   ```bash
   # Remove a specific worktree
   WORKTREE_PATH="/tmp/worktree_shard_001"
   git worktree remove "$WORKTREE_PATH" --force
   
   # If that fails, manual removal
   rm -rf "$WORKTREE_PATH"
   git worktree prune
   ```

3. **Bulk cleanup script**:
   ```bash
   #!/bin/bash
   # Clean up all worktrees older than 24 hours
   git worktree list --porcelain | grep "worktree" | cut -d' ' -f2 | while read wt; do
     if [ -d "$wt" ] && [ "$(find "$wt" -maxdepth 0 -mtime +1)" ]; then
       echo "Removing old worktree: $wt"
       git worktree remove "$wt" --force || rm -rf "$wt"
     fi
   done
   git worktree prune
   ```

### Post-Cleanup Verification

```bash
# Verify worktrees are cleaned up
git worktree list

# Check disk usage
df -h /tmp

# Verify Git repository health
git fsck
```

## Monitoring and Alerts

### Key Metrics to Watch

1. **System Metrics**:
   - CPU usage (should not exceed 80% sustained)
   - Memory usage (watch for memory leaks)
   - Disk I/O (monitor for bottlenecks)
   - Available disk space in `/tmp`

2. **Application Metrics**:
   - Number of active worktrees
   - Task execution duration
   - Success/failure rates
   - Queue depth

3. **Database Metrics**:
   - Database file size
   - Number of running tasks
   - Stale lock count

### Log Locations

1. **Airflow Logs**:
   ```
   $AIRFLOW_HOME/logs/dag_id=parallel_maintenance_poc/
   ```

2. **System Logs**:
   ```
   /var/log/syslog          # Ubuntu/Debian
   /var/log/messages        # RHEL/CentOS
   ```

3. **Application Logs**:
   ```
   <project_root>/logs/     # If configured
   ```

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Active Worktrees | > 20 | > 30 | Investigate stuck tasks |
| Task Duration | > 30 min | > 60 min | Check for deadlocks |
| Failed Tasks | > 5% | > 10% | Review error logs |
| Disk Space (/tmp) | < 20% | < 10% | Clean up worktrees |
| Database Size | > 100MB | > 500MB | Archive old records |
| Stale Locks | > 5 | > 10 | Manual intervention |

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. DAG Not Found
**Symptoms**: Airflow cannot find the parallel_maintenance_poc DAG

**Solutions**:
- Verify DAG file exists: `ls dags/parallel_maintenance_poc.py`
- Check Airflow DAG folder configuration
- Trigger DAG parsing: `airflow dags list`
- Check for syntax errors: `python dags/parallel_maintenance_poc.py`

#### 2. Worktree Creation Failures
**Symptoms**: Tasks fail with "Could not create worktree" errors

**Solutions**:
- Check disk space: `df -h /tmp`
- Verify Git version: `git --version` (needs ≥ 2.20)
- Clean up stale worktrees (see cleanup procedures)
- Check permissions on `/tmp`

#### 3. Database Lock Errors
**Symptoms**: "database is locked" errors in logs

**Solutions**:
- Check for multiple processes accessing database
- Implement retry logic with exponential backoff
- Consider migrating to PostgreSQL for production

#### 4. Memory Issues
**Symptoms**: Tasks killed with OOM errors

**Solutions**:
- Reduce PARALLEL_LIMIT in smoke_run.sh
- Increase system memory
- Monitor memory usage per task
- Implement memory limits in Airflow

#### 5. Slow Performance
**Symptoms**: Tasks taking longer than expected

**Solutions**:
- Check system resources (CPU, I/O)
- Review parallel limit settings
- Analyze task distribution
- Consider task-level optimization

### Debug Procedures

1. **Enable Debug Logging**:
   ```bash
   export AIRFLOW__LOGGING__LOGGING_LEVEL=DEBUG
   airflow dags test parallel_maintenance_poc $(date +%Y-%m-%d)
   ```

2. **Trace Specific Task**:
   ```bash
   # Get task logs
   airflow tasks test parallel_maintenance_poc <task_id> $(date +%Y-%m-%d)
   ```

3. **Database Debugging**:
   ```bash
   # Enable SQLite tracing
   sqlite3 shard_status.db
   .trace stdout
   SELECT * FROM shard_status;
   ```

4. **Git Worktree Debugging**:
   ```bash
   # Verbose worktree operations
   GIT_TRACE=1 git worktree add /tmp/test_worktree
   ```

### Escalation Paths

1. **Level 1 - Operations Team**:
   - Restart failed tasks
   - Clean up worktrees
   - Monitor system resources

2. **Level 2 - Development Team**:
   - Debug code issues
   - Optimize performance
   - Fix bugs in DAG logic

3. **Level 3 - Architecture Team**:
   - System design changes
   - Scaling decisions
   - Infrastructure upgrades

### Emergency Procedures

**System Overload**:
```bash
# Stop all running tasks
airflow dags pause parallel_maintenance_poc

# Clean up all worktrees
git worktree list | grep -v "bare" | awk '{print $1}' | xargs -I {} git worktree remove {} --force
git worktree prune

# Clear task instances
airflow tasks clear parallel_maintenance_poc -s $(date +%Y-%m-%d) -e $(date +%Y-%m-%d)
```

**Database Corruption**:
```bash
# Backup current database
cp shard_status.db shard_status.db.backup

# Verify integrity
sqlite3 shard_status.db "PRAGMA integrity_check;"

# If corrupted, restore from backup or reinitialize
```

## Maintenance Windows

Recommended maintenance schedule:
- **Daily**: Check logs for errors, monitor metrics
- **Weekly**: Clean up old worktrees, archive logs
- **Monthly**: Database maintenance, performance review
- **Quarterly**: System health check, capacity planning

## Contact Information

- **Operations Team**: ops@example.com
- **Development Team**: dev@example.com
- **On-Call**: +1-555-0123 (24/7)
- **Slack Channel**: #parallel-processing-support