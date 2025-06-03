"""
Integration tests for parallel maintenance PoC DAG.

Tests the complete workflow of parallel task processing using Dynamic Task Mapping
with worktree management, ensuring proper resource cleanup and parallel execution limits.
"""

import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import pytest
import git


# Mock Airflow components for testing
class MockTaskInstance:
    """Mock Airflow TaskInstance."""
    def __init__(self, task_id, dag_id="parallel_maintenance_poc"):
        self.task_id = task_id
        self.dag_id = dag_id
        self.state = "running"
        self.log = Mock()
        
    def xcom_push(self, key, value):
        """Mock xcom_push."""
        pass


class MockDagRun:
    """Mock Airflow DagRun."""
    def __init__(self, conf=None):
        self.conf = conf or {}
        self.run_id = f"test_run_{datetime.now().isoformat()}"


@pytest.fixture
def temp_git_repo():
    """Create a temporary Git repository with dummy source files."""
    temp_dir = tempfile.mkdtemp()
    repo = git.Repo.init(temp_dir)
    
    # Create dummy source files (8 shards)
    for i in range(8):
        shard_path = Path(temp_dir) / f"shard_{i}.py"
        shard_path.write_text(f"# Shard {i}\nprint('Processing shard {i}')\n")
        repo.index.add([str(shard_path)])
    
    # Create initial commit
    repo.index.commit("Initial commit with 8 shards")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_shard_db():
    """Create a mock shard status database."""
    db_path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create shard_status table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shard_status (
            shard_id TEXT PRIMARY KEY,
            status TEXT,
            updated_at TIMESTAMP
        )
    """)
    
    # Insert initial shard records
    for i in range(8):
        cursor.execute(
            "INSERT INTO shard_status (shard_id, status, updated_at) VALUES (?, ?, ?)",
            (f"shard_{i}", "PENDING", datetime.now())
        )
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def claude_runner_stub():
    """Create a stub for Claude runner that echoes shard ID and sleeps briefly."""
    def stub_runner(shard_id, **kwargs):
        """Stub implementation that simulates processing."""
        print(f"Processing {shard_id}")
        time.sleep(0.2)  # Simulate processing time
        return {"shard_id": shard_id, "status": "SUCCESS", "changes": 0}
    
    return stub_runner


@pytest.fixture
def cleanup_worktrees():
    """Ensure worktree directory is clean before and after tests."""
    worktree_dir = Path("/tmp/worktrees")
    
    # Clean before test
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir, ignore_errors=True)
    
    yield
    
    # Clean after test
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir, ignore_errors=True)


class TestParallelDAG:
    """Integration tests for parallel maintenance DAG."""
    
    def test_parallel_processing_with_limit(
        self, temp_git_repo, mock_shard_db, claude_runner_stub, cleanup_worktrees
    ):
        """
        Test parallel processing of 8 shards with PARALLEL_LIMIT=6.
        
        Verifies:
        - All mapped tasks succeed within 30 seconds
        - SQLite contains 8 SUCCESS rows
        - No worktrees remain after cleanup
        - No thread-safety warnings or git-lock errors
        """
        from indexagent.utils.worktree_manager import WorktreeManager
        
        # Mock Airflow context
        dag_run = MockDagRun(conf={"PARALLEL_LIMIT": 6})
        
        # Track execution metrics
        start_time = time.time()
        processed_shards = []
        active_worktrees = set()
        max_concurrent = 0
        
        # Mock the DAG task functions
        def mock_get_shards(**context):
            """Mock function to get shards from database."""
            conn = sqlite3.connect(mock_shard_db)
            cursor = conn.cursor()
            cursor.execute("SELECT shard_id FROM shard_status WHERE status = 'PENDING'")
            shards = [row[0] for row in cursor.fetchall()]
            conn.close()
            return shards
        
        def mock_process_shard(shard_id, **context):
            """Mock function to process a single shard."""
            nonlocal max_concurrent
            
            # Initialize worktree manager
            manager = WorktreeManager(temp_git_repo)
            worktree_path = None
            
            try:
                # Create worktree
                worktree_path = manager.create_worktree(f"test_{shard_id}")
                active_worktrees.add(worktree_path)
                
                # Track concurrent executions
                current_concurrent = len(active_worktrees)
                max_concurrent = max(max_concurrent, current_concurrent)
                
                # Simulate processing with stub
                result = claude_runner_stub(shard_id)
                
                # Update database
                conn = sqlite3.connect(mock_shard_db)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE shard_status SET status = ?, updated_at = ? WHERE shard_id = ?",
                    ("SUCCESS", datetime.now(), shard_id)
                )
                conn.commit()
                conn.close()
                
                processed_shards.append(shard_id)
                
                return result
                
            finally:
                # Cleanup worktree
                if worktree_path:
                    manager.remove_worktree(f"test_{shard_id}")
                    active_worktrees.discard(worktree_path)
        
        def mock_cleanup(**context):
            """Mock cleanup function."""
            # Verify all worktrees are cleaned up
            worktree_dir = Path("/tmp/worktrees")
            if worktree_dir.exists():
                remaining = list(worktree_dir.iterdir())
                assert len(remaining) == 0, f"Found remaining worktrees: {remaining}"
        
        # Simulate DAG execution with parallel processing
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
            # Configure mock executor to respect PARALLEL_LIMIT
            executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = executor_instance
            
            # Get shards
            shards = mock_get_shards()
            assert len(shards) == 8, f"Expected 8 shards, got {len(shards)}"
            
            # Process shards with parallel limit
            futures = []
            with mock_executor(max_workers=6) as executor:
                for shard in shards:
                    future = executor.submit(mock_process_shard, shard)
                    futures.append(future)
                
                # Wait for all tasks to complete
                for future in futures:
                    future.result()
            
            # Run cleanup
            mock_cleanup()
        
        # Verify execution time
        execution_time = time.time() - start_time
        assert execution_time < 30, f"Execution took {execution_time}s, expected < 30s"
        
        # Verify all shards processed
        assert len(processed_shards) == 8, f"Expected 8 processed shards, got {len(processed_shards)}"
        
        # Verify database status
        conn = sqlite3.connect(mock_shard_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM shard_status WHERE status = 'SUCCESS'")
        success_count = cursor.fetchone()[0]
        conn.close()
        assert success_count == 8, f"Expected 8 SUCCESS rows, got {success_count}"
        
        # Verify parallel limit was respected
        assert max_concurrent <= 6, f"Max concurrent executions {max_concurrent} exceeded limit of 6"
        
        # Verify no worktrees remain
        worktree_dir = Path("/tmp/worktrees")
        if worktree_dir.exists():
            remaining = list(worktree_dir.iterdir())
            assert len(remaining) == 0, f"Found remaining worktrees: {remaining}"
    
    def test_no_git_lock_errors(self, temp_git_repo, claude_runner_stub, cleanup_worktrees):
        """Test that parallel operations don't cause git lock errors."""
        from indexagent.utils.worktree_manager import WorktreeManager
        
        errors = []
        
        def process_with_error_tracking(shard_id):
            """Process shard and track any git errors."""
            try:
                manager = WorktreeManager(temp_git_repo)
                worktree_path = manager.create_worktree(f"test_{shard_id}")
                
                # Simulate some git operations
                time.sleep(0.1)
                
                manager.remove_worktree(f"test_{shard_id}")
                
            except Exception as e:
                if "lock" in str(e).lower() or "index.lock" in str(e):
                    errors.append(f"Git lock error for {shard_id}: {e}")
                else:
                    raise
        
        # Run parallel operations
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for i in range(8):
                future = executor.submit(process_with_error_tracking, f"shard_{i}")
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                future.result()
        
        # Verify no git lock errors occurred
        assert len(errors) == 0, f"Git lock errors occurred: {errors}"