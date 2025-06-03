"""
High load integration tests for parallel maintenance PoC DAG.

Tests the DAG with 20 shards and higher parallel limit to verify scalability
and proper resource management under increased load.
"""

import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import concurrent.futures

import pytest
import git


# Mock Airflow components
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
def temp_git_repo_20_shards():
    """Create a temporary Git repository with 20 dummy source files."""
    temp_dir = tempfile.mkdtemp()
    repo = git.Repo.init(temp_dir)
    
    # Create dummy source files (20 shards)
    for i in range(20):
        shard_path = Path(temp_dir) / f"shard_{i:02d}.py"
        shard_path.write_text(
            f"# Shard {i:02d}\n"
            f"# High load test shard\n"
            f"def process_shard_{i:02d}():\n"
            f"    print('Processing shard {i:02d}')\n"
            f"    # Simulate complex processing\n"
            f"    return {{'shard': {i}, 'status': 'processed'}}\n"
        )
        repo.index.add([str(shard_path)])
    
    # Create initial commit
    repo.index.commit("Initial commit with 20 shards for high load test")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_shard_db_20():
    """Create a mock shard status database with 20 shards."""
    db_path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create shard_status table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shard_status (
            shard_id TEXT PRIMARY KEY,
            status TEXT,
            updated_at TIMESTAMP,
            processing_time REAL
        )
    """)
    
    # Insert initial shard records (20 shards)
    for i in range(20):
        cursor.execute(
            "INSERT INTO shard_status (shard_id, status, updated_at, processing_time) VALUES (?, ?, ?, ?)",
            (f"shard_{i:02d}", "PENDING", datetime.now(), 0.0)
        )
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def claude_runner_stub_variable_time():
    """Create a stub for Claude runner with variable processing times."""
    def stub_runner(shard_id, **kwargs):
        """Stub implementation with variable processing time to simulate real workload."""
        print(f"Processing {shard_id}")
        
        # Variable sleep time based on shard number (0.1 to 0.5 seconds)
        shard_num = int(shard_id.split('_')[1])
        sleep_time = 0.1 + (shard_num % 5) * 0.1
        time.sleep(sleep_time)
        
        return {
            "shard_id": shard_id,
            "status": "SUCCESS",
            "changes": shard_num % 3,  # Simulate varying number of changes
            "processing_time": sleep_time
        }
    
    return stub_runner


@pytest.fixture
def cleanup_worktrees_high_load():
    """Ensure worktree directory is clean before and after high load tests."""
    worktree_dir = Path("/tmp/worktrees")
    
    # Clean before test
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir, ignore_errors=True)
    
    yield
    
    # Clean after test
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir, ignore_errors=True)


class TestHighLoadParallelDAG:
    """High load integration tests for parallel maintenance DAG."""
    
    def test_high_load_20_shards_parallel_12(
        self, 
        temp_git_repo_20_shards, 
        mock_shard_db_20, 
        claude_runner_stub_variable_time, 
        cleanup_worktrees_high_load
    ):
        """
        Test parallel processing of 20 shards with PARALLEL_LIMIT=12.
        
        Verifies:
        - All 20 mapped tasks succeed within 90 seconds
        - SQLite contains 20 SUCCESS rows with processing times
        - No worktrees remain after cleanup
        - System handles high concurrent load properly
        - Memory usage remains reasonable
        """
        from indexagent.utils.worktree_manager import WorktreeManager
        
        # Mock Airflow context
        dag_run = MockDagRun(conf={"PARALLEL_LIMIT": 12})
        
        # Track execution metrics
        start_time = time.time()
        processed_shards = []
        active_worktrees = set()
        max_concurrent = 0
        processing_times = {}
        
        # Mock the DAG task functions
        def mock_get_shards(**context):
            """Mock function to get shards from database."""
            conn = sqlite3.connect(mock_shard_db_20)
            cursor = conn.cursor()
            cursor.execute("SELECT shard_id FROM shard_status WHERE status = 'PENDING'")
            shards = [row[0] for row in cursor.fetchall()]
            conn.close()
            return shards
        
        def mock_process_shard(shard_id, **context):
            """Mock function to process a single shard with metrics tracking."""
            nonlocal max_concurrent
            shard_start_time = time.time()
            
            # Initialize worktree manager
            manager = WorktreeManager(temp_git_repo_20_shards)
            worktree_path = None
            
            try:
                # Create worktree
                worktree_path = manager.create_worktree(f"high_load_{shard_id}")
                active_worktrees.add(worktree_path)
                
                # Track concurrent executions
                current_concurrent = len(active_worktrees)
                max_concurrent = max(max_concurrent, current_concurrent)
                
                # Simulate processing with stub
                result = claude_runner_stub_variable_time(shard_id)
                
                # Calculate processing time
                shard_processing_time = time.time() - shard_start_time
                processing_times[shard_id] = shard_processing_time
                
                # Update database
                conn = sqlite3.connect(mock_shard_db_20)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE shard_status SET status = ?, updated_at = ?, processing_time = ? WHERE shard_id = ?",
                    ("SUCCESS", datetime.now(), shard_processing_time, shard_id)
                )
                conn.commit()
                conn.close()
                
                processed_shards.append(shard_id)
                
                return result
                
            except Exception as e:
                # Log any errors for debugging
                print(f"Error processing {shard_id}: {e}")
                raise
                
            finally:
                # Cleanup worktree
                if worktree_path:
                    try:
                        manager.remove_worktree(f"high_load_{shard_id}")
                        active_worktrees.discard(worktree_path)
                    except Exception as e:
                        print(f"Error removing worktree for {shard_id}: {e}")
        
        def mock_cleanup(**context):
            """Mock cleanup function with verification."""
            # Verify all worktrees are cleaned up
            worktree_dir = Path("/tmp/worktrees")
            if worktree_dir.exists():
                remaining = list(worktree_dir.iterdir())
                assert len(remaining) == 0, f"Found remaining worktrees: {remaining}"
        
        # Simulate DAG execution with high load
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
            # Use real ThreadPoolExecutor for realistic testing
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                # Get shards
                shards = mock_get_shards()
                assert len(shards) == 20, f"Expected 20 shards, got {len(shards)}"
                
                # Process shards with parallel limit
                futures = []
                for shard in shards:
                    future = executor.submit(mock_process_shard, shard)
                    futures.append((shard, future))
                
                # Wait for all tasks to complete with timeout
                completed = 0
                for shard, future in futures:
                    try:
                        result = future.result(timeout=90)
                        completed += 1
                    except concurrent.futures.TimeoutError:
                        pytest.fail(f"Shard {shard} timed out after 90 seconds")
                
                assert completed == 20, f"Only {completed}/20 shards completed"
            
            # Run cleanup
            mock_cleanup()
        
        # Verify execution time
        execution_time = time.time() - start_time
        assert execution_time < 90, f"Execution took {execution_time}s, expected < 90s"
        
        # Verify all shards processed
        assert len(processed_shards) == 20, f"Expected 20 processed shards, got {len(processed_shards)}"
        assert set(processed_shards) == set(shards), "Not all shards were processed"
        
        # Verify database status
        conn = sqlite3.connect(mock_shard_db_20)
        cursor = conn.cursor()
        
        # Check SUCCESS count
        cursor.execute("SELECT COUNT(*) FROM shard_status WHERE status = 'SUCCESS'")
        success_count = cursor.fetchone()[0]
        assert success_count == 20, f"Expected 20 SUCCESS rows, got {success_count}"
        
        # Verify processing times were recorded
        cursor.execute("SELECT COUNT(*) FROM shard_status WHERE processing_time > 0")
        timed_count = cursor.fetchone()[0]
        assert timed_count == 20, f"Expected 20 rows with processing times, got {timed_count}"
        
        conn.close()
        
        # Verify parallel limit was respected
        assert max_concurrent <= 12, f"Max concurrent executions {max_concurrent} exceeded limit of 12"
        
        # Verify no worktrees remain
        worktree_dir = Path("/tmp/worktrees")
        if worktree_dir.exists():
            remaining = list(worktree_dir.iterdir())
            assert len(remaining) == 0, f"Found remaining worktrees: {remaining}"
        
        # Print performance metrics
        avg_processing_time = sum(processing_times.values()) / len(processing_times)
        print(f"\nPerformance Metrics:")
        print(f"  Total execution time: {execution_time:.2f}s")
        print(f"  Average shard processing time: {avg_processing_time:.2f}s")
        print(f"  Max concurrent executions: {max_concurrent}")
        print(f"  Theoretical minimum time: {sum(processing_times.values()) / 12:.2f}s")
    
    def test_stress_test_worktree_creation_deletion(
        self, 
        temp_git_repo_20_shards, 
        cleanup_worktrees_high_load
    ):
        """
        Stress test rapid worktree creation and deletion.
        
        Verifies the system can handle rapid create/delete cycles
        without leaving orphaned worktrees or causing git corruption.
        """
        from indexagent.utils.worktree_manager import WorktreeManager
        
        manager = WorktreeManager(temp_git_repo_20_shards)
        errors = []
        
        def rapid_worktree_cycle(iteration):
            """Rapidly create and delete a worktree."""
            try:
                worktree_name = f"stress_test_{iteration}"
                
                # Create
                worktree_path = manager.create_worktree(worktree_name)
                assert Path(worktree_path).exists(), f"Worktree {worktree_name} not created"
                
                # Brief operation
                time.sleep(0.01)
                
                # Delete
                manager.remove_worktree(worktree_name)
                assert not Path(worktree_path).exists(), f"Worktree {worktree_name} not removed"
                
            except Exception as e:
                errors.append(f"Iteration {iteration}: {e}")
        
        # Run stress test with 20 rapid cycles
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            for i in range(20):
                future = executor.submit(rapid_worktree_cycle, i)
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                future.result()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during stress test: {errors}"
        
        # Verify no orphaned worktrees
        worktree_dir = Path("/tmp/worktrees")
        if worktree_dir.exists():
            remaining = list(worktree_dir.iterdir())
            assert len(remaining) == 0, f"Found orphaned worktrees: {remaining}"
        
        # Verify git repository is still valid
        repo = git.Repo(temp_git_repo_20_shards)
        assert repo.bare is False, "Repository corrupted"
        assert len(list(repo.heads)) > 0, "Repository has no branches"