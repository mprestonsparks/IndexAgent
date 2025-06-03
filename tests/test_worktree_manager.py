"""
Tests for the worktree manager module.

These tests use temporary Git repositories to test worktree operations
in isolation without affecting the main repository.
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from indexagent.utils.worktree_manager import (
    WorktreeError,
    WorktreeCreationError,
    WorktreeRemovalError,
    create_worktree,
    remove_worktree,
    cleanup_stale,
    list_worktrees,
    _validate_branch_name,
    _validate_path,
    _run_git_command,
)


class TestGitRepository:
    """Context manager for creating temporary Git repositories for testing."""
    
    def __init__(self):
        self.temp_dir = None
        self.original_cwd = None
    
    def __enter__(self):
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="test_worktree_")
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)
        
        # Create initial commit
        Path("README.md").write_text("# Test Repository")
        subprocess.run(["git", "add", "README.md"], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], check=True, capture_output=True)
        
        # Create main branch (for older Git versions that use master)
        try:
            subprocess.run(["git", "branch", "-M", "main"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # If main already exists or can't rename, that's fine
            pass
        
        return self.temp_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.original_cwd)
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestValidation(unittest.TestCase):
    """Test input validation functions."""
    
    def test_validate_branch_name_valid(self):
        """Test validation of valid branch names."""
        valid_names = [
            "main",
            "feature/new-feature",
            "cc-parallel/todo-001",
            "release-1.0.0",
            "hotfix_urgent",
        ]
        for name in valid_names:
            _validate_branch_name(name)  # Should not raise
    
    def test_validate_branch_name_invalid(self):
        """Test validation rejects invalid branch names."""
        invalid_names = [
            "",  # Empty
            "feature branch",  # Space
            "feature~branch",  # Tilde
            "feature^branch",  # Caret
            "feature:branch",  # Colon
            "feature?branch",  # Question mark
            "feature*branch",  # Asterisk
            "feature[branch",  # Bracket
            "feature\\branch",  # Backslash
            ".feature",  # Starts with dot
            "feature.",  # Ends with dot
            "feature.lock",  # Ends with .lock
            "feature//branch",  # Double slash
        ]
        for name in invalid_names:
            with self.assertRaises(ValueError):
                _validate_branch_name(name)
    
    def test_validate_path_valid(self):
        """Test validation of valid paths."""
        valid_paths = [
            "worktrees/todo-001",
            "./worktrees/todo-001",
            "../worktrees/todo-001",
            "/tmp/worktrees/todo-001",
            "/tmp/test/path",
        ]
        for path in valid_paths:
            _validate_path(path)  # Should not raise
    
    def test_validate_path_invalid(self):
        """Test validation rejects invalid paths."""
        invalid_paths = [
            "",  # Empty
            "/usr/bin/test",  # Absolute path not under /tmp
            "/home/user/test",  # Absolute path not under /tmp
        ]
        for path in invalid_paths:
            with self.assertRaises(ValueError):
                _validate_path(path)


class TestGitCommand(unittest.TestCase):
    """Test Git command execution."""
    
    def test_run_git_command_success(self):
        """Test successful Git command execution."""
        with TestGitRepository():
            result = _run_git_command(["status", "--short"])
            self.assertEqual(result.returncode, 0)
            self.assertIsInstance(result.stdout, str)
    
    def test_run_git_command_failure(self):
        """Test Git command failure handling."""
        with TestGitRepository():
            with self.assertRaises(WorktreeError) as cm:
                _run_git_command(["invalid-command"])
            self.assertIn("Git command failed", str(cm.exception))
    
    def test_run_git_command_with_cwd(self):
        """Test Git command execution in different directory."""
        with TestGitRepository() as repo_dir:
            # Create subdirectory
            subdir = Path(repo_dir) / "subdir"
            subdir.mkdir()
            
            # Run command from parent directory
            result = _run_git_command(["status", "--short"], cwd=str(subdir))
            self.assertEqual(result.returncode, 0)


class TestCreateWorktree(unittest.TestCase):
    """Test worktree creation functionality."""
    
    def test_create_worktree_new_branch(self):
        """Test creating worktree with a new branch."""
        with TestGitRepository():
            worktree_path = "worktrees/test-001"
            branch_name = "cc-parallel/test-001"
            
            create_worktree(branch_name, worktree_path)
            
            # Verify worktree exists
            self.assertTrue(Path(worktree_path).exists())
            self.assertTrue(Path(worktree_path).is_dir())
            
            # Verify branch was created
            result = subprocess.run(
                ["git", "branch", "--list", branch_name],
                capture_output=True,
                text=True
            )
            self.assertIn(branch_name, result.stdout)
            
            # Verify worktree is in list
            worktrees = list_worktrees()
            paths = [wt["path"] for wt in worktrees]
            self.assertTrue(any(worktree_path in path for path in paths))
    
    def test_create_worktree_existing_branch(self):
        """Test creating worktree with an existing branch."""
        with TestGitRepository():
            branch_name = "existing-branch"
            
            # Create branch first
            subprocess.run(["git", "branch", branch_name], check=True)
            
            worktree_path = "worktrees/existing"
            create_worktree(branch_name, worktree_path)
            
            # Verify worktree exists
            self.assertTrue(Path(worktree_path).exists())
            
            # Verify it's using the existing branch
            worktrees = list_worktrees()
            for wt in worktrees:
                if worktree_path in wt["path"]:
                    self.assertEqual(wt["branch"], f"refs/heads/{branch_name}")
                    break
    
    def test_create_worktree_creates_parent_dirs(self):
        """Test that parent directories are created automatically."""
        with TestGitRepository():
            worktree_path = "deep/nested/path/worktree"
            branch_name = "test-nested"
            
            create_worktree(branch_name, worktree_path)
            
            self.assertTrue(Path(worktree_path).exists())
            self.assertTrue(Path("deep/nested/path").is_dir())
    
    def test_create_worktree_path_exists_error(self):
        """Test error when worktree path already exists."""
        with TestGitRepository():
            worktree_path = "existing-path"
            Path(worktree_path).mkdir()
            
            with self.assertRaises(WorktreeCreationError) as cm:
                create_worktree("test-branch", worktree_path)
            self.assertIn("Path already exists", str(cm.exception))
    
    def test_create_worktree_invalid_branch_name(self):
        """Test error with invalid branch name."""
        with TestGitRepository():
            with self.assertRaises(ValueError):
                create_worktree("invalid branch name", "worktree-path")
    
    def test_create_worktree_invalid_path(self):
        """Test error with invalid path."""
        with TestGitRepository():
            with self.assertRaises(ValueError):
                create_worktree("valid-branch", "")
    
    def test_create_worktree_custom_base_branch(self):
        """Test creating worktree from custom base branch."""
        with TestGitRepository():
            # Create a custom base branch
            base_branch = "develop"
            subprocess.run(["git", "branch", base_branch], check=True)
            
            worktree_path = "worktrees/from-develop"
            branch_name = "feature/from-develop"
            
            create_worktree(branch_name, worktree_path, base_branch=base_branch)
            
            self.assertTrue(Path(worktree_path).exists())


class TestRemoveWorktree(unittest.TestCase):
    """Test worktree removal functionality."""
    
    def test_remove_worktree_success(self):
        """Test successful worktree removal."""
        with TestGitRepository():
            # Create a worktree first
            worktree_path = "worktrees/to-remove"
            branch_name = "to-remove"
            create_worktree(branch_name, worktree_path)
            
            # Remove it
            remove_worktree(worktree_path)
            
            # Verify it's gone
            self.assertFalse(Path(worktree_path).exists())
            
            # Verify it's not in the list
            worktrees = list_worktrees()
            paths = [wt["path"] for wt in worktrees]
            self.assertFalse(any(worktree_path in path for path in paths))
    
    def test_remove_worktree_force_locked(self):
        """Test force removal of locked worktree."""
        with TestGitRepository():
            # Create a worktree
            worktree_path = "worktrees/locked"
            branch_name = "locked-branch"
            create_worktree(branch_name, worktree_path)
            
            # Lock the worktree
            subprocess.run(["git", "worktree", "lock", worktree_path], check=True)
            
            # Should still be able to remove with force
            remove_worktree(worktree_path)
            
            # Verify it's gone
            self.assertFalse(Path(worktree_path).exists())
    
    def test_remove_worktree_invalid_path(self):
        """Test error with invalid path."""
        with TestGitRepository():
            with self.assertRaises(ValueError):
                remove_worktree("")
    
    def test_remove_worktree_nonexistent(self):
        """Test removing non-existent worktree."""
        with TestGitRepository():
            with self.assertRaises(WorktreeRemovalError):
                remove_worktree("nonexistent-worktree")


class TestCleanupStale(unittest.TestCase):
    """Test stale worktree cleanup functionality."""
    
    def test_cleanup_stale_removes_deleted_worktrees(self):
        """Test cleanup removes worktrees with deleted directories."""
        with TestGitRepository():
            # Create worktrees
            worktree_path1 = "worktrees/stale1"
            worktree_path2 = "worktrees/stale2"
            create_worktree("stale1", worktree_path1)
            create_worktree("stale2", worktree_path2)
            
            # Manually delete one worktree directory (simulating stale worktree)
            shutil.rmtree(worktree_path1)
            
            # Run cleanup
            removed = cleanup_stale()
            
            # Should have cleaned up the deleted worktree
            # Note: The exact behavior depends on Git version
            # Some versions might not report the path in prune output
            worktrees = list_worktrees()
            paths = [wt["path"] for wt in worktrees]
            self.assertFalse(any(worktree_path1 in path for path in paths))
    
    def test_cleanup_stale_no_stale_worktrees(self):
        """Test cleanup when no stale worktrees exist."""
        with TestGitRepository():
            # Create a healthy worktree
            create_worktree("healthy", "worktrees/healthy")
            
            # Run cleanup
            removed = cleanup_stale()
            
            # Should not remove anything
            self.assertEqual(len(removed), 0)
            
            # Worktree should still exist
            worktrees = list_worktrees()
            self.assertGreaterEqual(len(worktrees), 2)  # Main + healthy
    
    def test_cleanup_stale_with_custom_repo_path(self):
        """Test cleanup with custom repository path."""
        with TestGitRepository() as repo_dir:
            # Create worktree
            create_worktree("test", "worktrees/test")
            
            # Change to parent directory
            parent_dir = Path(repo_dir).parent
            os.chdir(parent_dir)
            
            # Run cleanup with repo path
            removed = cleanup_stale(repo_path=repo_dir)
            
            # Should work without errors
            self.assertIsInstance(removed, list)


class TestListWorktrees(unittest.TestCase):
    """Test worktree listing functionality."""
    
    def test_list_worktrees_empty(self):
        """Test listing worktrees in fresh repository."""
        with TestGitRepository():
            worktrees = list_worktrees()
            
            # Should have at least the main worktree
            self.assertGreaterEqual(len(worktrees), 1)
            
            # Main worktree should be present
            main_found = False
            for wt in worktrees:
                if wt.get("branch") in ["refs/heads/main", "refs/heads/master"]:
                    main_found = True
                    break
            self.assertTrue(main_found)
    
    def test_list_worktrees_multiple(self):
        """Test listing multiple worktrees."""
        with TestGitRepository():
            # Create multiple worktrees
            create_worktree("branch1", "worktrees/wt1")
            create_worktree("branch2", "worktrees/wt2")
            create_worktree("branch3", "worktrees/wt3")
            
            worktrees = list_worktrees()
            
            # Should have main + 3 created worktrees
            self.assertGreaterEqual(len(worktrees), 4)
            
            # Check all worktrees have required fields
            for wt in worktrees:
                self.assertIn("path", wt)
                self.assertIn("branch", wt)
                self.assertIn("commit", wt)
                self.assertIn("bare", wt)
    
    def test_list_worktrees_detached_head(self):
        """Test listing worktree with detached HEAD."""
        with TestGitRepository():
            # Create worktree
            create_worktree("test-branch", "worktrees/test")
            
            # Detach HEAD in the worktree
            subprocess.run(
                ["git", "checkout", "HEAD~0"],
                cwd="worktrees/test",
                check=True,
                capture_output=True
            )
            
            worktrees = list_worktrees()
            
            # Find the detached worktree
            detached_found = False
            for wt in worktrees:
                if "worktrees/test" in wt["path"]:
                    self.assertEqual(wt["branch"], "(detached)")
                    detached_found = True
                    break
            self.assertTrue(detached_found)
    
    def test_list_worktrees_with_custom_repo_path(self):
        """Test listing worktrees with custom repository path."""
        with TestGitRepository() as repo_dir:
            # Create worktree
            create_worktree("test", "worktrees/test")
            
            # Change to parent directory
            parent_dir = Path(repo_dir).parent
            os.chdir(parent_dir)
            
            # List worktrees with repo path
            worktrees = list_worktrees(repo_path=repo_dir)
            
            # Should find worktrees
            self.assertGreaterEqual(len(worktrees), 2)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of worktree operations."""
    
    def test_concurrent_worktree_creation(self):
        """Test creating worktrees concurrently."""
        with TestGitRepository():
            errors = []
            
            def create_worktree_thread(index):
                try:
                    branch = f"concurrent-{index}"
                    path = f"worktrees/concurrent-{index}"
                    create_worktree(branch, path)
                except Exception as e:
                    errors.append(e)
            
            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=create_worktree_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Check no errors occurred
            self.assertEqual(len(errors), 0)
            
            # Verify all worktrees were created
            worktrees = list_worktrees()
            created_count = sum(1 for wt in worktrees if "concurrent-" in wt.get("branch", ""))
            self.assertEqual(created_count, 5)
    
    def test_concurrent_mixed_operations(self):
        """Test mixed operations running concurrently."""
        with TestGitRepository():
            # Create some worktrees first
            for i in range(3):
                create_worktree(f"mixed-{i}", f"worktrees/mixed-{i}")
            
            errors = []
            results = {}
            
            def mixed_operations(operation, index):
                try:
                    if operation == "create":
                        create_worktree(f"new-{index}", f"worktrees/new-{index}")
                    elif operation == "remove":
                        remove_worktree(f"worktrees/mixed-{index}")
                    elif operation == "list":
                        results[f"list-{index}"] = list_worktrees()
                    elif operation == "cleanup":
                        results[f"cleanup-{index}"] = cleanup_stale()
                except Exception as e:
                    errors.append((operation, index, e))
            
            # Run mixed operations concurrently
            threads = []
            operations = [
                ("create", 0),
                ("remove", 0),
                ("list", 0),
                ("create", 1),
                ("remove", 1),
                ("cleanup", 0),
                ("list", 1),
            ]
            
            for op, idx in operations:
                thread = threading.Thread(target=mixed_operations, args=(op, idx))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Should complete without critical errors
            # Some operations might fail due to timing, but no corruption
            assert all(not isinstance(e[2], (OSError, IOError)) for e in errors)


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def test_git_not_installed(self):
        """Test behavior when git is not available."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            
            with pytest.raises(Exception):
                _run_git_command(["status"])
    
    def test_not_in_git_repository(self):
        """Test operations outside a git repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            
            with pytest.raises(WorktreeError):
                list_worktrees()
    
    def test_corrupted_worktree_handling(self):
        """Test handling of corrupted worktree metadata."""
        with TestGitRepository():
            # Create a worktree
            create_worktree("test", "worktrees/test")
            
            # Corrupt the worktree by removing .git file
            git_file = Path("worktrees/test/.git")
            if git_file.exists():
                git_file.unlink()
            
            # Operations should handle this gracefully
            worktrees = list_worktrees()
            assert isinstance(worktrees, list)
            
            # Cleanup should detect and handle it
            cleanup_stale()


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self):
        """Test a complete worktree lifecycle."""
        with TestGitRepository():
            branch = "cc-parallel/integration-test"
            path = "worktrees/integration"
            
            # Create worktree
            create_worktree(branch, path)
            
            # Verify it exists
            worktrees = list_worktrees()
            assert any(path in wt["path"] for wt in worktrees)
            
            # Make changes in the worktree
            test_file = Path(path) / "test.txt"
            test_file.write_text("Integration test")
            
            subprocess.run(
                ["git", "add", "test.txt"],
                cwd=path,
                check=True,
                capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", "Integration test commit"],
                cwd=path,
                check=True,
                capture_output=True
            )
            
            # Remove worktree
            remove_worktree(path)
            
            # Verify it's gone
            worktrees = list_worktrees()
            assert not any(path in wt["path"] for wt in worktrees)
            
            # Branch should still exist
            result = subprocess.run(
                ["git", "branch", "--list", branch],
                capture_output=True,
                text=True
            )
            assert branch in result.stdout
    
    def test_poc_dag_integration_pattern(self):
        """Test the pattern used in the PoC DAG."""
        with TestGitRepository():
            # Simulate the PoC DAG pattern
            shard_id = "todo-001"
            worktree_path = f"/tmp/worktrees/{shard_id}"
            branch_name = f"cc-parallel/{shard_id}"
            
            # Create worktree (as in process_shard)
            create_worktree(branch_name, worktree_path)
            
            # Verify creation
            assert Path(worktree_path).exists()
            
            # Simulate work in the worktree
            readme = Path(worktree_path) / "README.md"
            self.assertTrue(readme.exists())  # Should have the initial commit file
            
            # Remove worktree (as in process_shard cleanup)
            remove_worktree(worktree_path)
            
            # Verify removal
            self.assertFalse(Path(worktree_path).exists())


if __name__ == "__main__":
    unittest.main()