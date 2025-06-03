"""
Git worktree management utilities for parallel processing.

This module provides thread-safe operations for creating, managing, and cleaning up
Git worktrees used in parallel Claude Code invocations.
"""

import logging
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Thread lock for worktree operations
_worktree_lock = threading.Lock()


class WorktreeError(Exception):
    """Base exception for worktree operations."""
    pass


class WorktreeCreationError(WorktreeError):
    """Raised when worktree creation fails."""
    pass


class WorktreeRemovalError(WorktreeError):
    """Raised when worktree removal fails."""
    pass


def _run_git_command(cmd: List[str], cwd: str = ".") -> subprocess.CompletedProcess:
    """
    Execute a git command and return the result.
    
    Args:
        cmd: Git command as list of arguments
        cwd: Working directory for the command
        
    Returns:
        CompletedProcess instance
        
    Raises:
        WorktreeError: If git command fails
    """
    try:
        result = subprocess.run(
            ["git"] + cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {' '.join(['git'] + cmd)}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise WorktreeError(f"Git command failed: {e.stderr}") from e


def _validate_branch_name(branch: str) -> None:
    """
    Validate that a branch name follows Git conventions.
    
    Args:
        branch: Branch name to validate
        
    Raises:
        ValueError: If branch name is invalid
    """
    if not branch:
        raise ValueError("Branch name cannot be empty")
    
    # Check for invalid characters
    invalid_chars = r'[\s~^:?*\[\]\\]'
    if re.search(invalid_chars, branch):
        raise ValueError(f"Branch name contains invalid characters: {branch}")
    
    # Check for invalid patterns
    if branch.startswith('.') or branch.endswith('.'):
        raise ValueError("Branch name cannot start or end with a dot")
    
    if branch.endswith('.lock'):
        raise ValueError("Branch name cannot end with .lock")
    
    if '//' in branch:
        raise ValueError("Branch name cannot contain consecutive slashes")


def _validate_path(path: str) -> None:
    """
    Validate that a path is safe to use.
    
    Args:
        path: Path to validate
        
    Raises:
        ValueError: If path is invalid
    """
    if not path:
        raise ValueError("Path cannot be empty")
    
    # Convert to Path object for validation
    path_obj = Path(path)
    
    # Check if path is absolute when it shouldn't be
    if path_obj.is_absolute() and not path.startswith('/tmp/'):
        raise ValueError("Path must be relative or under /tmp/")


def create_worktree(branch: str, path: str, base_branch: str = "main") -> None:
    """
    Create a new worktree for the given branch at the specified path.
    
    This function creates a new Git worktree, checking out the specified branch.
    If the branch doesn't exist, it will be created from the base branch.
    Parent directories will be created automatically if they don't exist.
    
    Args:
        branch: Name of the branch to checkout in the worktree
        path: Path where the worktree should be created
        base_branch: Base branch to create from if branch doesn't exist (default: "main")
        
    Raises:
        ValueError: If branch name or path is invalid
        WorktreeCreationError: If worktree creation fails
        
    Example:
        >>> create_worktree("cc-parallel/todo-001", "/tmp/worktrees/todo-001")
        >>> # Creates worktree at /tmp/worktrees/todo-001 with branch cc-parallel/todo-001
    """
    # Validate inputs
    _validate_branch_name(branch)
    _validate_path(path)
    
    with _worktree_lock:
        logger.info(f"Creating worktree for branch '{branch}' at '{path}'")
        
        # Create parent directories if needed
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if path already exists
        if path_obj.exists():
            raise WorktreeCreationError(f"Path already exists: {path}")
        
        try:
            # Check if branch exists
            try:
                _run_git_command(["rev-parse", "--verify", f"refs/heads/{branch}"])
                branch_exists = True
            except WorktreeError:
                branch_exists = False
            
            if branch_exists:
                # Branch exists, create worktree with it
                _run_git_command(["worktree", "add", path, branch])
                logger.info(f"Created worktree at '{path}' with existing branch '{branch}'")
            else:
                # Branch doesn't exist, create it from base branch
                _run_git_command(["worktree", "add", "-b", branch, path, base_branch])
                logger.info(f"Created worktree at '{path}' with new branch '{branch}' from '{base_branch}'")
                
        except WorktreeError as e:
            raise WorktreeCreationError(f"Failed to create worktree: {e}") from e


def remove_worktree(path: str) -> None:
    """
    Remove a worktree and optionally delete its branch.
    
    This function removes a Git worktree at the specified path. It handles
    locked worktrees gracefully by forcing removal if necessary.
    
    Args:
        path: Path of the worktree to remove
        
    Raises:
        ValueError: If path is invalid
        WorktreeRemovalError: If worktree removal fails
        
    Example:
        >>> remove_worktree("/tmp/worktrees/todo-001")
        >>> # Removes the worktree at /tmp/worktrees/todo-001
    """
    _validate_path(path)
    
    with _worktree_lock:
        logger.info(f"Removing worktree at '{path}'")
        
        try:
            # First try normal removal
            _run_git_command(["worktree", "remove", path])
            logger.info(f"Successfully removed worktree at '{path}'")
        except WorktreeError:
            # If normal removal fails, try force removal
            logger.warning(f"Normal removal failed for '{path}', attempting force removal")
            try:
                _run_git_command(["worktree", "remove", "--force", path])
                logger.info(f"Successfully force-removed worktree at '{path}'")
            except WorktreeError:
                # If single force fails, try double force for locked worktrees
                logger.warning(f"Force removal failed for '{path}', attempting double-force removal")
                try:
                    _run_git_command(["worktree", "remove", "--force", "--force", path])
                    logger.info(f"Successfully double-force-removed worktree at '{path}'")
                except WorktreeError as e:
                    raise WorktreeRemovalError(f"Failed to remove worktree: {e}") from e


def cleanup_stale(repo_path: str = ".") -> List[str]:
    """
    Clean up stale worktrees and return list of removed paths.
    
    This function prunes worktrees that no longer have a valid working directory,
    such as those that were deleted manually or are on unmounted filesystems.
    
    Args:
        repo_path: Path to the Git repository (default: current directory)
        
    Returns:
        List of paths that were cleaned up
        
    Example:
        >>> removed = cleanup_stale()
        >>> print(f"Cleaned up {len(removed)} stale worktrees")
    """
    with _worktree_lock:
        logger.info(f"Cleaning up stale worktrees in '{repo_path}'")
        
        # Get list of worktrees before cleanup
        before = list_worktrees(repo_path)
        before_paths = {wt['path'] for wt in before}
        
        try:
            # Run prune command
            result = _run_git_command(["worktree", "prune", "-v"], cwd=repo_path)
            
            # Parse output to find removed worktrees
            removed_paths = []
            for line in result.stdout.splitlines():
                if line.startswith("Removing worktrees/"):
                    # Extract the worktree name from the message
                    parts = line.split(":")
                    if len(parts) > 1:
                        # The path is usually after the colon
                        path_part = parts[1].strip()
                        if path_part:
                            removed_paths.append(path_part)
            
            # If prune doesn't give us paths, compare before/after
            if not removed_paths:
                after = list_worktrees(repo_path)
                after_paths = {wt['path'] for wt in after}
                removed_paths = list(before_paths - after_paths)
            
            if removed_paths:
                logger.info(f"Cleaned up {len(removed_paths)} stale worktrees: {removed_paths}")
            else:
                logger.info("No stale worktrees found")
                
            return removed_paths
            
        except WorktreeError as e:
            logger.error(f"Failed to cleanup stale worktrees: {e}")
            return []


def list_worktrees(repo_path: str = ".") -> List[Dict[str, str]]:
    """
    List all active worktrees with their paths and branches.
    
    Args:
        repo_path: Path to the Git repository (default: current directory)
        
    Returns:
        List of dictionaries containing worktree information:
        - path: Absolute path to the worktree
        - branch: Branch name checked out in the worktree
        - commit: Current commit SHA
        - bare: Whether this is a bare worktree
        
    Example:
        >>> worktrees = list_worktrees()
        >>> for wt in worktrees:
        ...     print(f"{wt['branch']} -> {wt['path']}")
    """
    logger.info(f"Listing worktrees in '{repo_path}'")
    
    try:
        result = _run_git_command(["worktree", "list", "--porcelain"], cwd=repo_path)
        
        worktrees = []
        current_worktree: Dict[str, str] = {}
        
        for line in result.stdout.splitlines():
            if not line.strip():
                # Empty line marks end of a worktree entry
                if current_worktree:
                    worktrees.append(current_worktree)
                    current_worktree = {}
                continue
                
            if line.startswith("worktree "):
                current_worktree["path"] = line[9:].strip()
            elif line.startswith("HEAD "):
                current_worktree["commit"] = line[5:].strip()
            elif line.startswith("branch "):
                current_worktree["branch"] = line[7:].strip()
            elif line == "bare":
                current_worktree["bare"] = "true"
            elif line.startswith("detached"):
                current_worktree["branch"] = "(detached)"
        
        # Don't forget the last worktree
        if current_worktree:
            worktrees.append(current_worktree)
        
        # Ensure all worktrees have required fields
        for wt in worktrees:
            wt.setdefault("branch", "(none)")
            wt.setdefault("commit", "")
            wt.setdefault("bare", "false")
        
        logger.info(f"Found {len(worktrees)} worktrees")
        return worktrees
        
    except WorktreeError as e:
        logger.error(f"Failed to list worktrees: {e}")
        return []