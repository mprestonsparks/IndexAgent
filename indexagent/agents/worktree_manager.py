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


import asyncio
from datetime import datetime


class GitWorktreeManager:
    """REAL git worktree management using git commands for DEAN agent evolution"""
    
    def __init__(self, base_path: Path):
        """Initialize worktree manager with base path for all worktrees"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._active_worktrees: Dict[str, Path] = {}
        
    async def create_worktree(self, branch_name: str, agent_id: str, token_limit: int) -> Path:
        """Create actual git worktree, not just a directory
        
        Args:
            branch_name: Git branch name for the worktree
            agent_id: Unique agent identifier
            token_limit: Token budget limit (stored as metadata)
            
        Returns:
            Path to the created worktree
            
        Raises:
            WorktreeCreationError: If worktree creation fails
        """
        worktree_path = self.base_path / agent_id
        
        # Create branch name with timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_branch_name = f"dean/{branch_name}/{agent_id}/{timestamp}"
        
        try:
            # Execute real git command asynchronously from base_path
            proc = await asyncio.create_subprocess_exec(
                'git', 'worktree', 'add', 
                str(worktree_path), 
                '-b', full_branch_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.base_path)  # Run from base_path which should be a git repo
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                raise WorktreeCreationError(f"Git worktree creation failed: {stderr.decode()}")
            
            # Store metadata about the worktree
            metadata_file = worktree_path / ".dean_metadata.json"
            metadata = {
                "agent_id": agent_id,
                "branch_name": full_branch_name,
                "token_limit": token_limit,
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }
            
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            # Track active worktree
            self._active_worktrees[agent_id] = worktree_path
            
            logger.info(f"Created worktree for agent {agent_id} at {worktree_path}")
            return worktree_path
            
        except Exception as e:
            logger.error(f"Failed to create worktree for agent {agent_id}: {e}")
            raise WorktreeCreationError(str(e))
    
    async def remove_worktree(self, agent_id: str, force: bool = False) -> None:
        """Remove a worktree associated with an agent
        
        Args:
            agent_id: Agent identifier
            force: Force removal even if worktree has uncommitted changes
        """
        worktree_path = self._active_worktrees.get(agent_id)
        if not worktree_path:
            worktree_path = self.base_path / agent_id
            
        if not worktree_path.exists():
            logger.warning(f"Worktree for agent {agent_id} not found at {worktree_path}")
            return
            
        try:
            # Build command
            cmd = ['git', 'worktree', 'remove']
            if force:
                cmd.append('--force')
            cmd.append(str(worktree_path))
            
            # Execute removal
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0 and not force:
                # Retry with force if normal removal failed
                logger.warning(f"Normal removal failed, retrying with force: {stderr.decode()}")
                await self.remove_worktree(agent_id, force=True)
                return
                
            if proc.returncode != 0:
                raise WorktreeRemovalError(f"Failed to remove worktree: {stderr.decode()}")
                
            # Remove from tracking
            if agent_id in self._active_worktrees:
                del self._active_worktrees[agent_id]
                
            logger.info(f"Removed worktree for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error removing worktree for agent {agent_id}: {e}")
            raise WorktreeRemovalError(str(e))
    
    async def get_worktree_path(self, agent_id: str) -> Optional[Path]:
        """Get the path to an agent's worktree
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Path to worktree if it exists, None otherwise
        """
        path = self._active_worktrees.get(agent_id)
        if path and path.exists():
            return path
            
        # Check if worktree exists but isn't tracked
        potential_path = self.base_path / agent_id
        if potential_path.exists() and (potential_path / ".git").exists():
            self._active_worktrees[agent_id] = potential_path
            return potential_path
            
        return None
    
    async def list_agent_worktrees(self) -> List[Dict[str, any]]:
        """List all worktrees managed by this manager
        
        Returns:
            List of worktree information dictionaries
        """
        worktrees = []
        
        # Use the existing list_worktrees function
        all_worktrees = list_worktrees()
        
        for wt in all_worktrees:
            path = Path(wt['path'])
            # Check if this worktree is under our management
            if str(path).startswith(str(self.base_path)):
                # Try to load metadata
                metadata_file = path / ".dean_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            import json
                            metadata = json.load(f)
                            wt['dean_metadata'] = metadata
                            worktrees.append(wt)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {path}: {e}")
                        
        return worktrees
    
    async def cleanup_stale_worktrees(self) -> List[str]:
        """Clean up stale worktrees and return list of removed agent IDs
        
        Returns:
            List of agent IDs whose worktrees were cleaned up
        """
        removed_agent_ids = []
        
        # Get all worktrees under our base path
        for item in self.base_path.iterdir():
            if item.is_dir() and (item / ".git").exists():
                # Check if this is a stale worktree
                try:
                    # Try to run git status to check if worktree is valid
                    proc = await asyncio.create_subprocess_exec(
                        'git', 'status',
                        cwd=str(item),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    
                    if proc.returncode != 0:
                        # Worktree is stale
                        agent_id = item.name
                        logger.info(f"Found stale worktree for agent {agent_id}")
                        try:
                            await self.remove_worktree(agent_id, force=True)
                            removed_agent_ids.append(agent_id)
                        except Exception as e:
                            logger.error(f"Failed to remove stale worktree {agent_id}: {e}")
                            
                except Exception as e:
                    logger.error(f"Error checking worktree {item}: {e}")
                    
        # Also run git worktree prune
        try:
            proc = await asyncio.create_subprocess_exec(
                'git', 'worktree', 'prune',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
        except Exception as e:
            logger.error(f"Failed to prune worktrees: {e}")
            
        return removed_agent_ids
    
    async def get_worktree_status(self, agent_id: str) -> Optional[Dict[str, any]]:
        """Get status information about an agent's worktree
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary with status information or None if worktree doesn't exist
        """
        worktree_path = await self.get_worktree_path(agent_id)
        if not worktree_path:
            return None
            
        try:
            # Get git status
            proc = await asyncio.create_subprocess_exec(
                'git', 'status', '--porcelain',
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                return None
                
            # Count modified files
            modified_files = len([line for line in stdout.decode().splitlines() if line.strip()])
            
            # Get current branch
            proc = await asyncio.create_subprocess_exec(
                'git', 'branch', '--show-current',
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            current_branch = stdout.decode().strip()
            
            # Load metadata if available
            metadata = {}
            metadata_file = worktree_path / ".dean_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        import json
                        metadata = json.load(f)
                except Exception:
                    pass
                    
            return {
                "agent_id": agent_id,
                "path": str(worktree_path),
                "branch": current_branch,
                "modified_files": modified_files,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get worktree status for {agent_id}: {e}")
            return None