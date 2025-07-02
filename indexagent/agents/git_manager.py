#!/usr/bin/env python3
"""
Git Worktree Manager for DEAN Agents
Provides isolated git worktrees for each agent's code evolution
"""

import os
import asyncio
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import subprocess
import json

logger = logging.getLogger(__name__)

class GitWorktreeManager:
    """
    Manages isolated git worktrees for agents.
    Each agent gets its own worktree to prevent conflicts.
    """
    
    def __init__(self, base_repo_path: str, worktree_base_path: str):
        """
        Initialize worktree manager.
        
        Args:
            base_repo_path: Path to main git repository
            worktree_base_path: Directory where worktrees will be created
        """
        self.base_repo_path = Path(base_repo_path)
        self.worktree_base_path = Path(worktree_base_path)
        
        # Ensure directories exist
        self.worktree_base_path.mkdir(parents=True, exist_ok=True)
        
        # Verify base repo is a git repository
        if not (self.base_repo_path / ".git").exists():
            raise ValueError(f"{base_repo_path} is not a git repository")
        
        # Track active worktrees
        self.active_worktrees: Dict[str, Path] = {}
        
    async def create_worktree(self, agent_id: str, branch_name: Optional[str] = None) -> Path:
        """
        Create isolated worktree for an agent.
        
        Args:
            agent_id: Unique agent identifier
            branch_name: Optional branch name (defaults to agent-{agent_id})
            
        Returns:
            Path to the created worktree
        """
        if agent_id in self.active_worktrees:
            logger.info(f"Worktree already exists for agent {agent_id}")
            return self.active_worktrees[agent_id]
        
        # Generate branch name if not provided
        if not branch_name:
            branch_name = f"agent-{agent_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create worktree path
        worktree_path = self.worktree_base_path / agent_id
        
        try:
            # Create new branch and worktree
            cmd = [
                "git", "worktree", "add",
                "-b", branch_name,
                str(worktree_path),
                "HEAD"
            ]
            
            result = await self._run_git_command(cmd, cwd=self.base_repo_path)
            
            if result[0] == 0:
                self.active_worktrees[agent_id] = worktree_path
                logger.info(f"Created worktree for agent {agent_id} at {worktree_path}")
                
                # Initialize agent metadata
                await self._init_agent_metadata(agent_id, worktree_path, branch_name)
                
                return worktree_path
            else:
                raise RuntimeError(f"Failed to create worktree: {result[2]}")
                
        except Exception as e:
            logger.error(f"Error creating worktree for agent {agent_id}: {e}")
            raise
    
    async def remove_worktree(self, agent_id: str, force: bool = False) -> bool:
        """
        Remove agent's worktree.
        
        Args:
            agent_id: Agent identifier
            force: Force removal even with uncommitted changes
            
        Returns:
            True if successful
        """
        if agent_id not in self.active_worktrees:
            logger.warning(f"No worktree found for agent {agent_id}")
            return False
        
        worktree_path = self.active_worktrees[agent_id]
        
        try:
            # Check for uncommitted changes
            if not force:
                status = await self.get_status(agent_id)
                if status.get("has_changes"):
                    raise ValueError("Worktree has uncommitted changes. Use force=True to remove anyway.")
            
            # Remove worktree
            cmd = ["git", "worktree", "remove", str(worktree_path)]
            if force:
                cmd.append("--force")
            
            result = await self._run_git_command(cmd, cwd=self.base_repo_path)
            
            if result[0] == 0:
                del self.active_worktrees[agent_id]
                logger.info(f"Removed worktree for agent {agent_id}")
                return True
            else:
                logger.error(f"Failed to remove worktree: {result[2]}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing worktree for agent {agent_id}: {e}")
            return False
    
    async def commit_changes(self, agent_id: str, message: str, 
                           files: Optional[List[str]] = None) -> Optional[str]:
        """
        Commit changes in agent's worktree.
        
        Args:
            agent_id: Agent identifier
            message: Commit message
            files: Specific files to commit (None = all changes)
            
        Returns:
            Commit hash if successful
        """
        if agent_id not in self.active_worktrees:
            raise ValueError(f"No worktree found for agent {agent_id}")
        
        worktree_path = self.active_worktrees[agent_id]
        
        try:
            # Stage files
            if files:
                for file in files:
                    cmd = ["git", "add", file]
                    await self._run_git_command(cmd, cwd=worktree_path)
            else:
                # Stage all changes
                cmd = ["git", "add", "-A"]
                await self._run_git_command(cmd, cwd=worktree_path)
            
            # Create commit
            cmd = ["git", "commit", "-m", message]
            result = await self._run_git_command(cmd, cwd=worktree_path)
            
            if result[0] == 0:
                # Get commit hash
                cmd = ["git", "rev-parse", "HEAD"]
                hash_result = await self._run_git_command(cmd, cwd=worktree_path)
                
                if hash_result[0] == 0:
                    commit_hash = hash_result[1].strip()
                    logger.info(f"Agent {agent_id} committed changes: {commit_hash[:8]}")
                    return commit_hash
            
            return None
            
        except Exception as e:
            logger.error(f"Error committing changes for agent {agent_id}: {e}")
            return None
    
    async def get_status(self, agent_id: str) -> Dict[str, any]:
        """
        Get git status for agent's worktree.
        
        Returns:
            Dictionary with status information
        """
        if agent_id not in self.active_worktrees:
            raise ValueError(f"No worktree found for agent {agent_id}")
        
        worktree_path = self.active_worktrees[agent_id]
        
        try:
            # Get current branch
            cmd = ["git", "branch", "--show-current"]
            branch_result = await self._run_git_command(cmd, cwd=worktree_path)
            current_branch = branch_result[1].strip() if branch_result[0] == 0 else "unknown"
            
            # Get status
            cmd = ["git", "status", "--porcelain"]
            status_result = await self._run_git_command(cmd, cwd=worktree_path)
            
            if status_result[0] == 0:
                lines = status_result[1].strip().split('\n') if status_result[1].strip() else []
                
                modified_files = []
                untracked_files = []
                
                for line in lines:
                    if line.startswith(' M'):
                        modified_files.append(line[3:])
                    elif line.startswith('??'):
                        untracked_files.append(line[3:])
                
                return {
                    "branch": current_branch,
                    "has_changes": len(lines) > 0,
                    "modified_files": modified_files,
                    "untracked_files": untracked_files,
                    "total_changes": len(lines)
                }
            
            return {"error": "Failed to get status"}
            
        except Exception as e:
            logger.error(f"Error getting status for agent {agent_id}: {e}")
            return {"error": str(e)}
    
    async def sync_with_main(self, agent_id: str) -> bool:
        """
        Sync agent's branch with main branch.
        
        Returns:
            True if successful
        """
        if agent_id not in self.active_worktrees:
            raise ValueError(f"No worktree found for agent {agent_id}")
        
        worktree_path = self.active_worktrees[agent_id]
        
        try:
            # Fetch latest changes
            cmd = ["git", "fetch", "origin", "main"]
            result = await self._run_git_command(cmd, cwd=worktree_path)
            
            if result[0] == 0:
                # Merge main into current branch
                cmd = ["git", "merge", "origin/main", "--no-edit"]
                merge_result = await self._run_git_command(cmd, cwd=worktree_path)
                
                if merge_result[0] == 0:
                    logger.info(f"Agent {agent_id} synced with main branch")
                    return True
                else:
                    logger.warning(f"Merge conflicts for agent {agent_id}: {merge_result[2]}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error syncing agent {agent_id} with main: {e}")
            return False
    
    async def get_diff(self, agent_id: str, target_branch: str = "main") -> str:
        """
        Get diff between agent's branch and target branch.
        
        Returns:
            Diff output
        """
        if agent_id not in self.active_worktrees:
            raise ValueError(f"No worktree found for agent {agent_id}")
        
        worktree_path = self.active_worktrees[agent_id]
        
        try:
            cmd = ["git", "diff", f"origin/{target_branch}...HEAD"]
            result = await self._run_git_command(cmd, cwd=worktree_path)
            
            if result[0] == 0:
                return result[1]
            else:
                return f"Error getting diff: {result[2]}"
                
        except Exception as e:
            logger.error(f"Error getting diff for agent {agent_id}: {e}")
            return f"Error: {str(e)}"
    
    async def cleanup_inactive_worktrees(self, max_age_hours: int = 24) -> int:
        """
        Clean up worktrees that haven't been used recently.
        
        Returns:
            Number of worktrees cleaned up
        """
        cleaned = 0
        
        try:
            # List all worktrees
            cmd = ["git", "worktree", "list", "--porcelain"]
            result = await self._run_git_command(cmd, cwd=self.base_repo_path)
            
            if result[0] == 0:
                # Parse worktree list
                current_time = datetime.now()
                
                for worktree_info in self._parse_worktree_list(result[1]):
                    worktree_path = Path(worktree_info["worktree"])
                    
                    # Check last modification time
                    if worktree_path.exists():
                        mtime = datetime.fromtimestamp(worktree_path.stat().st_mtime)
                        age_hours = (current_time - mtime).total_seconds() / 3600
                        
                        if age_hours > max_age_hours:
                            # Find agent_id for this worktree
                            agent_id = None
                            for aid, path in self.active_worktrees.items():
                                if path == worktree_path:
                                    agent_id = aid
                                    break
                            
                            if agent_id:
                                if await self.remove_worktree(agent_id, force=True):
                                    cleaned += 1
                                    logger.info(f"Cleaned up inactive worktree for agent {agent_id}")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning up worktrees: {e}")
            return cleaned
    
    async def _init_agent_metadata(self, agent_id: str, worktree_path: Path, branch_name: str):
        """Initialize agent metadata in worktree."""
        metadata = {
            "agent_id": agent_id,
            "branch": branch_name,
            "created_at": datetime.now().isoformat(),
            "worktree_path": str(worktree_path)
        }
        
        metadata_path = worktree_path / ".agent_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Commit metadata
        await self.commit_changes(agent_id, f"Initialize agent {agent_id} worktree", [".agent_metadata.json"])
    
    async def _run_git_command(self, cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
        """
        Run git command asynchronously.
        
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        
        return (
            proc.returncode,
            stdout.decode('utf-8'),
            stderr.decode('utf-8')
        )
    
    def _parse_worktree_list(self, output: str) -> List[Dict[str, str]]:
        """Parse git worktree list output."""
        worktrees = []
        current = {}
        
        for line in output.strip().split('\n'):
            if line.startswith('worktree '):
                if current:
                    worktrees.append(current)
                current = {"worktree": line[9:]}
            elif line.startswith('HEAD '):
                current["HEAD"] = line[5:]
            elif line.startswith('branch '):
                current["branch"] = line[7:]
        
        if current:
            worktrees.append(current)
        
        return worktrees
    
    def get_active_worktrees(self) -> Dict[str, Path]:
        """Get dictionary of active worktrees."""
        return self.active_worktrees.copy()