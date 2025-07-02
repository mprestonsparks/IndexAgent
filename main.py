#!/usr/bin/env python3
"""
IndexAgent Main API Server

Provides REST API endpoints for code indexing, search, and maintenance operations.
Integrates with Sourcebot and Zoekt for comprehensive code management.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import local modules
from src.models import DatabaseManager
from indexagent.utils.worktree_manager import WorktreeManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
db_manager: Optional[DatabaseManager] = None
worktree_manager: Optional[WorktreeManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup
    logger.info("Starting IndexAgent API Server...")
    await initialize_services()
    logger.info("IndexAgent API Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down IndexAgent API Server...")
    await cleanup_services()
    logger.info("IndexAgent API Server shut down")

async def initialize_services():
    """Initialize all core services."""
    global db_manager, worktree_manager
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Initialize worktree manager
        repos_path = os.getenv('REPOS_PATH', '/repos')
        worktree_manager = WorktreeManager(base_path=repos_path)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

async def cleanup_services():
    """Cleanup all services."""
    global db_manager, worktree_manager
    
    if worktree_manager:
        try:
            await worktree_manager.cleanup_all()
        except Exception as e:
            logger.error(f"Error cleaning up worktree manager: {e}")
    
    if db_manager:
        try:
            await db_manager.close()
        except Exception as e:
            logger.error(f"Error closing database: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="IndexAgent API",
    description="Code indexing and search API with Sourcebot/Zoekt integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class RepositoryInfo(BaseModel):
    """Repository information model."""
    name: str
    path: str
    last_indexed: Optional[str] = None
    file_count: int = 0
    size_mb: float = 0.0

class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query")
    repositories: Optional[List[str]] = Field(default=None, description="Repositories to search")
    file_types: Optional[List[str]] = Field(default=None, description="File extensions to include")
    max_results: int = Field(default=100, ge=1, le=1000, description="Maximum results to return")

class SearchResult(BaseModel):
    """Search result model."""
    repository: str
    file_path: str
    line_number: int
    content: str
    score: float

class IndexRequest(BaseModel):
    """Indexing request model."""
    repositories: Optional[List[str]] = Field(default=None, description="Repositories to index")
    force_reindex: bool = Field(default=False, description="Force re-indexing")

class WorktreeRequest(BaseModel):
    """Worktree creation request."""
    repository: str
    branch: str = "main"
    task_id: Optional[str] = None

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "IndexAgent API",
        "version": "1.0.0",
        "components": {
            "database": "connected" if db_manager else "disconnected",
            "worktree_manager": "active" if worktree_manager else "inactive"
        }
    }

@app.get("/repositories", response_model=List[RepositoryInfo])
async def list_repositories():
    """List all available repositories."""
    try:
        repos_path = Path(os.getenv('REPOS_PATH', '/repos'))
        repositories = []
        
        if repos_path.exists():
            for repo_dir in repos_path.iterdir():
                if repo_dir.is_dir() and (repo_dir / '.git').exists():
                    # Get repository info
                    file_count = sum(1 for _ in repo_dir.rglob('*') if _.is_file())
                    size_mb = sum(f.stat().st_size for f in repo_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                    
                    repositories.append(RepositoryInfo(
                        name=repo_dir.name,
                        path=str(repo_dir),
                        file_count=file_count,
                        size_mb=round(size_mb, 2)
                    ))
        
        return repositories
        
    except Exception as e:
        logger.error(f"Failed to list repositories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list repositories: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_code(request: SearchRequest):
    """Search code across indexed repositories."""
    try:
        # For now, return mock results
        # In a real implementation, this would integrate with Zoekt
        mock_results = [
            SearchResult(
                repository="example-repo",
                file_path="src/main.py",
                line_number=42,
                content=f"def search_function(): # Contains: {request.query}",
                score=0.95
            ),
            SearchResult(
                repository="example-repo", 
                file_path="tests/test_search.py",
                line_number=15,
                content=f"assert search_result == '{request.query}'",
                score=0.87
            )
        ]
        
        # Filter by repositories if specified
        if request.repositories:
            mock_results = [r for r in mock_results if r.repository in request.repositories]
        
        # Limit results
        return mock_results[:request.max_results]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/index")
async def trigger_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """Trigger indexing of repositories."""
    try:
        # Add background task for indexing
        background_tasks.add_task(
            perform_indexing,
            repositories=request.repositories,
            force_reindex=request.force_reindex
        )
        
        return {
            "message": "Indexing started",
            "repositories": request.repositories or "all",
            "force_reindex": request.force_reindex
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger indexing: {str(e)}")

async def perform_indexing(repositories: Optional[List[str]] = None, force_reindex: bool = False):
    """Background task to perform indexing."""
    try:
        logger.info(f"Starting indexing: repositories={repositories}, force_reindex={force_reindex}")
        
        # In a real implementation, this would:
        # 1. Call Zoekt indexing service
        # 2. Update database with indexing status
        # 3. Handle errors and retries
        
        # Simulate indexing work
        import asyncio
        await asyncio.sleep(5)  # Simulate indexing time
        
        logger.info("Indexing completed successfully")
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")

@app.post("/worktrees")
async def create_worktree(request: WorktreeRequest):
    """Create a new git worktree."""
    try:
        if not worktree_manager:
            raise HTTPException(status_code=503, detail="Worktree manager not available")
        
        worktree_path = await worktree_manager.create_worktree(
            repository=request.repository,
            branch=request.branch,
            task_id=request.task_id
        )
        
        return {
            "worktree_path": str(worktree_path),
            "repository": request.repository,
            "branch": request.branch,
            "task_id": request.task_id
        }
        
    except Exception as e:
        logger.error(f"Failed to create worktree: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create worktree: {str(e)}")

@app.get("/worktrees")
async def list_worktrees():
    """List all active worktrees."""
    try:
        if not worktree_manager:
            raise HTTPException(status_code=503, detail="Worktree manager not available")
        
        worktrees = await worktree_manager.list_worktrees()
        
        return {
            "worktrees": [
                {
                    "path": str(wt.path),
                    "repository": wt.repository,
                    "branch": wt.branch,
                    "task_id": wt.task_id,
                    "created_at": wt.created_at.isoformat() if wt.created_at else None
                }
                for wt in worktrees
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to list worktrees: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list worktrees: {str(e)}")

@app.delete("/worktrees/{task_id}")
async def cleanup_worktree(task_id: str):
    """Clean up a specific worktree."""
    try:
        if not worktree_manager:
            raise HTTPException(status_code=503, detail="Worktree manager not available")
        
        success = await worktree_manager.cleanup_worktree(task_id)
        
        if success:
            return {"message": f"Worktree for task {task_id} cleaned up successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Worktree for task {task_id} not found")
        
    except Exception as e:
        logger.error(f"Failed to cleanup worktree: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup worktree: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get system metrics and statistics."""
    try:
        repos_path = Path(os.getenv('REPOS_PATH', '/repos'))
        
        # Count repositories
        repo_count = 0
        total_files = 0
        total_size_mb = 0.0
        
        if repos_path.exists():
            for repo_dir in repos_path.iterdir():
                if repo_dir.is_dir() and (repo_dir / '.git').exists():
                    repo_count += 1
                    files = list(repo_dir.rglob('*'))
                    file_count = sum(1 for f in files if f.is_file())
                    size_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
                    total_files += file_count
                    total_size_mb += size_mb
        
        # Get worktree metrics
        worktree_count = 0
        if worktree_manager:
            try:
                worktrees = await worktree_manager.list_worktrees()
                worktree_count = len(worktrees)
            except:
                pass
        
        return {
            "repositories": {
                "count": repo_count,
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2)
            },
            "worktrees": {
                "active_count": worktree_count
            },
            "system": {
                "status": "operational",
                "uptime": "unknown"  # Would track actual uptime
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )