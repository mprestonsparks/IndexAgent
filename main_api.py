#!/usr/bin/env python3
"""
IndexAgent API Service - Core Agent Operations API
Port 8081 per DEAN specifications.

This service provides the primary API for agent management, pattern detection,
and core IndexAgent functionality.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from indexagent.agents.base_agent import FractalAgent, AgentGenome, TokenBudget
    from indexagent.agents.evolution.cellular_automata import CellularAutomataEngine, CARule
    from indexagent.agents.economy.token_manager import TokenEconomyManager
    from indexagent.agents.evolution.diversity_manager import GeneticDiversityManager
    from indexagent.agents.patterns.monitor import EmergentBehaviorMonitor
    from indexagent.agents.patterns.detector import PatternDetector
    from indexagent.agents.patterns.classifier import BehaviorClassifier
    from indexagent.agents.evolution.dspy_optimizer import DEANOptimizer
    from indexagent.agents.worktree_manager import GitWorktreeManager
except ImportError as e:
    logging.error(f"Failed to import IndexAgent components: {e}")
    # Provide fallback implementations
    class FractalAgent:
        def __init__(self, **kwargs):
            self.id = kwargs.get('id', 'fallback')
            self.name = kwargs.get('name', 'Fallback Agent')
    
    class TokenEconomyManager:
        def __init__(self, *args, **kwargs):
            pass
    
    class PatternDetector:
        def __init__(self):
            pass
    
    class EmergentBehaviorMonitor:
        def __init__(self):
            pass
    
    class DEANOptimizer:
        def __init__(self):
            pass
    
    class GitWorktreeManager:
        def __init__(self):
            pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
token_economy: Optional[TokenEconomyManager] = None
pattern_detector: Optional[PatternDetector] = None
behavior_monitor: Optional[EmergentBehaviorMonitor] = None
dspy_optimizer: Optional[DEANOptimizer] = None
worktree_manager: Optional[GitWorktreeManager] = None

# Active agents registry
active_agents: Dict[str, FractalAgent] = {}
agent_worktrees: Dict[str, str] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    await initialize_indexagent_services()
    yield
    # Shutdown
    await cleanup_indexagent_services()

async def initialize_indexagent_services():
    """Initialize IndexAgent core services."""
    global token_economy, pattern_detector, behavior_monitor, dspy_optimizer, worktree_manager
    
    try:
        logger.info("Initializing IndexAgent services...")
        
        # Initialize token economy (IndexAgent gets 50K budget allocation)
        token_economy = TokenEconomyManager(global_budget=50000)
        
        # Initialize pattern detection services
        pattern_detector = PatternDetector()
        behavior_monitor = EmergentBehaviorMonitor()
        
        # Initialize DSPy optimizer
        if 'DEANOptimizer' in globals():
            dspy_optimizer = DEANOptimizer()
        
        # Initialize worktree manager
        if 'GitWorktreeManager' in globals():
            worktree_manager = GitWorktreeManager()
        
        logger.info("IndexAgent services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize IndexAgent services: {e}")

async def cleanup_indexagent_services():
    """Cleanup IndexAgent services and agents."""
    global active_agents, agent_worktrees
    
    logger.info("Cleaning up IndexAgent services...")
    
    # Cleanup worktrees
    if worktree_manager:
        for worktree_path in agent_worktrees.values():
            try:
                worktree_manager.remove_worktree(worktree_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup worktree {worktree_path}: {e}")
    
    active_agents.clear()
    agent_worktrees.clear()

# Initialize FastAPI app
app = FastAPI(
    title="IndexAgent API Service",
    description="DEAN System - Core Agent Operations and Pattern Detection API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class AgentCreateRequest(BaseModel):
    """Request model for creating a new agent."""
    goal: str = Field(..., description="Primary goal/objective for the agent")
    token_budget: int = Field(default=1000, ge=100, le=50000, description="Initial token budget")
    diversity_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for diversity in evolution")
    specialized_domain: Optional[str] = Field(default=None, description="Domain of specialization")
    agent_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class AgentResponse(BaseModel):
    """Response model for agent information."""
    id: str
    name: str
    goal: str
    fitness_score: float
    token_budget: Dict[str, Any]
    diversity_score: float
    generation: int
    status: str
    specialized_domain: Optional[str]
    worktree_path: Optional[str]
    emergent_patterns: List[str]
    created_at: str
    updated_at: str

class PatternDetectionRequest(BaseModel):
    """Request for pattern detection analysis."""
    agent_ids: List[str]
    analysis_window: str = Field(default="recent", description="Time window for analysis")
    pattern_types: Optional[List[str]] = Field(default=None, description="Specific pattern types to detect")

class CodeAnalysisRequest(BaseModel):
    """Request for code analysis via IndexAgent."""
    repository_path: str
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    file_patterns: Optional[List[str]] = Field(default=None, description="File patterns to analyze")

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "IndexAgent API Service",
        "version": "1.0.0",
        "port": 8081,
        "active_agents": len(active_agents),
        "services": {
            "token_economy": token_economy is not None,
            "pattern_detector": pattern_detector is not None,
            "behavior_monitor": behavior_monitor is not None,
            "dspy_optimizer": dspy_optimizer is not None,
            "worktree_manager": worktree_manager is not None
        }
    }

@app.post("/api/v1/agents", response_model=AgentResponse)
async def create_agent(request: AgentCreateRequest):
    """Create a new agent with specified parameters."""
    try:
        # Create TokenBudget
        token_budget = TokenBudget(total=request.token_budget)
        
        # Create AgentGenome with domain-specific traits
        traits = {"efficiency": 0.5, "creativity": 0.5, "exploration": 0.5}
        
        # Adjust traits based on specialized domain
        if request.specialized_domain == "code_optimization":
            traits["efficiency"] = 0.8
        elif request.specialized_domain == "pattern_detection":
            traits["creativity"] = 0.8
        elif request.specialized_domain == "resource_efficiency":
            traits["efficiency"] = 0.9
            traits["exploration"] = 0.3
        elif request.specialized_domain == "collaboration":
            traits["creativity"] = 0.7
            traits["exploration"] = 0.7
        
        genome = AgentGenome(
            traits=traits,
            strategies=[
                f"goal_oriented_strategy",
                f"domain_{request.specialized_domain or 'general'}_strategy",
                f"token_optimization"
            ],
            mutation_rate=0.1
        )
        
        # Create FractalAgent
        agent = FractalAgent(
            name=f"IndexAgent_{len(active_agents)+1}",
            genome=genome,
            token_budget=token_budget,
            diversity_score=request.diversity_weight
        )
        
        # Create dedicated worktree for agent
        worktree_path = None
        if worktree_manager:
            try:
                branch_name = f"agent-{agent.id}"
                worktree_path = f"/tmp/worktrees/agent-{agent.id}"
                worktree_manager.create_worktree(branch_name, worktree_path)
                agent.worktree_path = worktree_path
                agent_worktrees[agent.id] = worktree_path
            except Exception as e:
                logger.warning(f"Failed to create worktree for agent {agent.id}: {e}")
        
        # Allocate tokens if token economy is available
        if token_economy:
            await token_economy.allocate_tokens(agent.id, request.token_budget)
        
        # Add to active agents
        active_agents[agent.id] = agent
        
        logger.info(f"Created agent {agent.id} with goal: {request.goal}")
        
        return AgentResponse(
            id=agent.id,
            name=agent.name,
            goal=request.goal,
            fitness_score=agent.fitness_score,
            token_budget={
                "total": agent.token_budget.total,
                "used": agent.token_budget.used,
                "remaining": agent.token_budget.remaining,
                "efficiency_score": agent.token_budget.efficiency_score
            },
            diversity_score=agent.diversity_score,
            generation=agent.generation,
            status=agent.status,
            specialized_domain=request.specialized_domain,
            worktree_path=agent.worktree_path,
            emergent_patterns=agent.emergent_patterns,
            created_at=agent.created_at.isoformat(),
            updated_at=agent.updated_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@app.get("/api/v1/agents")
async def list_agents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    specialized_domain: Optional[str] = Query(default=None)
):
    """List agents with optional filtering."""
    agents = list(active_agents.values())
    
    # Apply filters
    if status:
        agents = [a for a in agents if a.status == status]
    
    if specialized_domain:
        # Would need to store specialized_domain in agent metadata
        pass
    
    # Apply pagination
    total = len(agents)
    agents = agents[offset:offset + limit]
    
    agent_list = []
    for agent in agents:
        agent_list.append({
            "id": agent.id,
            "name": agent.name,
            "fitness_score": agent.fitness_score,
            "diversity_score": agent.diversity_score,
            "generation": agent.generation,
            "status": agent.status,
            "token_budget": {
                "total": agent.token_budget.total,
                "used": agent.token_budget.used,
                "remaining": agent.token_budget.remaining
            },
            "emergent_patterns_count": len(agent.emergent_patterns),
            "created_at": agent.created_at.isoformat()
        })
    
    return {
        "agents": agent_list,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/api/v1/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get detailed information about a specific agent."""
    if agent_id not in active_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = active_agents[agent_id]
    
    # Get agent's token budget status
    budget_status = None
    if token_economy:
        budget_status = await token_economy.get_agent_budget_status(agent_id)
    
    return {
        **agent.to_dict(),
        "worktree_path": agent_worktrees.get(agent_id),
        "budget_status": budget_status
    }

@app.post("/api/v1/agents/{agent_id}/evolve")
async def evolve_agent(agent_id: str, background_tasks: BackgroundTasks):
    """Trigger evolution for a specific agent."""
    if agent_id not in active_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = active_agents[agent_id]
    
    try:
        # Monitor behavior during evolution
        if behavior_monitor:
            action_context = {
                "actions": ["evolve", "optimize"],
                "tokens_consumed": 0,
                "task_type": "evolution",
                "success_rate": 1.0
            }
            
            background_tasks.add_task(
                behavior_monitor.observe_agent_behavior,
                agent,
                action_context
            )
        
        # Perform evolution
        environment = {
            "population": list(active_agents.values()),
            "fitness_threshold": 0.7,
            "population_diversity": 0.5
        }
        
        evolution_result = await agent.evolve(environment)
        
        # Apply DSPy optimization if available
        if dspy_optimizer:
            task_context = {"success_rate": 0.8, "task_type": "evolution"}
            optimization_result = await dspy_optimizer.optimize_agent_prompt(agent, task_context)
            evolution_result["optimization_applied"] = optimization_result
        
        logger.info(f"Evolution completed for agent {agent_id}")
        return evolution_result
        
    except Exception as e:
        logger.error(f"Evolution failed for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Evolution failed: {str(e)}")

@app.delete("/api/v1/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent and cleanup its resources."""
    if agent_id not in active_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        # Cleanup worktree
        if agent_id in agent_worktrees and worktree_manager:
            worktree_path = agent_worktrees[agent_id]
            worktree_manager.remove_worktree(worktree_path)
            del agent_worktrees[agent_id]
        
        # Remove from active agents
        del active_agents[agent_id]
        
        logger.info(f"Deleted agent {agent_id}")
        return {"deleted": True, "agent_id": agent_id}
        
    except Exception as e:
        logger.error(f"Failed to delete agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")

@app.post("/api/v1/patterns/detect")
async def detect_patterns(request: PatternDetectionRequest):
    """Detect patterns in specified agents."""
    if not pattern_detector:
        raise HTTPException(status_code=503, detail="Pattern detector not available")
    
    try:
        results = {}
        
        for agent_id in request.agent_ids:
            if agent_id in active_agents:
                agent = active_agents[agent_id]
                patterns = await pattern_detector.analyze_agent_actions(agent)
                results[agent_id] = patterns
        
        return {
            "analysis_window": request.analysis_window,
            "pattern_results": results,
            "total_agents_analyzed": len(results),
            "total_patterns_found": sum(len(patterns) for patterns in results.values())
        }
        
    except Exception as e:
        logger.error(f"Pattern detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern detection failed: {str(e)}")

@app.get("/api/v1/patterns/discovered")
async def list_discovered_patterns(
    limit: int = Query(default=100, ge=1, le=500),
    pattern_type: Optional[str] = Query(default=None),
    min_effectiveness: Optional[float] = Query(default=None, ge=0.0, le=1.0)
):
    """List discovered patterns across all agents."""
    if not behavior_monitor:
        return {"patterns": [], "total": 0}
    
    try:
        summary = behavior_monitor.get_pattern_summary()
        
        # Filter patterns if criteria provided
        patterns = summary.get("top_patterns", [])
        
        if pattern_type:
            patterns = [p for p in patterns if p.get("type") == pattern_type]
        
        if min_effectiveness:
            patterns = [p for p in patterns if p.get("effectiveness", 0) >= min_effectiveness]
        
        # Apply limit
        patterns = patterns[:limit]
        
        return {
            "patterns": patterns,
            "total": len(patterns),
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Failed to list patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list patterns: {str(e)}")

@app.post("/api/v1/code/analyze")
async def analyze_code(request: CodeAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze code using IndexAgent capabilities."""
    try:
        # This would integrate with actual IndexAgent code analysis
        # For now, provide a mock analysis
        
        analysis_result = {
            "repository_path": request.repository_path,
            "analysis_type": request.analysis_type,
            "files_analyzed": 0,
            "patterns_detected": [],
            "recommendations": [],
            "complexity_score": 0.5,
            "maintainability_score": 0.7,
            "timestamp": str(datetime.utcnow())
        }
        
        logger.info(f"Code analysis completed for {request.repository_path}")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Code analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Code analysis failed: {str(e)}")

@app.get("/api/v1/budget/global")
async def get_global_budget():
    """Get global token budget status for IndexAgent."""
    if not token_economy:
        return {"error": "Token economy not available"}
    
    try:
        status = await token_economy.get_global_budget_status()
        return {
            **status,
            "service": "IndexAgent",
            "budget_available": status["available"] > 1000  # Reserve 1K tokens
        }
        
    except Exception as e:
        logger.error(f"Failed to get budget status: {e}")
        return {"error": str(e)}

@app.get("/api/v1/metrics/efficiency")
async def get_efficiency_metrics():
    """Get efficiency metrics for IndexAgent operations."""
    if not active_agents:
        return {"metrics": {}, "summary": "No active agents"}
    
    agents = list(active_agents.values())
    
    # Calculate efficiency metrics
    efficiency_scores = [agent.fitness_score for agent in agents]
    token_efficiency = []
    
    for agent in agents:
        if agent.token_budget.used > 0:
            eff = agent.fitness_score / agent.token_budget.used
            token_efficiency.append(eff)
    
    # Pattern discovery efficiency
    total_patterns = sum(len(agent.emergent_patterns) for agent in agents)
    
    metrics = {
        "agent_performance": {
            "average_fitness": sum(efficiency_scores) / len(efficiency_scores),
            "max_fitness": max(efficiency_scores),
            "min_fitness": min(efficiency_scores),
            "total_agents": len(agents)
        },
        "token_efficiency": {
            "average_efficiency": sum(token_efficiency) / len(token_efficiency) if token_efficiency else 0.0,
            "total_tokens_used": sum(a.token_budget.used for a in agents)
        },
        "pattern_discovery": {
            "total_patterns": total_patterns,
            "patterns_per_agent": total_patterns / len(agents),
            "unique_pattern_types": len(set(
                pattern for agent in agents for pattern in agent.emergent_patterns
            ))
        }
    }
    
    return {"metrics": metrics, "service": "IndexAgent"}

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint for IndexAgent."""
    agents_count = len(active_agents)
    avg_fitness = sum(a.fitness_score for a in active_agents.values()) / len(active_agents) if active_agents else 0
    total_patterns = sum(len(a.emergent_patterns) for a in active_agents.values())
    
    metrics = f"""# HELP indexagent_active_agents Currently active agents
# TYPE indexagent_active_agents gauge
indexagent_active_agents {agents_count}

# HELP indexagent_average_fitness Average fitness score
# TYPE indexagent_average_fitness gauge
indexagent_average_fitness {avg_fitness}

# HELP indexagent_total_patterns Total discovered patterns
# TYPE indexagent_total_patterns counter
indexagent_total_patterns {total_patterns}

# HELP indexagent_service_status Service health status
# TYPE indexagent_service_status gauge
indexagent_service_status 1
"""
    
    return metrics

if __name__ == "__main__":
    import datetime
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info"
    )