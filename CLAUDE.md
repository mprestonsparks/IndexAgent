# CLAUDE.md - IndexAgent Repository

This file provides repository-specific guidance for the IndexAgent system.

## Review Report Requirements

<review-report-standards>
When creating review reports for completed tasks, Claude Code MUST follow these standards:

1. **Naming Convention**: 
   - XML format: `REVIEW_REPORT_YYYY-MM-DD-HHMM_XML.md`
   - Markdown format: `REVIEW_REPORT_YYYY-MM-DD-HHMM_MD.md`
   - Example: `REVIEW_REPORT_2025-06-27-1452_XML.md` and `REVIEW_REPORT_2025-06-27-1452_MD.md`
   - Time should be in 24-hour format (e.g., 1452 for 2:52 PM)

2. **Dual Format Requirement**:
   - Always create TWO versions of each review report
   - XML version: Use XML syntax throughout for structured data
   - Markdown version: Use standard markdown formatting for readability
   - Both must contain identical information, only formatting differs

3. **Storage Location**:
   - All review reports MUST be saved in: `.claude/review-reports/`
   - Create the directory if it doesn't exist: `mkdir -p .claude/review-reports`
   - This applies to ALL repositories in the DEAN system

4. **Required Metadata**:
   Each review report MUST include metadata at the top:
   ```xml
   <report-metadata>
     <creation-date>YYYY-MM-DD</creation-date>
     <creation-time>HH:MM PST/EST</creation-time>
     <report-type>Implementation Review/Bug Fix/Feature Addition/etc.</report-type>
     <author>Claude Code Assistant</author>
     <system>DEAN</system>
     <component>Component Name</component>
     <task-id>Unique Task Identifier</task-id>
   </report-metadata>
   ```
</review-report-standards>

## Repository Context

IndexAgent implements the core agent logic for the DEAN (Distributed Evolutionary Agent Network) system. This repository contains the evolutionary algorithms, pattern detection systems, and agent implementations that form the intelligence layer of the distributed system.

## CRITICAL: This is Part of a Distributed System

<distributed_system_warning>
⚠️ **WARNING: The DEAN system spans FOUR repositories** ⚠️

This repository contains ONLY the agent logic and evolution algorithms. Other components are located in:
- **DEAN**: Orchestration, authentication, monitoring (Port 8082-8083)
- **infra**: Docker configs, database schemas, deployment scripts
- **airflow-hub**: DAGs, operators, workflow orchestration (Port 8080)

**Specification Documents Location**: DEAN/specifications/ (read-only)

Always check all repositories before implementing features!
</distributed_system_warning>

## Available MCP Tools

### remote_exec
- **Location**: `/Users/preston/dev/mcp-tools/remote_exec/remote_exec_launcher.sh`
- **Purpose**: Execute PowerShell commands on Windows deployment PC (10.7.0.2)
- **Usage**: The remote_exec tool allows execution of PowerShell scripts on the remote Windows deployment server
- **SSH Key**: `~/.ssh/claude_remote_exec`
- **Target**: `deployer@10.7.0.2`

## Critical Implementation Requirements

### NO MOCK IMPLEMENTATIONS

<implementation_standards>
When implementing any feature in this codebase, Claude Code MUST create actual, working code. The following are STRICTLY PROHIBITED:
- Mock implementations or stub functions presented as complete
- Placeholder code with TODO comments in "finished" work
- Simulated test results or hypothetical outputs
- Documentation of what "would" happen instead of what "does" happen
- Pseudocode or conceptual implementations claimed as functional

Every implementation MUST:
- Be fully functional and executable with proper error handling
- Work with actual services and dependencies
- Be tested with real commands showing actual output
- Include complete implementations of all code paths
</implementation_standards>

## IndexAgent-Specific Architecture

### Core Responsibilities
- **Agent Implementation**: FractalAgent class with genetic traits and strategies
- **Evolution Algorithms**: Genetic algorithms and cellular automata rules
- **Pattern Detection**: Behavioral, optimization, and emergent pattern recognition
- **Token Economy**: Budget management and resource allocation
- **DSPy Integration**: Prompt optimization for agent efficiency

### Key Components in This Repository

```
indexagent/
├── agents/
│   ├── base_agent.py          # FractalAgent implementation
│   ├── evolution/
│   │   ├── genetic_algorithm.py    # Crossover, mutation, selection
│   │   ├── cellular_automata.py    # CA rules (110, 30, 90, 184)
│   │   ├── diversity_manager.py    # Genetic diversity enforcement
│   │   └── dspy_optimizer.py       # Prompt optimization
│   ├── patterns/
│   │   ├── detector.py            # Pattern detection system
│   │   ├── classifier.py          # Behavior classification
│   │   └── monitor.py             # Emergent behavior monitoring
│   └── economy/
│       └── token_manager.py       # Token budget enforcement
├── database/
│   ├── schema.py                 # SQLAlchemy models
│   └── migrations.py             # Database migrations
└── main_api.py                   # FastAPI service (Port 8081)
```

### What This Repository Does NOT Contain
- **Orchestration Logic**: Located in DEAN/src/dean_orchestration/
- **DAG Definitions**: Located in airflow-hub/dags/dean/
- **Docker Configurations**: Located in infra/docker-compose.dean.yml
- **Deployment Scripts**: Located in infra/scripts/

## Development Standards

### Evolution Algorithm Requirements

<evolution_rules>
<rule context="genetic_diversity">
Population variance must remain above 0.3. Implement automatic mutation injection when diversity drops.
</rule>

<rule context="cellular_automata">
Implement all four required CA rules:
- Rule 110: Complexity generation
- Rule 30: Randomness generation
- Rule 90: Fractal patterns
- Rule 184: Traffic flow dynamics
</rule>

<rule context="token_economy">
Every agent operation must check and respect token budgets. Never allow unlimited token consumption.
</rule>
</evolution_rules>

### API Development Standards

```python
# REQUIRED: Complete implementation pattern for IndexAgent APIs
@app.post("/agents/{agent_id}/evolve")
async def evolve_agent(
    agent_id: str,
    params: EvolutionParams,
    token_budget: int = Query(..., gt=0)
):
    """Full implementation with budget enforcement"""
    # Check agent exists
    agent = await get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    
    # Enforce token budget
    if token_budget > agent.remaining_budget:
        raise HTTPException(400, f"Requested {token_budget} but only {agent.remaining_budget} available")
    
    # Execute evolution with monitoring
    try:
        result = await evolution_engine.evolve(
            agent=agent,
            params=params,
            budget=token_budget
        )
        
        # Update metrics
        await metrics_collector.record_evolution(agent_id, result)
        
        return result
    except Exception as e:
        logger.error(f"Evolution failed for {agent_id}: {e}")
        raise HTTPException(500, "Evolution failed")
```

## Configuration

```yaml
# indexagent/agents/config/evolution.yaml
evolution:
  population_size: 8
  generations: 50
  diversity_threshold: 0.3
  token_budget_per_agent: 4096
  cellular_automata:
    rules: [110, 30, 90, 184]
  genetic_algorithm:
    crossover_rate: 0.7
    mutation_rate: 0.1
    elitism_size: 2
```

## Common Commands

```bash
# Development
python main_api.py                    # Start API server on port 8081
pytest tests/ -v --cov=indexagent    # Run tests with coverage
docker build -t indexagent .         # Build container

# Agent Operations
python -m indexagent.agents.cli create --genome "default"  # Create agent
python -m indexagent.agents.cli evolve --agent-id "xyz"    # Evolve agent
python -m indexagent.agents.cli patterns --detect          # Detect patterns

# Database
alembic upgrade head                 # Apply migrations
python -m indexagent.database.init   # Initialize schema
```

## Integration Points

### Service Communication
- **DEAN Orchestrator**: http://dean-orchestration:8082
- **Evolution API**: http://evolution-api:8090
- **Airflow**: http://airflow:8080
- **Database**: postgresql://postgres:5432/agent_evolution

### Required Environment Variables
```bash
INDEXAGENT_PORT=8081
DATABASE_URL=postgresql://user:pass@postgres:5432/agent_evolution
REDIS_URL=redis://redis:6379
DEAN_API_URL=http://dean-orchestration:8082
JWT_SECRET_KEY=<shared-secret>
```

## Testing Requirements

<testing_standards>
- Unit tests for all evolution algorithms
- Integration tests with database operations
- Performance tests for large populations
- Diversity maintenance validation
- Token budget enforcement tests
- Pattern detection accuracy tests
</testing_standards>

## Security Requirements

<security_rules>
- Validate all agent genomes before execution
- Sanitize pattern detection inputs
- Enforce rate limiting on evolution endpoints
- Never expose internal agent state
- Log all token transactions
</security_rules>