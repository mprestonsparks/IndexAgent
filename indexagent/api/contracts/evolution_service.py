#!/usr/bin/env python3
"""
Evolution Service API Contracts
Implementation of exact API contracts per Service Communication Section 3.4

This module implements the service communication patterns between IndexAgent
and Evolution API as specified in the architectural design document.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp
from pydantic import BaseModel, Field, validator

# Implements Service Communication Section 3.4: API contracts between services
logger = logging.getLogger(__name__)


class ServiceCommunicationError(Exception):
    """Exception raised for service communication failures"""
    pass


class EvolutionRequestType(str, Enum):
    """Evolution request types per specification"""
    PATTERN_PROPAGATION = "pattern_propagation"
    AGENT_EVOLUTION = "agent_evolution"
    POPULATION_METRICS = "population_metrics"
    LINEAGE_TRACKING = "lineage_tracking"


class PatternPropagationRequest(BaseModel):
    """Pattern propagation request per FR-010"""
    pattern_id: str = Field(..., description="Unique pattern identifier")
    pattern_type: str = Field(..., description="Pattern classification")
    effectiveness_score: float = Field(..., ge=0.0, le=1.0, description="Pattern effectiveness")
    source_agent_id: str = Field(..., description="Agent that discovered the pattern")
    pattern_data: Dict[str, Any] = Field(..., description="Pattern implementation details")
    propagation_scope: str = Field(default="population", description="Propagation scope")


class AgentEvolutionRequest(BaseModel):
    """Agent evolution request per FR-009"""
    agent_id: str = Field(..., description="Agent to evolve")
    parent_id: Optional[str] = Field(None, description="Parent agent for lineage")
    generation: int = Field(..., ge=0, description="Generation number")
    evolution_type: str = Field(..., description="Type of evolution")
    ca_rule: str = Field(..., description="Cellular automata rule to apply")
    token_budget: int = Field(..., gt=0, description="Token budget for evolution")
    diversity_target: float = Field(..., ge=0.0, le=1.0, description="Target diversity score")

    @validator('ca_rule')
    def validate_ca_rule(cls, v):
        """Validate cellular automata rule per FR-009"""
        valid_rules = ['rule_110', 'rule_30', 'rule_90', 'rule_184', 'rule_1']
        if v not in valid_rules:
            raise ValueError(f"CA rule must be one of {valid_rules}")
        return v


class PopulationMetricsRequest(BaseModel):
    """Population metrics request for analytics"""
    metric_types: List[str] = Field(..., description="Types of metrics to collect")
    time_range: Optional[str] = Field(None, description="Time range for metrics")
    agent_filter: Optional[Dict[str, Any]] = Field(None, description="Agent filtering criteria")


class ServiceResponse(BaseModel):
    """Standard service response format"""
    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    service_name: str = Field(..., description="Responding service name")


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    base_url: str
    timeout: int = 30
    retry_count: int = 3


class EvolutionServiceClient:
    """
    Evolution Service API client implementing Service Communication Section 3.4
    
    Provides standardized communication patterns between IndexAgent and Evolution API
    with proper error handling, retries, and metrics collection.
    """
    
    def __init__(self, evolution_api_url: str = "http://agent-evolution:8080/api/v1"):
        """
        Initialize Evolution Service client
        
        Args:
            evolution_api_url: Base URL for Evolution API service
        """
        self.evolution_endpoint = ServiceEndpoint(
            name="evolution_api",
            base_url=evolution_api_url
        )
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.evolution_endpoint.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def propagate_pattern(self, request: PatternPropagationRequest) -> ServiceResponse:
        """
        Propagate pattern to Evolution API per FR-010
        
        Implements pattern propagation across agent population via Redis cache
        as specified in Service Communication Section 3.4.
        
        Args:
            request: Pattern propagation request
            
        Returns:
            ServiceResponse with propagation results
            
        Raises:
            ServiceCommunicationError: If propagation fails
        """
        endpoint = f"{self.evolution_endpoint.base_url}/patterns/propagate"
        
        try:
            response_data = await self._make_request(
                "POST", 
                endpoint, 
                request.dict()
            )
            
            logger.info(f"Pattern {request.pattern_id} propagated successfully")
            return ServiceResponse(
                success=True,
                message=f"Pattern {request.pattern_id} propagated to population",
                data=response_data,
                request_id=self._generate_request_id(),
                service_name="evolution_api"
            )
            
        except Exception as e:
            logger.error(f"Pattern propagation failed: {e}")
            raise ServiceCommunicationError(f"Pattern propagation failed: {e}")
    
    async def trigger_agent_evolution(self, request: AgentEvolutionRequest) -> ServiceResponse:
        """
        Trigger agent evolution per FR-009 and FR-011
        
        Implements cellular automata evolution with lineage tracking
        as specified in Service Communication Section 3.4.
        
        Args:
            request: Agent evolution request
            
        Returns:
            ServiceResponse with evolution results
            
        Raises:
            ServiceCommunicationError: If evolution fails
        """
        endpoint = f"{self.evolution_endpoint.base_url}/agents/{request.agent_id}/evolve"
        
        try:
            response_data = await self._make_request(
                "POST",
                endpoint,
                request.dict()
            )
            
            logger.info(f"Agent {request.agent_id} evolution triggered with {request.ca_rule}")
            return ServiceResponse(
                success=True,
                message=f"Agent {request.agent_id} evolution completed",
                data=response_data,
                request_id=self._generate_request_id(),
                service_name="evolution_api"
            )
            
        except Exception as e:
            logger.error(f"Agent evolution failed: {e}")
            raise ServiceCommunicationError(f"Agent evolution failed: {e}")
    
    async def get_population_metrics(self, request: PopulationMetricsRequest) -> ServiceResponse:
        """
        Get population metrics from Evolution API
        
        Implements metrics collection for population analysis
        as specified in Service Communication Section 3.4.
        
        Args:
            request: Population metrics request
            
        Returns:
            ServiceResponse with metrics data
            
        Raises:
            ServiceCommunicationError: If metrics collection fails
        """
        endpoint = f"{self.evolution_endpoint.base_url}/population/metrics"
        
        try:
            response_data = await self._make_request(
                "POST",
                endpoint,
                request.dict()
            )
            
            logger.info("Population metrics retrieved successfully")
            return ServiceResponse(
                success=True,
                message="Population metrics retrieved",
                data=response_data,
                request_id=self._generate_request_id(),
                service_name="evolution_api"
            )
            
        except Exception as e:
            logger.error(f"Population metrics retrieval failed: {e}")
            raise ServiceCommunicationError(f"Population metrics retrieval failed: {e}")
    
    async def register_agent_lineage(self, agent_id: str, parent_id: Optional[str], 
                                   generation: int) -> ServiceResponse:
        """
        Register agent lineage per FR-004
        
        Implements agent lineage tracking in PostgreSQL agent_evolution.agents table
        as specified in Service Communication Section 3.4.
        
        Args:
            agent_id: Agent identifier
            parent_id: Parent agent identifier (None for root agents)
            generation: Generation number
            
        Returns:
            ServiceResponse with registration results
            
        Raises:
            ServiceCommunicationError: If registration fails
        """
        endpoint = f"{self.evolution_endpoint.base_url}/agents/{agent_id}/lineage"
        
        lineage_data = {
            "agent_id": agent_id,
            "parent_id": parent_id,
            "generation": generation,
            "registered_at": datetime.now().isoformat()
        }
        
        try:
            response_data = await self._make_request(
                "POST",
                endpoint,
                lineage_data
            )
            
            logger.info(f"Agent {agent_id} lineage registered (parent: {parent_id}, gen: {generation})")
            return ServiceResponse(
                success=True,
                message=f"Agent {agent_id} lineage registered",
                data=response_data,
                request_id=self._generate_request_id(),
                service_name="evolution_api"
            )
            
        except Exception as e:
            logger.error(f"Agent lineage registration failed: {e}")
            raise ServiceCommunicationError(f"Agent lineage registration failed: {e}")
    
    async def health_check(self) -> ServiceResponse:
        """
        Check Evolution API health per service monitoring requirements
        
        Returns:
            ServiceResponse with health status
        """
        endpoint = f"{self.evolution_endpoint.base_url}/health"
        
        try:
            response_data = await self._make_request("GET", endpoint)
            
            return ServiceResponse(
                success=True,
                message="Evolution API health check passed",
                data=response_data,
                request_id=self._generate_request_id(),
                service_name="evolution_api"
            )
            
        except Exception as e:
            logger.error(f"Evolution API health check failed: {e}")
            return ServiceResponse(
                success=False,
                message=f"Evolution API health check failed: {e}",
                data={"error": str(e)},
                request_id=self._generate_request_id(),
                service_name="evolution_api"
            )
    
    async def _make_request(self, method: str, url: str, 
                          data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and error handling
        
        Args:
            method: HTTP method
            url: Request URL
            data: Request data
            
        Returns:
            Response data
            
        Raises:
            ServiceCommunicationError: If request fails after retries
        """
        if not self.session:
            raise ServiceCommunicationError("Session not initialized")
        
        for attempt in range(self.evolution_endpoint.retry_count):
            try:
                async with self.session.request(
                    method, 
                    url, 
                    json=data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        raise ServiceCommunicationError(f"Endpoint not found: {url}")
                    elif response.status == 503:
                        raise ServiceCommunicationError("Evolution API service unavailable")
                    else:
                        error_text = await response.text()
                        raise ServiceCommunicationError(
                            f"HTTP {response.status}: {error_text}"
                        )
                        
            except aiohttp.ClientError as e:
                if attempt == self.evolution_endpoint.retry_count - 1:
                    raise ServiceCommunicationError(f"Request failed after {self.evolution_endpoint.retry_count} attempts: {e}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                
        raise ServiceCommunicationError("Request failed after all retry attempts")
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid
        return str(uuid.uuid4())[:8]


# Service factory function
def create_evolution_service_client(evolution_api_url: Optional[str] = None) -> EvolutionServiceClient:
    """
    Factory function to create Evolution Service client
    
    Args:
        evolution_api_url: Custom Evolution API URL (uses default if None)
        
    Returns:
        Configured EvolutionServiceClient
    """
    if evolution_api_url is None:
        evolution_api_url = "http://agent-evolution:8080/api/v1"
    
    return EvolutionServiceClient(evolution_api_url)


# Convenience functions for common operations
async def propagate_pattern_to_population(pattern_id: str, pattern_type: str, 
                                        effectiveness_score: float, source_agent_id: str,
                                        pattern_data: Dict[str, Any]) -> ServiceResponse:
    """
    Convenience function for pattern propagation
    
    Implements FR-010: Pattern propagation via Redis
    """
    async with create_evolution_service_client() as client:
        request = PatternPropagationRequest(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            effectiveness_score=effectiveness_score,
            source_agent_id=source_agent_id,
            pattern_data=pattern_data
        )
        return await client.propagate_pattern(request)


async def evolve_agent_with_ca_rule(agent_id: str, ca_rule: str, token_budget: int,
                                  parent_id: Optional[str] = None) -> ServiceResponse:
    """
    Convenience function for agent evolution
    
    Implements FR-009: Cellular automata evolution
    """
    async with create_evolution_service_client() as client:
        request = AgentEvolutionRequest(
            agent_id=agent_id,
            parent_id=parent_id,
            generation=1 if parent_id else 0,
            evolution_type="cellular_automata",
            ca_rule=ca_rule,
            token_budget=token_budget,
            diversity_target=0.7  # Default diversity target
        )
        return await client.trigger_agent_evolution(request)


async def check_evolution_service_health() -> bool:
    """
    Convenience function for health checking
    
    Returns:
        True if Evolution API is healthy, False otherwise
    """
    try:
        async with create_evolution_service_client() as client:
            response = await client.health_check()
            return response.success
    except Exception:
        return False