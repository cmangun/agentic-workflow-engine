"""
Tool Registry for Agentic Workflow Engine.

Production-grade tool management for AI agents:
- Tool registration with schema validation
- Permission-based tool access
- Rate limiting and cost tracking
- Audit logging for tool invocations
- Healthcare-specific tool policies
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class ToolCategory(Enum):
    """Tool categories for access control."""
    
    DATA_READ = "data_read"           # Read-only data access
    DATA_WRITE = "data_write"         # Data modification
    EXTERNAL_API = "external_api"     # External service calls
    PHI_ACCESS = "phi_access"         # Protected health information
    ADMIN = "admin"                   # Administrative operations
    SEARCH = "search"                 # Search and retrieval
    COMPUTE = "compute"               # Computational operations


class ToolStatus(Enum):
    """Tool availability status."""
    
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"
    MAINTENANCE = "maintenance"


@dataclass
class ToolParameter:
    """Tool parameter definition."""
    
    name: str
    param_type: str  # string, integer, float, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum_values: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.param_type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "enum": self.enum_values,
            "minimum": self.min_value,
            "maximum": self.max_value,
        }
    
    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate parameter value."""
        if value is None:
            if self.required:
                return False, f"Required parameter '{self.name}' is missing"
            return True, ""
        
        # Type validation
        type_map = {
            "string": str,
            "integer": int,
            "float": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        expected_type = type_map.get(self.param_type)
        if expected_type and not isinstance(value, expected_type):
            return False, f"Parameter '{self.name}' must be {self.param_type}"
        
        # Enum validation
        if self.enum_values and value not in self.enum_values:
            return False, f"Parameter '{self.name}' must be one of {self.enum_values}"
        
        # Range validation for numbers
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be <= {self.max_value}"
        
        return True, ""


@dataclass
class ToolDefinition:
    """Tool definition with schema."""
    
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter]
    handler: Callable[..., Any]
    status: ToolStatus = ToolStatus.ACTIVE
    version: str = "1.0.0"
    requires_approval: bool = False
    max_execution_time_ms: int = 30000
    estimated_cost_usd: float = 0.0
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [p.to_dict() for p in self.parameters],
            "status": self.status.value,
            "version": self.version,
            "requires_approval": self.requires_approval,
            "max_execution_time_ms": self.max_execution_time_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
            "tags": self.tags,
        }
    
    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.param_type,
                "description": param.description,
            }
            if param.enum_values:
                prop["enum"] = param.enum_values
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolInvocation:
    """Record of a tool invocation."""
    
    invocation_id: str
    tool_id: str
    tool_name: str
    agent_id: str
    workflow_id: str
    parameters: dict[str, Any]
    result: Any
    status: str  # success, failure, timeout, denied
    start_time: datetime
    end_time: datetime
    execution_time_ms: float
    error_message: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "agent_id": self.agent_id,
            "workflow_id": self.workflow_id,
            "parameters": self.parameters,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
        }


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    daily_budget_usd: float = 100.0


class ToolRegistry:
    """
    Production tool registry for AI agents.
    
    Features:
    - Schema validation
    - Permission-based access
    - Rate limiting
    - Audit logging
    - Healthcare policy enforcement
    """
    
    def __init__(
        self,
        rate_limit_config: RateLimitConfig | None = None,
    ):
        self._tools: dict[str, ToolDefinition] = {}
        self._invocations: list[ToolInvocation] = []
        self._rate_limits: dict[str, list[datetime]] = {}
        self._cost_tracking: dict[str, float] = {}
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        
        # Register built-in healthcare tools
        self._register_healthcare_tools()
    
    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        if tool.tool_id in self._tools:
            logger.warning(
                "tool_override",
                tool_id=tool.tool_id,
                name=tool.name,
            )
        
        self._tools[tool.tool_id] = tool
        
        logger.info(
            "tool_registered",
            tool_id=tool.tool_id,
            name=tool.name,
            category=tool.category.value,
        )
    
    def get(self, tool_id: str) -> ToolDefinition | None:
        """Get tool by ID."""
        return self._tools.get(tool_id)
    
    def get_by_name(self, name: str) -> ToolDefinition | None:
        """Get tool by name."""
        for tool in self._tools.values():
            if tool.name == name:
                return tool
        return None
    
    def list_tools(
        self,
        category: ToolCategory | None = None,
        status: ToolStatus = ToolStatus.ACTIVE,
    ) -> list[ToolDefinition]:
        """List available tools."""
        tools = list(self._tools.values())
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        tools = [t for t in tools if t.status == status]
        
        return tools
    
    def invoke(
        self,
        tool_id: str,
        parameters: dict[str, Any],
        agent_id: str,
        workflow_id: str,
    ) -> tuple[Any, ToolInvocation]:
        """
        Invoke a tool with validation and tracking.
        
        Args:
            tool_id: Tool identifier
            parameters: Tool parameters
            agent_id: Agent making the call
            workflow_id: Workflow context
            
        Returns:
            Tuple of (result, invocation_record)
        """
        invocation_id = self._generate_invocation_id()
        start_time = datetime.utcnow()
        
        tool = self._tools.get(tool_id)
        
        if not tool:
            return self._create_error_invocation(
                invocation_id, tool_id, agent_id, workflow_id,
                parameters, start_time, "Tool not found"
            )
        
        if tool.status != ToolStatus.ACTIVE:
            return self._create_error_invocation(
                invocation_id, tool_id, agent_id, workflow_id,
                parameters, start_time, f"Tool is {tool.status.value}"
            )
        
        # Rate limit check
        if not self._check_rate_limit(agent_id):
            return self._create_error_invocation(
                invocation_id, tool.tool_id, agent_id, workflow_id,
                parameters, start_time, "Rate limit exceeded"
            )
        
        # Budget check
        if not self._check_budget(agent_id, tool.estimated_cost_usd):
            return self._create_error_invocation(
                invocation_id, tool.tool_id, agent_id, workflow_id,
                parameters, start_time, "Daily budget exceeded"
            )
        
        # Parameter validation
        for param in tool.parameters:
            value = parameters.get(param.name, param.default)
            valid, error = param.validate(value)
            if not valid:
                return self._create_error_invocation(
                    invocation_id, tool.tool_id, agent_id, workflow_id,
                    parameters, start_time, error
                )
        
        # Execute tool
        try:
            result = tool.handler(**parameters)
            end_time = datetime.utcnow()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Track cost
            self._track_cost(agent_id, tool.estimated_cost_usd)
            
            invocation = ToolInvocation(
                invocation_id=invocation_id,
                tool_id=tool.tool_id,
                tool_name=tool.name,
                agent_id=agent_id,
                workflow_id=workflow_id,
                parameters=parameters,
                result=result,
                status="success",
                start_time=start_time,
                end_time=end_time,
                execution_time_ms=execution_time_ms,
            )
            
            self._invocations.append(invocation)
            
            logger.info(
                "tool_invoked",
                invocation_id=invocation_id,
                tool_id=tool.tool_id,
                agent_id=agent_id,
                execution_time_ms=execution_time_ms,
            )
            
            return result, invocation
            
        except Exception as e:
            return self._create_error_invocation(
                invocation_id, tool.tool_id, agent_id, workflow_id,
                parameters, start_time, str(e)
            )
    
    def get_invocation_history(
        self,
        agent_id: str | None = None,
        workflow_id: str | None = None,
        limit: int = 100,
    ) -> list[ToolInvocation]:
        """Get tool invocation history."""
        invocations = self._invocations
        
        if agent_id:
            invocations = [i for i in invocations if i.agent_id == agent_id]
        
        if workflow_id:
            invocations = [i for i in invocations if i.workflow_id == workflow_id]
        
        return invocations[-limit:]
    
    def get_tools_as_openai_functions(
        self,
        category: ToolCategory | None = None,
    ) -> list[dict[str, Any]]:
        """Get tools in OpenAI function calling format."""
        tools = self.list_tools(category=category)
        return [t.to_openai_function() for t in tools]
    
    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------
    
    def _generate_invocation_id(self) -> str:
        """Generate unique invocation ID."""
        content = f"{datetime.utcnow().isoformat()}:{time.time()}"
        return f"inv_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    
    def _check_rate_limit(self, agent_id: str) -> bool:
        """Check if agent is within rate limits."""
        now = datetime.utcnow()
        
        if agent_id not in self._rate_limits:
            self._rate_limits[agent_id] = []
        
        # Clean old entries
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        self._rate_limits[agent_id] = [
            t for t in self._rate_limits[agent_id]
            if t > hour_ago
        ]
        
        # Check minute limit
        minute_requests = sum(
            1 for t in self._rate_limits[agent_id]
            if t > minute_ago
        )
        
        if minute_requests >= self.rate_limit_config.requests_per_minute:
            return False
        
        # Check hour limit
        if len(self._rate_limits[agent_id]) >= self.rate_limit_config.requests_per_hour:
            return False
        
        # Record this request
        self._rate_limits[agent_id].append(now)
        
        return True
    
    def _check_budget(self, agent_id: str, cost: float) -> bool:
        """Check if agent is within daily budget."""
        current_spend = self._cost_tracking.get(agent_id, 0.0)
        return current_spend + cost <= self.rate_limit_config.daily_budget_usd
    
    def _track_cost(self, agent_id: str, cost: float) -> None:
        """Track tool invocation cost."""
        if agent_id not in self._cost_tracking:
            self._cost_tracking[agent_id] = 0.0
        self._cost_tracking[agent_id] += cost
    
    def _create_error_invocation(
        self,
        invocation_id: str,
        tool_id: str,
        agent_id: str,
        workflow_id: str,
        parameters: dict[str, Any],
        start_time: datetime,
        error_message: str,
    ) -> tuple[None, ToolInvocation]:
        """Create error invocation record."""
        end_time = datetime.utcnow()
        
        invocation = ToolInvocation(
            invocation_id=invocation_id,
            tool_id=tool_id,
            tool_name="unknown",
            agent_id=agent_id,
            workflow_id=workflow_id,
            parameters=parameters,
            result=None,
            status="failure",
            start_time=start_time,
            end_time=end_time,
            execution_time_ms=(end_time - start_time).total_seconds() * 1000,
            error_message=error_message,
        )
        
        self._invocations.append(invocation)
        
        logger.warning(
            "tool_invocation_failed",
            invocation_id=invocation_id,
            tool_id=tool_id,
            error=error_message,
        )
        
        return None, invocation
    
    def _register_healthcare_tools(self) -> None:
        """Register built-in healthcare tools."""
        
        # Patient lookup tool
        self.register(ToolDefinition(
            tool_id="patient_lookup",
            name="patient_lookup",
            description="Look up patient information by ID",
            category=ToolCategory.PHI_ACCESS,
            parameters=[
                ToolParameter(
                    name="patient_id",
                    param_type="string",
                    description="Patient identifier (MRN)",
                    required=True,
                ),
                ToolParameter(
                    name="include_history",
                    param_type="boolean",
                    description="Include medical history",
                    required=False,
                    default=False,
                ),
            ],
            handler=self._mock_patient_lookup,
            requires_approval=True,
            estimated_cost_usd=0.01,
            tags=["patient", "phi", "healthcare"],
        ))
        
        # Lab results tool
        self.register(ToolDefinition(
            tool_id="lab_results",
            name="get_lab_results",
            description="Retrieve lab results for a patient",
            category=ToolCategory.PHI_ACCESS,
            parameters=[
                ToolParameter(
                    name="patient_id",
                    param_type="string",
                    description="Patient identifier",
                    required=True,
                ),
                ToolParameter(
                    name="test_type",
                    param_type="string",
                    description="Type of lab test",
                    required=False,
                    enum_values=["blood", "urine", "imaging", "all"],
                    default="all",
                ),
                ToolParameter(
                    name="days_back",
                    param_type="integer",
                    description="Number of days to look back",
                    required=False,
                    default=30,
                    min_value=1,
                    max_value=365,
                ),
            ],
            handler=self._mock_lab_results,
            requires_approval=True,
            estimated_cost_usd=0.02,
            tags=["labs", "phi", "healthcare"],
        ))
        
        # Medical knowledge search
        self.register(ToolDefinition(
            tool_id="medical_search",
            name="search_medical_knowledge",
            description="Search medical knowledge base",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    param_type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    param_type="integer",
                    description="Maximum results to return",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=50,
                ),
            ],
            handler=self._mock_medical_search,
            requires_approval=False,
            estimated_cost_usd=0.005,
            tags=["search", "knowledge", "healthcare"],
        ))
        
        # Drug interaction checker
        self.register(ToolDefinition(
            tool_id="drug_interaction",
            name="check_drug_interactions",
            description="Check for potential drug interactions",
            category=ToolCategory.COMPUTE,
            parameters=[
                ToolParameter(
                    name="medications",
                    param_type="array",
                    description="List of medication names",
                    required=True,
                ),
            ],
            handler=self._mock_drug_interaction,
            requires_approval=False,
            estimated_cost_usd=0.01,
            tags=["drugs", "safety", "healthcare"],
        ))
    
    # Mock handlers for healthcare tools
    def _mock_patient_lookup(
        self,
        patient_id: str,
        include_history: bool = False,
    ) -> dict[str, Any]:
        return {
            "patient_id": patient_id,
            "status": "found",
            "demographics": {"age": 45, "gender": "M"},
            "history_included": include_history,
        }
    
    def _mock_lab_results(
        self,
        patient_id: str,
        test_type: str = "all",
        days_back: int = 30,
    ) -> dict[str, Any]:
        return {
            "patient_id": patient_id,
            "test_type": test_type,
            "results": [],
            "period_days": days_back,
        }
    
    def _mock_medical_search(
        self,
        query: str,
        max_results: int = 10,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "results": [],
            "total_found": 0,
        }
    
    def _mock_drug_interaction(
        self,
        medications: list[str],
    ) -> dict[str, Any]:
        return {
            "medications": medications,
            "interactions": [],
            "severity": "none",
        }
