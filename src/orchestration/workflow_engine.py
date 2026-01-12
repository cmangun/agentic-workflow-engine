"""
Agentic Workflow Engine - Production Agent Orchestration

Enterprise-grade agent orchestration supporting:
- Multi-step reasoning with checkpoints
- Tool registration and invocation
- Policy enforcement and guardrails
- State management and rollback
- Comprehensive audit logging
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class AgentState(str, Enum):
    """Agent execution states."""
    
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepType(str, Enum):
    """Types of workflow steps."""
    
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    HUMAN_INPUT = "human_input"
    CHECKPOINT = "checkpoint"
    DECISION = "decision"
    PARALLEL = "parallel"


class PolicyDecision(str, Enum):
    """Policy enforcement decisions."""
    
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    MODIFY = "modify"


@dataclass
class ToolDefinition:
    """Definition of an available tool."""
    
    name: str
    description: str
    parameters: dict[str, Any]
    required_permissions: list[str]
    cost_estimate_usd: float = 0.0
    timeout_seconds: int = 30
    requires_approval: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required_permissions": self.required_permissions,
            "cost_estimate_usd": self.cost_estimate_usd,
            "timeout_seconds": self.timeout_seconds,
            "requires_approval": self.requires_approval,
        }


@dataclass
class ToolInvocation:
    """Record of a tool invocation."""
    
    invocation_id: str
    tool_name: str
    parameters: dict[str, Any]
    result: Any
    success: bool
    error_message: str | None
    duration_ms: float
    timestamp: datetime
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "result": self.result if self.success else None,
            "success": self.success,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class WorkflowStep:
    """A step in the agent workflow."""
    
    step_id: str
    step_type: StepType
    name: str
    description: str
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None
    status: AgentState
    started_at: datetime | None
    completed_at: datetime | None
    duration_ms: float | None
    error: str | None
    tool_invocation: ToolInvocation | None = None
    child_steps: list[WorkflowStep] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "name": self.name,
            "description": self.description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "tool_invocation": self.tool_invocation.to_dict() if self.tool_invocation else None,
            "child_steps": [s.to_dict() for s in self.child_steps],
        }


@dataclass
class Checkpoint:
    """Workflow checkpoint for recovery."""
    
    checkpoint_id: str
    workflow_id: str
    step_id: str
    state: dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class WorkflowExecution:
    """Complete workflow execution record."""
    
    workflow_id: str
    name: str
    description: str
    user_id: str
    state: AgentState
    steps: list[WorkflowStep]
    checkpoints: list[Checkpoint]
    context: dict[str, Any]
    started_at: datetime
    completed_at: datetime | None
    total_duration_ms: float | None
    total_cost_usd: float
    total_tool_calls: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "user_id": self.user_id,
            "state": self.state.value,
            "steps": [s.to_dict() for s in self.steps],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "context_keys": list(self.context.keys()),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "total_cost_usd": self.total_cost_usd,
            "total_tool_calls": self.total_tool_calls,
        }


class Policy(ABC):
    """Abstract policy for workflow governance."""
    
    @abstractmethod
    def evaluate(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[PolicyDecision, str | None]:
        """
        Evaluate an action against the policy.
        
        Returns (decision, reason).
        """
        pass


class CostLimitPolicy(Policy):
    """Policy limiting total workflow cost."""
    
    def __init__(self, max_cost_usd: float = 10.0):
        self.max_cost_usd = max_cost_usd
    
    def evaluate(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[PolicyDecision, str | None]:
        current_cost = context.get("total_cost_usd", 0.0)
        estimated_cost = context.get("estimated_cost_usd", 0.0)
        
        if current_cost + estimated_cost > self.max_cost_usd:
            return (
                PolicyDecision.DENY,
                f"Would exceed cost limit of ${self.max_cost_usd:.2f}"
            )
        
        if current_cost + estimated_cost > self.max_cost_usd * 0.8:
            return (
                PolicyDecision.REQUIRE_APPROVAL,
                f"Approaching cost limit (${current_cost + estimated_cost:.2f} / ${self.max_cost_usd:.2f})"
            )
        
        return (PolicyDecision.ALLOW, None)


class StepLimitPolicy(Policy):
    """Policy limiting number of workflow steps."""
    
    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
    
    def evaluate(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[PolicyDecision, str | None]:
        current_steps = context.get("step_count", 0)
        
        if current_steps >= self.max_steps:
            return (
                PolicyDecision.DENY,
                f"Exceeded maximum steps ({self.max_steps})"
            )
        
        return (PolicyDecision.ALLOW, None)


class SensitiveDataPolicy(Policy):
    """Policy for handling sensitive data access."""
    
    def __init__(self, sensitive_tools: list[str] | None = None):
        self.sensitive_tools = sensitive_tools or [
            "database_write",
            "file_delete",
            "send_email",
            "api_call_external",
        ]
    
    def evaluate(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[PolicyDecision, str | None]:
        tool_name = context.get("tool_name", "")
        
        if tool_name in self.sensitive_tools:
            return (
                PolicyDecision.REQUIRE_APPROVAL,
                f"Tool '{tool_name}' requires human approval"
            )
        
        return (PolicyDecision.ALLOW, None)


class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}
    
    def register(
        self,
        tool: ToolDefinition,
        handler: Callable[..., Any],
    ) -> None:
        """Register a tool with its handler."""
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler
        
        logger.info("tool_registered", tool_name=tool.name)
    
    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name."""
        return self._tools.get(name)
    
    def get_handler(self, name: str) -> Callable[..., Any] | None:
        """Get tool handler by name."""
        return self._handlers.get(name)
    
    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """Get tool definitions formatted for LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]


class WorkflowEngineConfig(BaseModel):
    """Configuration for workflow engine."""
    
    # Execution limits
    max_steps: int = Field(default=50, ge=1)
    max_cost_usd: float = Field(default=10.0, ge=0.0)
    max_duration_seconds: int = Field(default=300, ge=1)
    
    # Checkpointing
    checkpoint_interval_steps: int = Field(default=5, ge=1)
    enable_checkpoints: bool = True
    
    # Policies
    require_approval_for_sensitive: bool = True
    
    # Parallelism
    max_parallel_steps: int = Field(default=5, ge=1)


class AgenticWorkflowEngine:
    """
    Production agent workflow orchestration engine.
    
    Features:
    - Multi-step execution with state management
    - Tool registration and invocation
    - Policy enforcement
    - Checkpointing and recovery
    - Cost tracking
    - Comprehensive audit logging
    """
    
    def __init__(
        self,
        config: WorkflowEngineConfig | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.config = config or WorkflowEngineConfig()
        self.tool_registry = tool_registry or ToolRegistry()
        self._policies: list[Policy] = []
        self._workflows: dict[str, WorkflowExecution] = {}
        self._pending_approvals: dict[str, WorkflowStep] = {}
        
        # Register default policies
        self._policies.append(CostLimitPolicy(self.config.max_cost_usd))
        self._policies.append(StepLimitPolicy(self.config.max_steps))
        if self.config.require_approval_for_sensitive:
            self._policies.append(SensitiveDataPolicy())
    
    def add_policy(self, policy: Policy) -> None:
        """Add a policy to the engine."""
        self._policies.append(policy)
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        user_id: str,
        initial_context: dict[str, Any] | None = None,
    ) -> WorkflowExecution:
        """
        Create a new workflow execution.
        
        Args:
            name: Workflow name
            description: Workflow description
            user_id: User initiating the workflow
            initial_context: Initial context data
        
        Returns:
            WorkflowExecution instance
        """
        workflow_id = self._generate_id("wf")
        
        workflow = WorkflowExecution(
            workflow_id=workflow_id,
            name=name,
            description=description,
            user_id=user_id,
            state=AgentState.IDLE,
            steps=[],
            checkpoints=[],
            context=initial_context or {},
            started_at=datetime.utcnow(),
            completed_at=None,
            total_duration_ms=None,
            total_cost_usd=0.0,
            total_tool_calls=0,
        )
        
        self._workflows[workflow_id] = workflow
        
        logger.info(
            "workflow_created",
            workflow_id=workflow_id,
            name=name,
            user_id=user_id,
        )
        
        return workflow
    
    async def execute_step(
        self,
        workflow_id: str,
        step_type: StepType,
        name: str,
        description: str,
        input_data: dict[str, Any],
        tool_name: str | None = None,
    ) -> WorkflowStep:
        """
        Execute a workflow step.
        
        Args:
            workflow_id: Workflow to execute in
            step_type: Type of step
            name: Step name
            description: Step description
            input_data: Input data for the step
            tool_name: Tool to invoke (for TOOL_CALL steps)
        
        Returns:
            Completed WorkflowStep
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        # Create step
        step_id = self._generate_id("step")
        step = WorkflowStep(
            step_id=step_id,
            step_type=step_type,
            name=name,
            description=description,
            input_data=input_data,
            output_data=None,
            status=AgentState.IDLE,
            started_at=None,
            completed_at=None,
            duration_ms=None,
            error=None,
        )
        
        # Policy check
        policy_context = {
            "step_count": len(workflow.steps),
            "total_cost_usd": workflow.total_cost_usd,
            "tool_name": tool_name or "",
            "user_id": workflow.user_id,
        }
        
        if tool_name:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                policy_context["estimated_cost_usd"] = tool.cost_estimate_usd
        
        for policy in self._policies:
            decision, reason = policy.evaluate(step_type.value, policy_context)
            
            if decision == PolicyDecision.DENY:
                step.status = AgentState.FAILED
                step.error = f"Policy denied: {reason}"
                workflow.steps.append(step)
                
                logger.warning(
                    "step_policy_denied",
                    workflow_id=workflow_id,
                    step_id=step_id,
                    reason=reason,
                )
                return step
            
            elif decision == PolicyDecision.REQUIRE_APPROVAL:
                step.status = AgentState.WAITING_APPROVAL
                workflow.steps.append(step)
                self._pending_approvals[step_id] = step
                
                logger.info(
                    "step_requires_approval",
                    workflow_id=workflow_id,
                    step_id=step_id,
                    reason=reason,
                )
                return step
        
        # Execute step
        step.status = AgentState.EXECUTING
        step.started_at = datetime.utcnow()
        workflow.state = AgentState.EXECUTING
        
        start_time = time.perf_counter()
        
        try:
            if step_type == StepType.TOOL_CALL and tool_name:
                output = await self._invoke_tool(
                    workflow, tool_name, input_data
                )
                step.output_data = {"result": output}
                workflow.total_tool_calls += 1
            
            elif step_type == StepType.REASONING:
                step.output_data = {"reasoning": input_data.get("reasoning", "")}
            
            elif step_type == StepType.CHECKPOINT:
                checkpoint = await self._create_checkpoint(workflow, step_id)
                step.output_data = {"checkpoint_id": checkpoint.checkpoint_id}
            
            else:
                step.output_data = input_data
            
            step.status = AgentState.COMPLETED
            
        except Exception as e:
            step.status = AgentState.FAILED
            step.error = str(e)
            logger.error(
                "step_execution_failed",
                workflow_id=workflow_id,
                step_id=step_id,
                error=str(e),
            )
        
        finally:
            step.completed_at = datetime.utcnow()
            step.duration_ms = (time.perf_counter() - start_time) * 1000
            workflow.steps.append(step)
        
        # Auto-checkpoint if configured
        if (
            self.config.enable_checkpoints and
            len(workflow.steps) % self.config.checkpoint_interval_steps == 0
        ):
            await self._create_checkpoint(workflow, step_id)
        
        logger.info(
            "step_completed",
            workflow_id=workflow_id,
            step_id=step_id,
            status=step.status.value,
            duration_ms=step.duration_ms,
        )
        
        return step
    
    async def _invoke_tool(
        self,
        workflow: WorkflowExecution,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> Any:
        """Invoke a registered tool."""
        tool = self.tool_registry.get_tool(tool_name)
        handler = self.tool_registry.get_handler(tool_name)
        
        if not tool or not handler:
            raise ValueError(f"Tool not found: {tool_name}")
        
        invocation_id = self._generate_id("inv")
        start_time = time.perf_counter()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_handler(handler, parameters),
                timeout=tool.timeout_seconds,
            )
            
            invocation = ToolInvocation(
                invocation_id=invocation_id,
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                success=True,
                error_message=None,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                timestamp=datetime.utcnow(),
            )
            
            # Track cost
            workflow.total_cost_usd += tool.cost_estimate_usd
            
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool {tool_name} timed out after {tool.timeout_seconds}s")
        except Exception as e:
            raise RuntimeError(f"Tool {tool_name} failed: {str(e)}")
    
    async def _execute_handler(
        self,
        handler: Callable[..., Any],
        parameters: dict[str, Any],
    ) -> Any:
        """Execute tool handler (sync or async)."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(**parameters)
        else:
            return handler(**parameters)
    
    async def _create_checkpoint(
        self,
        workflow: WorkflowExecution,
        step_id: str,
    ) -> Checkpoint:
        """Create a workflow checkpoint."""
        checkpoint = Checkpoint(
            checkpoint_id=self._generate_id("ckpt"),
            workflow_id=workflow.workflow_id,
            step_id=step_id,
            state={
                "context": workflow.context.copy(),
                "step_count": len(workflow.steps),
                "total_cost_usd": workflow.total_cost_usd,
            },
            created_at=datetime.utcnow(),
        )
        
        workflow.checkpoints.append(checkpoint)
        
        logger.info(
            "checkpoint_created",
            workflow_id=workflow.workflow_id,
            checkpoint_id=checkpoint.checkpoint_id,
        )
        
        return checkpoint
    
    async def approve_step(
        self,
        step_id: str,
        approver_id: str,
    ) -> WorkflowStep:
        """Approve a pending step."""
        step = self._pending_approvals.get(step_id)
        if not step:
            raise ValueError(f"No pending approval for step: {step_id}")
        
        step.status = AgentState.COMPLETED
        step.completed_at = datetime.utcnow()
        step.output_data = {"approved_by": approver_id}
        
        del self._pending_approvals[step_id]
        
        logger.info(
            "step_approved",
            step_id=step_id,
            approver_id=approver_id,
        )
        
        return step
    
    async def complete_workflow(
        self,
        workflow_id: str,
        success: bool = True,
    ) -> WorkflowExecution:
        """Complete a workflow execution."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow.completed_at = datetime.utcnow()
        workflow.total_duration_ms = (
            workflow.completed_at - workflow.started_at
        ).total_seconds() * 1000
        workflow.state = AgentState.COMPLETED if success else AgentState.FAILED
        
        logger.info(
            "workflow_completed",
            workflow_id=workflow_id,
            state=workflow.state.value,
            total_steps=len(workflow.steps),
            total_cost_usd=workflow.total_cost_usd,
            duration_ms=workflow.total_duration_ms,
        )
        
        return workflow
    
    def get_workflow(self, workflow_id: str) -> WorkflowExecution | None:
        """Get workflow by ID."""
        return self._workflows.get(workflow_id)
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"
