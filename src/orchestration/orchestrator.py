"""
Agentic Workflow Orchestrator

Production orchestration for AI agents supporting:
- Multi-step reasoning with checkpoints
- Tool execution with policy controls
- State management and recovery
- Audit logging for compliance
- Human-in-the-loop escalation
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class StepStatus(str, Enum):
    """Status of a workflow step."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_APPROVAL = "waiting_approval"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    """Overall workflow status."""
    
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_HUMAN = "waiting_human"


class PolicyAction(str, Enum):
    """Actions from policy evaluation."""
    
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    MODIFY = "modify"


@dataclass
class ToolCall:
    """A tool call within a step."""
    
    tool_name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
        }


@dataclass
class StepResult:
    """Result of a workflow step."""
    
    step_id: str
    status: StepStatus
    output: Any
    tool_calls: list[ToolCall]
    reasoning: str | None
    started_at: datetime
    completed_at: datetime
    duration_ms: float
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "output": str(self.output)[:1000] if self.output else None,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "reasoning": self.reasoning,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow recovery."""
    
    checkpoint_id: str
    workflow_id: str
    step_index: int
    state: dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "step_index": self.step_index,
            "state_keys": list(self.state.keys()),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class PolicyEvaluation:
    """Result of policy evaluation."""
    
    action: PolicyAction
    reason: str
    modified_input: Any | None = None
    required_approvers: list[str] | None = None


class Tool(ABC):
    """Abstract base class for tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool."""
        pass


class Policy(ABC):
    """Abstract base class for policies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> PolicyEvaluation:
        """Evaluate policy for a tool call."""
        pass


class CostLimitPolicy(Policy):
    """Policy to enforce cost limits."""
    
    def __init__(self, max_cost_per_workflow: float = 10.0):
        self.max_cost = max_cost_per_workflow
    
    @property
    def name(self) -> str:
        return "cost_limit"
    
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> PolicyEvaluation:
        current_cost = context.get("total_cost", 0.0)
        estimated_cost = context.get("estimated_cost", 0.0)
        
        if current_cost + estimated_cost > self.max_cost:
            return PolicyEvaluation(
                action=PolicyAction.DENY,
                reason=f"Would exceed cost limit: ${current_cost + estimated_cost:.2f} > ${self.max_cost:.2f}",
            )
        
        return PolicyEvaluation(
            action=PolicyAction.ALLOW,
            reason="Within cost limits",
        )


class PHIAccessPolicy(Policy):
    """Policy to control PHI access."""
    
    PHI_TOOLS = ["patient_lookup", "medical_record_access", "prescription_history"]
    
    @property
    def name(self) -> str:
        return "phi_access"
    
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> PolicyEvaluation:
        if tool_name in self.PHI_TOOLS:
            user_roles = context.get("user_roles", [])
            
            if "healthcare_provider" not in user_roles and "admin" not in user_roles:
                return PolicyEvaluation(
                    action=PolicyAction.DENY,
                    reason="PHI access requires healthcare_provider or admin role",
                )
            
            # Log access
            logger.info(
                "phi_access_granted",
                tool=tool_name,
                user_id=context.get("user_id"),
            )
        
        return PolicyEvaluation(
            action=PolicyAction.ALLOW,
            reason="Access permitted",
        )


class HumanApprovalPolicy(Policy):
    """Policy requiring human approval for sensitive operations."""
    
    SENSITIVE_OPERATIONS = ["delete", "modify_treatment", "override_alert"]
    
    @property
    def name(self) -> str:
        return "human_approval"
    
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> PolicyEvaluation:
        operation = arguments.get("operation", "")
        
        if operation in self.SENSITIVE_OPERATIONS:
            return PolicyEvaluation(
                action=PolicyAction.REQUIRE_APPROVAL,
                reason=f"Operation '{operation}' requires human approval",
                required_approvers=["supervisor", "compliance_officer"],
            )
        
        return PolicyEvaluation(
            action=PolicyAction.ALLOW,
            reason="No approval required",
        )


@dataclass
class WorkflowStep:
    """Definition of a workflow step."""
    
    step_id: str
    name: str
    description: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    condition: Callable[[dict], bool] | None = None
    on_failure: str = "stop"  # "stop", "skip", "retry"
    max_retries: int = 3
    timeout_seconds: int = 300


@dataclass
class Workflow:
    """A complete workflow definition."""
    
    workflow_id: str
    name: str
    description: str
    steps: list[WorkflowStep]
    created_at: datetime
    created_by: str
    max_duration_seconds: int = 3600
    enable_checkpoints: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "num_steps": len(self.steps),
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "max_duration_seconds": self.max_duration_seconds,
        }


@dataclass
class WorkflowExecution:
    """A workflow execution instance."""
    
    execution_id: str
    workflow: Workflow
    status: WorkflowStatus
    current_step_index: int
    context: dict[str, Any]
    step_results: list[StepResult]
    checkpoints: list[WorkflowCheckpoint]
    started_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow.workflow_id,
            "workflow_name": self.workflow.name,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "total_steps": len(self.workflow.steps),
            "step_results": [r.to_dict() for r in self.step_results],
            "checkpoint_count": len(self.checkpoints),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


class WorkflowOrchestratorConfig(BaseModel):
    """Configuration for workflow orchestrator."""
    
    max_concurrent_workflows: int = Field(default=10, ge=1)
    default_step_timeout_seconds: int = Field(default=300, ge=1)
    checkpoint_interval_steps: int = Field(default=1, ge=1)
    enable_audit_logging: bool = True
    retry_backoff_base_seconds: float = Field(default=1.0, ge=0.1)
    max_workflow_duration_seconds: int = Field(default=3600, ge=60)


class WorkflowOrchestrator:
    """
    Production workflow orchestrator for AI agents.
    
    Features:
    - Multi-step workflow execution
    - Tool registration and execution
    - Policy enforcement
    - Checkpoint and recovery
    - Human-in-the-loop support
    - Comprehensive audit logging
    """
    
    def __init__(self, config: WorkflowOrchestratorConfig | None = None):
        self.config = config or WorkflowOrchestratorConfig()
        self._tools: dict[str, Tool] = {}
        self._policies: list[Policy] = []
        self._workflows: dict[str, Workflow] = {}
        self._executions: dict[str, WorkflowExecution] = {}
        self._pending_approvals: dict[str, dict[str, Any]] = {}
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool for use in workflows."""
        self._tools[tool.name] = tool
        logger.info("tool_registered", tool_name=tool.name)
    
    def register_policy(self, policy: Policy) -> None:
        """Register a policy for enforcement."""
        self._policies.append(policy)
        logger.info("policy_registered", policy_name=policy.name)
    
    def create_workflow(
        self,
        name: str,
        description: str,
        steps: list[WorkflowStep],
        created_by: str,
    ) -> Workflow:
        """Create a new workflow definition."""
        workflow = Workflow(
            workflow_id=str(uuid.uuid4()),
            name=name,
            description=description,
            steps=steps,
            created_at=datetime.utcnow(),
            created_by=created_by,
        )
        
        self._workflows[workflow.workflow_id] = workflow
        
        logger.info(
            "workflow_created",
            workflow_id=workflow.workflow_id,
            name=name,
            num_steps=len(steps),
        )
        
        return workflow
    
    async def execute_workflow(
        self,
        workflow_id: str,
        initial_context: dict[str, Any] | None = None,
    ) -> WorkflowExecution:
        """
        Execute a workflow.
        
        Args:
            workflow_id: Workflow to execute
            initial_context: Initial context variables
        
        Returns:
            WorkflowExecution with results
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        execution = WorkflowExecution(
            execution_id=str(uuid.uuid4()),
            workflow=workflow,
            status=WorkflowStatus.RUNNING,
            current_step_index=0,
            context=initial_context or {},
            step_results=[],
            checkpoints=[],
            started_at=datetime.utcnow(),
        )
        
        self._executions[execution.execution_id] = execution
        
        logger.info(
            "workflow_execution_started",
            execution_id=execution.execution_id,
            workflow_id=workflow_id,
        )
        
        try:
            await self._run_workflow(execution)
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            logger.exception(
                "workflow_execution_failed",
                execution_id=execution.execution_id,
                error=str(e),
            )
        finally:
            execution.completed_at = datetime.utcnow()
        
        return execution
    
    async def _run_workflow(self, execution: WorkflowExecution) -> None:
        """Run workflow steps."""
        workflow = execution.workflow
        
        for i, step in enumerate(workflow.steps):
            execution.current_step_index = i
            
            # Check workflow timeout
            elapsed = (datetime.utcnow() - execution.started_at).total_seconds()
            if elapsed > workflow.max_duration_seconds:
                execution.status = WorkflowStatus.FAILED
                execution.error = "Workflow timeout exceeded"
                return
            
            # Check condition
            if step.condition and not step.condition(execution.context):
                execution.step_results.append(StepResult(
                    step_id=step.step_id,
                    status=StepStatus.SKIPPED,
                    output=None,
                    tool_calls=[],
                    reasoning="Condition not met",
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    duration_ms=0,
                ))
                continue
            
            # Execute step
            result = await self._execute_step(step, execution.context)
            execution.step_results.append(result)
            
            # Update context with result
            execution.context[f"step_{step.step_id}_result"] = result.output
            
            # Handle failure
            if result.status == StepStatus.FAILED:
                if step.on_failure == "stop":
                    execution.status = WorkflowStatus.FAILED
                    execution.error = f"Step {step.step_id} failed: {result.error}"
                    return
                elif step.on_failure == "skip":
                    continue
            
            # Handle waiting for approval
            if result.status == StepStatus.WAITING_APPROVAL:
                execution.status = WorkflowStatus.WAITING_HUMAN
                return
            
            # Create checkpoint
            if workflow.enable_checkpoints and (i + 1) % self.config.checkpoint_interval_steps == 0:
                checkpoint = self._create_checkpoint(execution)
                execution.checkpoints.append(checkpoint)
        
        execution.status = WorkflowStatus.COMPLETED
        
        logger.info(
            "workflow_execution_completed",
            execution_id=execution.execution_id,
            num_steps_completed=len(execution.step_results),
        )
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
    ) -> StepResult:
        """Execute a single workflow step."""
        import time
        
        started_at = datetime.utcnow()
        start_time = time.perf_counter()
        tool_calls: list[ToolCall] = []
        
        try:
            if step.tool_name:
                # Check policies
                policy_result = self._evaluate_policies(
                    step.tool_name,
                    step.tool_arguments or {},
                    context,
                )
                
                if policy_result.action == PolicyAction.DENY:
                    return StepResult(
                        step_id=step.step_id,
                        status=StepStatus.FAILED,
                        output=None,
                        tool_calls=[],
                        reasoning=None,
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                        duration_ms=(time.perf_counter() - start_time) * 1000,
                        error=f"Policy denied: {policy_result.reason}",
                    )
                
                if policy_result.action == PolicyAction.REQUIRE_APPROVAL:
                    return StepResult(
                        step_id=step.step_id,
                        status=StepStatus.WAITING_APPROVAL,
                        output=None,
                        tool_calls=[],
                        reasoning=policy_result.reason,
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                        duration_ms=(time.perf_counter() - start_time) * 1000,
                    )
                
                # Execute tool
                tool = self._tools.get(step.tool_name)
                if not tool:
                    raise ValueError(f"Tool not found: {step.tool_name}")
                
                tool_call = ToolCall(
                    tool_name=step.tool_name,
                    arguments=step.tool_arguments or {},
                    started_at=datetime.utcnow(),
                )
                
                try:
                    result = await asyncio.wait_for(
                        tool.execute(**(step.tool_arguments or {})),
                        timeout=step.timeout_seconds,
                    )
                    tool_call.result = result
                except asyncio.TimeoutError:
                    tool_call.error = "Tool execution timeout"
                    raise
                except Exception as e:
                    tool_call.error = str(e)
                    raise
                finally:
                    tool_call.completed_at = datetime.utcnow()
                    if tool_call.started_at:
                        tool_call.duration_ms = (
                            tool_call.completed_at - tool_call.started_at
                        ).total_seconds() * 1000
                
                tool_calls.append(tool_call)
                output = result
            else:
                output = None
            
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output=output,
                tool_calls=tool_calls,
                reasoning=None,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                output=None,
                tool_calls=tool_calls,
                reasoning=None,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )
    
    def _evaluate_policies(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> PolicyEvaluation:
        """Evaluate all policies for a tool call."""
        for policy in self._policies:
            result = policy.evaluate(tool_name, arguments, context)
            
            if result.action in (PolicyAction.DENY, PolicyAction.REQUIRE_APPROVAL):
                logger.info(
                    "policy_evaluation",
                    policy=policy.name,
                    action=result.action.value,
                    reason=result.reason,
                )
                return result
        
        return PolicyEvaluation(
            action=PolicyAction.ALLOW,
            reason="All policies passed",
        )
    
    def _create_checkpoint(self, execution: WorkflowExecution) -> WorkflowCheckpoint:
        """Create a checkpoint for recovery."""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            workflow_id=execution.workflow.workflow_id,
            step_index=execution.current_step_index,
            state=execution.context.copy(),
            created_at=datetime.utcnow(),
        )
        
        logger.debug(
            "checkpoint_created",
            checkpoint_id=checkpoint.checkpoint_id,
            step_index=checkpoint.step_index,
        )
        
        return checkpoint
    
    def get_execution_status(self, execution_id: str) -> WorkflowExecution | None:
        """Get execution status."""
        return self._executions.get(execution_id)
    
    def list_executions(
        self,
        workflow_id: str | None = None,
        status: WorkflowStatus | None = None,
    ) -> list[WorkflowExecution]:
        """List workflow executions."""
        executions = list(self._executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow.workflow_id == workflow_id]
        if status:
            executions = [e for e in executions if e.status == status]
        
        return sorted(executions, key=lambda e: e.started_at, reverse=True)
