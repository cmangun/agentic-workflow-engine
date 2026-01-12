"""
Policy Engine for Agentic Workflow Engine.

Production-grade policy enforcement for AI agents:
- Rule-based access control
- PHI access policies with audit requirements
- Cost limits and budget enforcement
- Content filtering and safety checks
- Approval workflows for sensitive operations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class PolicyDecision(Enum):
    """Policy evaluation decision."""
    
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    ALLOW_WITH_AUDIT = "allow_with_audit"


class PolicyType(Enum):
    """Types of policies."""
    
    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"
    RATE_LIMIT = "rate_limit"
    CONTENT_SAFETY = "content_safety"
    COST_CONTROL = "cost_control"
    COMPLIANCE = "compliance"


@dataclass
class PolicyCondition:
    """Condition for policy evaluation."""
    
    field: str           # Field to evaluate (e.g., "tool_category", "user_role")
    operator: str        # Comparison operator (eq, ne, in, not_in, contains, regex)
    value: Any           # Expected value
    
    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        actual = self._get_nested_value(context, self.field)
        
        if actual is None:
            return False
        
        if self.operator == "eq":
            return actual == self.value
        elif self.operator == "ne":
            return actual != self.value
        elif self.operator == "in":
            return actual in self.value
        elif self.operator == "not_in":
            return actual not in self.value
        elif self.operator == "contains":
            return self.value in actual
        elif self.operator == "regex":
            return bool(re.match(self.value, str(actual)))
        elif self.operator == "gt":
            return actual > self.value
        elif self.operator == "lt":
            return actual < self.value
        elif self.operator == "gte":
            return actual >= self.value
        elif self.operator == "lte":
            return actual <= self.value
        
        return False
    
    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation."""
        keys = path.split(".")
        value = data
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        
        return value
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
        }


@dataclass
class Policy:
    """Policy definition."""
    
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    conditions: list[PolicyCondition]
    decision: PolicyDecision
    priority: int = 100              # Lower = higher priority
    enabled: bool = True
    require_audit: bool = False
    message: str = ""                # Message to include in decision
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, context: dict[str, Any]) -> PolicyDecision | None:
        """
        Evaluate policy against context.
        
        Returns decision if all conditions match, None otherwise.
        """
        if not self.enabled:
            return None
        
        # All conditions must match
        for condition in self.conditions:
            if not condition.evaluate(context):
                return None
        
        return self.decision
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type.value,
            "conditions": [c.to_dict() for c in self.conditions],
            "decision": self.decision.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "require_audit": self.require_audit,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass
class PolicyEvaluationResult:
    """Result of policy evaluation."""
    
    decision: PolicyDecision
    matched_policies: list[Policy]
    evaluation_time_ms: float
    context_hash: str
    audit_required: bool
    messages: list[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "matched_policy_count": len(self.matched_policies),
            "matched_policies": [p.policy_id for p in self.matched_policies],
            "evaluation_time_ms": self.evaluation_time_ms,
            "audit_required": self.audit_required,
            "messages": self.messages,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ApprovalRequest:
    """Request for policy approval."""
    
    request_id: str
    policy_id: str
    context: dict[str, Any]
    requested_by: str
    requested_at: datetime
    status: str = "pending"  # pending, approved, denied, expired
    approved_by: str | None = None
    approved_at: datetime | None = None
    expiry_hours: int = 24
    reason: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "policy_id": self.policy_id,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "status": self.status,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "reason": self.reason,
        }


class PolicyEngine:
    """
    Production policy engine for AI agents.
    
    Features:
    - Rule-based policy evaluation
    - Priority-based policy ordering
    - Approval workflow support
    - Audit logging
    - Healthcare-specific policies
    """
    
    def __init__(self):
        self._policies: dict[str, Policy] = {}
        self._approval_requests: dict[str, ApprovalRequest] = {}
        self._audit_log: list[dict[str, Any]] = []
        
        # Register default healthcare policies
        self._register_default_policies()
    
    def add_policy(self, policy: Policy) -> None:
        """Add a policy."""
        self._policies[policy.policy_id] = policy
        logger.info(
            "policy_added",
            policy_id=policy.policy_id,
            name=policy.name,
            type=policy.policy_type.value,
        )
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            return True
        return False
    
    def get_policy(self, policy_id: str) -> Policy | None:
        """Get policy by ID."""
        return self._policies.get(policy_id)
    
    def list_policies(
        self,
        policy_type: PolicyType | None = None,
        enabled_only: bool = True,
    ) -> list[Policy]:
        """List policies."""
        policies = list(self._policies.values())
        
        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]
        
        if enabled_only:
            policies = [p for p in policies if p.enabled]
        
        return sorted(policies, key=lambda p: p.priority)
    
    def evaluate(
        self,
        context: dict[str, Any],
    ) -> PolicyEvaluationResult:
        """
        Evaluate all policies against context.
        
        Args:
            context: Evaluation context containing:
                - tool_id: Tool being invoked
                - tool_category: Tool category
                - agent_id: Agent making request
                - user_role: User role
                - data_classification: Data sensitivity
                - estimated_cost: Estimated cost
                - parameters: Tool parameters
                
        Returns:
            Policy evaluation result
        """
        import hashlib
        import time
        
        start_time = time.time()
        
        # Generate context hash for auditing
        context_str = str(sorted(context.items()))
        context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
        
        # Get sorted policies
        policies = sorted(
            [p for p in self._policies.values() if p.enabled],
            key=lambda p: p.priority,
        )
        
        matched_policies: list[Policy] = []
        messages: list[str] = []
        audit_required = False
        final_decision = PolicyDecision.ALLOW
        
        for policy in policies:
            decision = policy.evaluate(context)
            
            if decision:
                matched_policies.append(policy)
                
                if policy.message:
                    messages.append(policy.message)
                
                if policy.require_audit:
                    audit_required = True
                
                # Apply decision based on priority
                if decision == PolicyDecision.DENY:
                    final_decision = PolicyDecision.DENY
                    break  # Deny takes precedence
                elif decision == PolicyDecision.REQUIRE_APPROVAL:
                    if final_decision != PolicyDecision.DENY:
                        final_decision = PolicyDecision.REQUIRE_APPROVAL
                elif decision == PolicyDecision.ALLOW_WITH_AUDIT:
                    audit_required = True
        
        evaluation_time_ms = (time.time() - start_time) * 1000
        
        result = PolicyEvaluationResult(
            decision=final_decision,
            matched_policies=matched_policies,
            evaluation_time_ms=evaluation_time_ms,
            context_hash=context_hash,
            audit_required=audit_required,
            messages=messages,
        )
        
        # Log evaluation
        self._log_evaluation(context, result)
        
        logger.info(
            "policy_evaluation_completed",
            decision=final_decision.value,
            matched_count=len(matched_policies),
            audit_required=audit_required,
            evaluation_time_ms=evaluation_time_ms,
        )
        
        return result
    
    def request_approval(
        self,
        policy_id: str,
        context: dict[str, Any],
        requested_by: str,
        reason: str = "",
    ) -> ApprovalRequest:
        """Create approval request for a policy."""
        import hashlib
        
        request_id = f"apr_{hashlib.sha256(f'{policy_id}:{datetime.utcnow().isoformat()}'.encode()).hexdigest()[:12]}"
        
        request = ApprovalRequest(
            request_id=request_id,
            policy_id=policy_id,
            context=context,
            requested_by=requested_by,
            requested_at=datetime.utcnow(),
            reason=reason,
        )
        
        self._approval_requests[request_id] = request
        
        logger.info(
            "approval_requested",
            request_id=request_id,
            policy_id=policy_id,
            requested_by=requested_by,
        )
        
        return request
    
    def approve_request(
        self,
        request_id: str,
        approved_by: str,
    ) -> ApprovalRequest | None:
        """Approve a pending request."""
        request = self._approval_requests.get(request_id)
        
        if not request or request.status != "pending":
            return None
        
        request.status = "approved"
        request.approved_by = approved_by
        request.approved_at = datetime.utcnow()
        
        logger.info(
            "approval_granted",
            request_id=request_id,
            approved_by=approved_by,
        )
        
        return request
    
    def deny_request(
        self,
        request_id: str,
        denied_by: str,
        reason: str = "",
    ) -> ApprovalRequest | None:
        """Deny a pending request."""
        request = self._approval_requests.get(request_id)
        
        if not request or request.status != "pending":
            return None
        
        request.status = "denied"
        request.approved_by = denied_by
        request.approved_at = datetime.utcnow()
        request.reason = reason
        
        logger.info(
            "approval_denied",
            request_id=request_id,
            denied_by=denied_by,
        )
        
        return request
    
    def get_audit_log(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get policy evaluation audit log."""
        return self._audit_log[-limit:]
    
    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------
    
    def _log_evaluation(
        self,
        context: dict[str, Any],
        result: PolicyEvaluationResult,
    ) -> None:
        """Log policy evaluation for audit."""
        self._audit_log.append({
            "timestamp": result.timestamp.isoformat(),
            "context_hash": result.context_hash,
            "decision": result.decision.value,
            "matched_policies": [p.policy_id for p in result.matched_policies],
            "audit_required": result.audit_required,
            "agent_id": context.get("agent_id"),
            "tool_id": context.get("tool_id"),
        })
        
        # Keep last 10000 entries
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]
    
    def _register_default_policies(self) -> None:
        """Register default healthcare policies."""
        
        # PHI Access requires audit
        self.add_policy(Policy(
            policy_id="phi_access_audit",
            name="PHI Access Audit",
            description="Require audit logging for all PHI access",
            policy_type=PolicyType.DATA_PROTECTION,
            conditions=[
                PolicyCondition(
                    field="tool_category",
                    operator="eq",
                    value="phi_access",
                ),
            ],
            decision=PolicyDecision.ALLOW_WITH_AUDIT,
            priority=10,
            require_audit=True,
            message="PHI access logged for HIPAA compliance",
        ))
        
        # Admin operations require approval
        self.add_policy(Policy(
            policy_id="admin_approval",
            name="Admin Operations Approval",
            description="Require approval for administrative operations",
            policy_type=PolicyType.ACCESS_CONTROL,
            conditions=[
                PolicyCondition(
                    field="tool_category",
                    operator="eq",
                    value="admin",
                ),
            ],
            decision=PolicyDecision.REQUIRE_APPROVAL,
            priority=5,
            message="Administrative operations require explicit approval",
        ))
        
        # Cost limit enforcement
        self.add_policy(Policy(
            policy_id="cost_limit",
            name="Cost Limit Enforcement",
            description="Deny requests exceeding cost threshold",
            policy_type=PolicyType.COST_CONTROL,
            conditions=[
                PolicyCondition(
                    field="estimated_cost",
                    operator="gt",
                    value=1.0,  # $1 per request limit
                ),
            ],
            decision=PolicyDecision.DENY,
            priority=1,
            message="Request denied: estimated cost exceeds $1.00 limit",
        ))
        
        # Block external API calls for restricted roles
        self.add_policy(Policy(
            policy_id="external_api_restriction",
            name="External API Restriction",
            description="Restrict external API access for limited roles",
            policy_type=PolicyType.ACCESS_CONTROL,
            conditions=[
                PolicyCondition(
                    field="tool_category",
                    operator="eq",
                    value="external_api",
                ),
                PolicyCondition(
                    field="user_role",
                    operator="in",
                    value=["guest", "limited"],
                ),
            ],
            decision=PolicyDecision.DENY,
            priority=20,
            message="External API access not permitted for this role",
        ))
        
        # Content safety check
        self.add_policy(Policy(
            policy_id="content_safety",
            name="Content Safety Filter",
            description="Block potentially harmful content patterns",
            policy_type=PolicyType.CONTENT_SAFETY,
            conditions=[
                PolicyCondition(
                    field="parameters.query",
                    operator="regex",
                    value=r".*(hack|exploit|bypass|inject).*",
                ),
            ],
            decision=PolicyDecision.DENY,
            priority=1,
            message="Request blocked by content safety filter",
        ))
        
        # Allow all data read operations
        self.add_policy(Policy(
            policy_id="allow_data_read",
            name="Allow Data Read",
            description="Allow read-only data access",
            policy_type=PolicyType.ACCESS_CONTROL,
            conditions=[
                PolicyCondition(
                    field="tool_category",
                    operator="eq",
                    value="data_read",
                ),
            ],
            decision=PolicyDecision.ALLOW,
            priority=100,
            message="Read access permitted",
        ))
        
        # Data write requires elevated role
        self.add_policy(Policy(
            policy_id="data_write_control",
            name="Data Write Control",
            description="Control data modification access",
            policy_type=PolicyType.DATA_PROTECTION,
            conditions=[
                PolicyCondition(
                    field="tool_category",
                    operator="eq",
                    value="data_write",
                ),
                PolicyCondition(
                    field="user_role",
                    operator="not_in",
                    value=["admin", "data_manager", "clinician"],
                ),
            ],
            decision=PolicyDecision.DENY,
            priority=15,
            message="Data modification requires elevated privileges",
        ))
