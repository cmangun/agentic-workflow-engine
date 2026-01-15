# Agentic Workflow Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade agent orchestration engine with multi-step reasoning, policy enforcement, and comprehensive audit trails.**

## Business Impact

- **Deterministic execution** with checkpointing and rollback
- **Cost controls** with per-request and per-user limits
- **Policy enforcement** for sensitive operations
- **100% audit coverage** for compliance requirements

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Request  │────▶│   Workflow       │────▶│   Policy        │
│                 │     │   Engine         │     │   Enforcement   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │   Tool Registry  │◀─────────────┘
                        └────────┬─────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Checkpoints    │    │  Audit Logger   │    │   Cost Tracker  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

### 🔄 Multi-Step Orchestration
- Sequential and parallel step execution
- Automatic checkpointing for recovery
- State management across steps
- Timeout and retry policies

### Policy Enforcement
- Cost limits (per-request, hourly, daily)
- Step count limits
- Sensitive operation approval gates
- Custom policy plugins

### 🔧 Tool Registry
- Dynamic tool registration
- Permission-based access control
- Cost estimation per tool
- Timeout management

## Quick Start

```python
from src.orchestration.workflow_engine import (
    AgenticWorkflowEngine, ToolRegistry, ToolDefinition, StepType
)

# Initialize engine
engine = AgenticWorkflowEngine()

# Register tools
engine.tool_registry.register(
    ToolDefinition(
        name="search_documents",
        description="Search internal documents",
        parameters={"query": {"type": "string"}},
        required_permissions=["read:documents"],
        cost_estimate_usd=0.01,
    ),
    handler=search_documents_handler,
)

# Create and execute workflow
workflow = await engine.create_workflow(
    name="Research Task",
    description="Find relevant documents",
    user_id="user_123",
)

step = await engine.execute_step(
    workflow_id=workflow.workflow_id,
    step_type=StepType.TOOL_CALL,
    name="Search",
    description="Search for relevant documents",
    input_data={"query": "diabetes guidelines"},
    tool_name="search_documents",
)

await engine.complete_workflow(workflow.workflow_id)
```

## Project Structure

```
agentic-workflow-engine/
├── src/
│   ├── orchestration/
│   │   └── workflow_engine.py   # Core orchestration
│   ├── tools/
│   │   └── registry.py          # Tool management
│   ├── policies/
│   │   └── enforcement.py       # Policy plugins
│   └── checkpoints/
│       └── store.py             # State persistence
├── tests/
└── examples/
```

## Author

**Christopher Mangun** - [github.com/cmangun](https://github.com/cmangun)
