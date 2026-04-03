"""
Customer Support OpenEnv Server - HTTP/WebSocket interface.

Provides OpenEnv-compliant REST API for customer support environment.
Standard endpoints (/reset, /step, /state, /schema, /ws) provided by manual routing.
Uses SINGLE environment instance maintained across all requests (not using create_app).
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.requests import Request
import json
import gradio as gr

# Support both in-repo and standalone imports
try:
    from ..models import SupportAction, SupportObservation
    from .customer_support_environment import CustomerSupportEnvironment
except ImportError:
    import sys
    from pathlib import Path
    root = Path(__file__).parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models import SupportAction, SupportObservation
    from my_env.server.customer_support_environment import CustomerSupportEnvironment


# ============================================================================
# CRITICAL: Create SINGLE environment instance that persists across HTTP requests
# ============================================================================
_environment = CustomerSupportEnvironment()

app = FastAPI(
    title="Customer Support OpenEnv",
    description="""
**Customer Support Environment** - An AI training environment for handling support tickets.

Agents learn to:
- **Classify issues** (billing, account, bug, feature)
- **Choose solutions** (pick the right action for each category)
- **Make escalation decisions** (when to escalate vs. close)
- **Close tickets** (finalize with proper rewards)

## How to Use

1. Call **`POST /reset`** to start a new episode and load a random support ticket
2. Call **`POST /step`** repeatedly with your agent's actions
3. Episode ends when the observation returns `done: true`
4. Call **`GET /state`** anytime to check current state

## Reward Structure

| Step | Max Reward |
|------|-----------|
| 1. Classify Issue | 0.2 |
| 2. Choose Solution | 0.3 |
| 3. Escalation Decision | 0.3 |
| 4. Close Ticket | 0.2 |
| **Total** | **1.0** |

## Action Format for `/step`

```json
{
  "action": {
    "action_type": "classify_issue|choose_solution|escalate_decision|close_ticket",
    "classification": "billing|account|bug|feature",
    "category": "category_name",
    "solution": "solution_name",
    "should_escalate": true|false
  }
}
```
""",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Environment Control",
            "description": "Core operations for environment interaction (reset, step, state)",
        },
        {
            "name": "Schema & Metadata",
            "description": "Get JSON schemas and environment information",
        },
        {
            "name": "Tasks",
            "description": "Task difficulty levels (Easy, Medium, Hard)",
        },
        {
            "name": "Health",
            "description": "Health checks and status",
        },
    ],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic validation errors into readable format."""
    errors = []
    for error in exc.errors():
        field = error.get("loc", [])[-1] if error.get("loc") else "unknown"
        error_type = error.get("type", "unknown")
        
        if error_type == "missing":
            message = f"Missing required field: {field}"
        elif error_type == "string_type":
            message = f"Field '{field}' must be a string"
        elif error_type == "enum":
            message = f"Field '{field}' has an invalid value"
        else:
            message = f"Field '{field}': {error.get('msg', 'invalid value')}"
        
        errors.append({"field": field, "message": message})
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": errors,
            "hint": "Please check that all required fields are filled in correctly."
        }
    )


# ============================================================================
# MANUAL ENDPOINT ROUTING (using persistent environment instance)
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root path returns blank response."""
    return Response(content="", status_code=200)


# ============================================================================
# GRADIO UI INTERFACE
# ============================================================================


# Create Gradio interface - simple and compatible with FastAPI mounting
def get_status(action: str) -> str:
    """Get environment status."""
    try:
        state = _environment.state
        if action == "Status":
            result = "Customer Support Environment\n"
            result += f"Status: Running\n"
            result += f"Episode ID: {state.episode_id}\n"
            result += f"Step Count: {state.step_count}\n"
        elif action == "Reset":
            obs = _environment.reset()
            state = _environment.state
            result = f"Environment Reset\n"
            result += f"Episode ID: {state.episode_id}\n"
            result += f"New Ticket: {obs.message}\n"
        else:
            result = "Available API Endpoints:\n"
            result += "POST /reset - Start new episode\n"
            result += "POST /step - Execute action\n"
            result += "GET /state - Check state\n"
            result += "GET /schema - Get action schema\n"
            result += "GET /health - Health check\n"
            result += "POST /tasks - Available tasks"
        return result
    except Exception as e:
        return f"Error: {str(e)}"


gradio_app = gr.Interface(
    fn=get_status,
    inputs=gr.Dropdown(choices=["Status", "Reset", "API"], label="Select"),
    outputs="text",
    title="Customer Support OpenEnv Demo",
    description="Use API endpoints for full interaction"
)


# Mount Gradio app at /web after all other routes
gradio_app.queue()
gr.mount_gradio_app(app, gradio_app, path="/web")


@app.post("/reset", tags=["Environment Control"])
async def reset_endpoint():
    """Reset the environment and load a new support ticket."""
    observation = _environment.reset()
    return {
        "observation": observation.dict(),
        "reward": 0.0,
        "done": False
    }


@app.post("/step", tags=["Environment Control"])
async def step_endpoint(request_body: dict):
    """Execute an action in the environment."""
    try:
        # Extract action from request
        action_dict = request_body.get("action")
        if action_dict is None:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Validation Error",
                    "details": [{"field": "action", "message": "Missing required field: action"}],
                    "hint": "Please check that all required fields are filled in correctly."
                }
            )
        
        # Convert to SupportAction
        action = SupportAction(**action_dict)
        
        # Execute action
        observation = _environment.step(action)
        
        return {
            "observation": observation.dict(),
            "reward": observation.reward,
            "done": observation.done
        }
    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "details": [{"field": "action", "message": str(e)}],
                "hint": "Please check that all required fields are filled in correctly."
            }
        )


@app.get("/state", tags=["Environment Control"])
async def state_endpoint():
    """Get current environment state."""
    state = _environment.state
    return {
        "episode_id": state.episode_id,
        "step_count": state.step_count
    }


@app.get("/schema", tags=["Schema & Metadata"])
async def schema_endpoint():
    """Get JSON schemas for actions, observations, and state."""
    return {
        "action": SupportAction.model_json_schema(),
        "observation": SupportObservation.model_json_schema(),
        "state": {"type": "object", "properties": {
            "episode_id": {"type": "string"},
            "step_count": {"type": "integer"}
        }}
    }


@app.post("/tasks", tags=["Tasks"])
async def tasks_endpoint():
    """List available task difficulties."""
    return {
        "tasks": [
            {"id": 1, "name": "Easy", "description": "Simple ticket classification"},
            {"id": 2, "name": "Medium", "description": "Mixed ticket types with escalation"},
            {"id": 3, "name": "Hard", "description": "Complex cases requiring expertise"}
        ]
    }


@app.get("/health", tags=["Health"])
async def health_endpoint():
    """Health check endpoint."""
    return {"status": "healthy"}
