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



# ============================================================================
# GRADIO UI INTERFACE
# ============================================================================


# Create Gradio interface - simple and compatible with FastAPI mounting
def reset_env():
    """Reset environment."""
    try:
        observation = _environment.reset()
        state = _environment.state
        status = f"Environment reset successfully.\nEpisode ID: {state.episode_id}\nStep: {state.step_count}"
        return status, json.dumps(observation.dict(), indent=2)
    except Exception as e:
        return f"Error: {str(e)}", ""

def get_env_state():
    """Get current state."""
    try:
        state = _environment.state
        status = f"Current State\nEpisode ID: {state.episode_id}\nStep: {state.step_count}"
        return status, json.dumps({"episode_id": state.episode_id, "step_count": state.step_count}, indent=2)
    except Exception as e:
        return f"Error: {str(e)}", ""

def execute_step(action_type, classification, category, solution, should_escalate, escalate_reason, message):
    """Execute action in environment."""
    try:
        action_dict = {
            "action_type": action_type or "classify_issue",
            "classification": classification or "billing",
            "category": category or "general",
            "solution": solution or "refund",
            "should_escalate": should_escalate.lower() == "true",
            "escalation_reason": escalate_reason or "customer_request"
        }
        # Optionally include message if your environment expects it
        if message:
            action_dict["message"] = message
        action = SupportAction(**action_dict)
        observation = _environment.step(action)
        status = f"Step executed. Reward: {observation.reward}\nDone: {observation.done}"
        return status, json.dumps(observation.dict(), indent=2)
    except Exception as e:
        return f"Error: {str(e)}", ""




# --- Redesigned Gradio UI to match reference ---

# --- Refactored Gradio UI: Project name at top, no duplicate state/status, Current State below State Observer, simple layout ---
with gr.Blocks(title="Customer Support OpenEnv") as gradio_app:
    gr.Markdown("# Customer Support OpenEnv Playground")
    with gr.Row():
        # Left: Action input only
        with gr.Column(scale=1):
            gr.Markdown("#### Take Action")
            action_type = gr.Textbox(label="Action Type", placeholder="e.g. classify_issue")
            classification = gr.Textbox(label="Classification", placeholder="e.g. billing")
            category = gr.Textbox(label="Category", placeholder="Enter category")
            solution = gr.Textbox(label="Solution", placeholder="Enter solution")
            should_escalate = gr.Textbox(label="Should Escalate", placeholder="true/false")
            escalate_reason = gr.Textbox(label="Escalate Reason", placeholder="Enter escalate reason")
            message = gr.Textbox(label="Message", placeholder="Enter message (optional)")
            step_btn = gr.Button("Step", variant="primary")
            with gr.Row():
                reset_btn = gr.Button("Reset Environment", variant="secondary")
                state_btn = gr.Button("Get State", variant="secondary")

        # Right: State observer and current state
        with gr.Column(scale=1):
            gr.Markdown("#### State Observer")
            observation_box = gr.Textbox(label="Current Observation", interactive=False, lines=6, value="")
            action_history_box = gr.Textbox(label="Action History", interactive=False, lines=6, value="")
            reward_box = gr.Textbox(label="Reward", interactive=False, value="")
            gr.Markdown("#### Current State")
            state_box = gr.Textbox(label="Status / Episode / Step", interactive=False, value="Click Reset to start")

    # Button click handlers
    def step_and_update(*args):
        status, response = execute_step(*args)
        try:
            obs = json.loads(response) if response else {}
            reward = obs.get("reward", "")
            observation = json.dumps(obs.get("observation", {}), indent=2) if obs.get("observation") else ""
            action_hist = json.dumps(args[:7], indent=2)
            state_info = f"Episode: {obs.get('observation', {}).get('episode_id', '-')}, Step: {obs.get('observation', {}).get('step_count', '-')}\nReward: {reward}"
        except Exception:
            observation = ""
            action_hist = ""
            reward = ""
            state_info = status
        return observation, action_hist, str(reward), state_info

    def reset_and_update():
        status, response = reset_env()
        try:
            obs = json.loads(response) if response else {}
            observation = json.dumps(obs, indent=2)
            state_info = f"Episode: {obs.get('episode_id', '-')}, Step: {obs.get('step_count', '-')}"
        except Exception:
            observation = ""
            state_info = status
        return observation, "", "", state_info

    def state_and_update():
        status, response = get_env_state()
        try:
            obs = json.loads(response) if response else {}
            state_info = f"Episode: {obs.get('episode_id', '-')}, Step: {obs.get('step_count', '-')}"
        except Exception:
            state_info = status
        return state_info

    step_btn.click(
        step_and_update,
        inputs=[action_type, classification, category, solution, should_escalate, escalate_reason, message],
        outputs=[observation_box, action_history_box, reward_box, state_box]
    )
    reset_btn.click(reset_and_update, outputs=[observation_box, action_history_box, reward_box, state_box])
    state_btn.click(state_and_update, outputs=[state_box])


# Mount Gradio app at /web after all other routes
gradio_app.queue()
# For HuggingFace Spaces, mount at /web and redirect / to /web
gr.mount_gradio_app(app, gradio_app, path="/web")

from fastapi.responses import RedirectResponse
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/web")


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
