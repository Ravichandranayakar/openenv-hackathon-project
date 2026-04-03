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
from typing import Tuple

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

def reset_environment():
    """Reset the environment and return initial state."""
    observation = _environment.reset()
    state = _environment.state
    
    message = f"Environment Reset\n"
    message += f"Episode ID: {state.episode_id}\n"
    message += f"Support Ticket: {observation.current_ticket['subject']}\n"
    message += f"Ticket ID: {observation.current_ticket['id']}"
    
    return message, ""

def get_current_state():
    """Get the current environment state."""
    state = _environment.state
    message = f"Current State\n"
    message += f"Episode ID: {state.episode_id}\n"
    message += f"Step Count: {state.step_count}\n"
    
    if hasattr(_environment, 'current_ticket') and _environment.current_ticket:
        message += f"\nTicket: {_environment.current_ticket['subject']}"
    
    return message

def take_action(action_type, category_choice, solution_choice, escalate_choice):
    """Execute an action in the environment."""
    try:
        # Build action based on step count
        step = _environment.state.step_count
        
        if step == 0:
            # Reset first
            return "ERROR: Please click 'Reset Environment' first", ""
        
        action_dict = {"action_type": action_type}
        
        if step == 1:  # Classify
            if not category_choice or category_choice == "Select...":
                return "ERROR: Please select a classification", ""
            action_dict["classification"] = category_choice
        elif step == 2:  # Choose Solution
            if not solution_choice or solution_choice == "Select...":
                return "ERROR: Please select a solution", ""
            action_dict["category"] = "general"  # Default category
            action_dict["solution"] = solution_choice
        elif step == 3:  # Escalation
            action_dict["should_escalate"] = escalate_choice == "Yes, escalate"
        elif step == 4:  # Close
            pass
        
        # Execute action
        observation = _environment.step(SupportAction(**action_dict))
        
        # Build response
        result = f"Action Executed\n"
        result += f"Step: {_environment.state.step_count}\n"
        result += f"Reward: +{observation.reward}\n"
        
        if observation.done:
            result += f"\nEpisode Complete!\n"
            result += f"Final Score: {observation.episode_score}"
        
        return result, ""
    except Exception as e:
        return f"ERROR: {str(e)}", ""


# Create Gradio interface
theme = gr.themes.Soft(primary_hue="green", secondary_hue="red")
with gr.Blocks(title="Customer Support OpenEnv", theme=theme) as gradio_app:
    gr.Markdown("# Customer Support OpenEnv Interface")
    gr.Markdown("*Interactive demo for the customer support RL environment*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Control Panel")
            reset_btn = gr.Button("Reset Environment", size="lg", variant="primary")
            state_btn = gr.Button("Get Current State", size="lg", variant="secondary")
        
        with gr.Column(scale=1):
            state_display = gr.Textbox(
                label="State Monitor",
                interactive=False,
                lines=6,
                value="Status: Ready. Click 'Reset Environment' to begin."
            )
    
    gr.Markdown("### Take Action")
    
    with gr.Row():
        action_type = gr.Dropdown(
            choices=[
                "classify_issue",
                "choose_solution",
                "escalate_decision",
                "close_ticket"
            ],
            label="Action Type",
            value="classify_issue"
        )
        category = gr.Dropdown(
            choices=["Select...", "billing", "account", "bug", "feature"],
            label="Classification / Category",
            value="Select..."
        )
        solution = gr.Dropdown(
            choices=["Select...", "refund", "reset_password", "apply_patch", "add_feature"],
            label="Solution",
            value="Select..."
        )
        escalate = gr.Radio(
            choices=["No", "Yes, escalate"],
            label="Escalate?",
            value="No"
        )
    
    action_btn = gr.Button("Execute Action", size="lg", variant="primary")
    action_result = gr.Textbox(
        label="Action Result",
        interactive=False,
        lines=5,
        value="Results will appear here..."
    )
    action_history = gr.Textbox(
        label="Action History",
        interactive=False,
        lines=4,
        value="No actions yet"
    )
    
    # Set up event handlers
    @reset_btn.click(outputs=[state_display, action_history])
    def on_reset():
        return reset_environment()
    
    @state_btn.click(outputs=[state_display])
    def on_state():
        return get_current_state()
    
    @action_btn.click(inputs=[action_type, category, solution, escalate], outputs=[action_result, action_history])
    def on_action(atype, cat, sol, esc):
        return take_action(atype, cat, sol, esc)
    
    gr.Markdown("---")
    gr.Markdown(
        """
        ## How to Use
        
        1. Reset: Click 'Reset Environment' to load a new support ticket
        2. Classify: Select the issue classification and execute
        3. Solve: Choose a solution for the category and execute
        4. Escalate: Decide whether to escalate the ticket
        5. Close: Close the ticket and complete the episode
        
        Earn rewards for correct decisions! Maximum score is 1.0 across 4 steps.
        """
    )


@app.get("/web", include_in_schema=False)
async def web():
    """Web path - serve Gradio UI."""
    return Response(content="", status_code=200)


# Mount Gradio app at /web after all other routes
gradio_app.queue()
app = gr.mount_gradio_app(app, gradio_app, path="/web")


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
