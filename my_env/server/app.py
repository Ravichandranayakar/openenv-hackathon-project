"""
FastAPI application for the Customer Support OpenEnv Environment.
Uses OpenEnv create_app for stable REST API + mount custom Gradio UI at root.

Key Design:
1. create_app() builds the FastAPI app with /reset, /step, /state, /health, /schema endpoints
2. Custom Gradio UI is mounted at / to completely replace OpenEnv's default UI
3. Gradio UI makes HTTP calls to the API endpoints internally
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Install: pip install 'openenv-core[core]>=0.2.1'"
    ) from e

import gradio as gr
from fastapi import FastAPI, Body
from typing import Dict, Any

try:
    from ..models import SupportAction, SupportObservation
    from .customer_support_environment import CustomerSupportEnvironment
    from .gradio_ui import build_gradio_app
except ImportError:
    from models import SupportAction, SupportObservation
    from my_env.server.customer_support_environment import CustomerSupportEnvironment
    from my_env.server.gradio_ui import build_gradio_app

# Create the app with OpenEnv's REST API endpoints (/reset, /step, /state, /health, /schema, etc.)
app = create_app(
    CustomerSupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="customer_support_env",
    max_concurrent_envs=1,
)

# Customize Swagger UI with project metadata
app.title = "Customer Support OpenEnv"
app.version = "1.0.0"
app.description = """🤖 Customer Support Environment – An AI training environment for handling support tickets.

Agents learn to:

- Classify issues (billing, account, bug, feature)
- Choose solutions (pick the right action for each category)
- Make escalation decisions (when to escalate vs. close)
- Close tickets (finalize with proper rewards)

## How to Use

- Call POST /reset to start a new episode and load a random support ticket
- Call POST /step repeatedly with your agent's actions
- Episode ends when the observation returns done: true
- Call GET /state anytime to check current state

## Reward Structure

| Step | Max Reward |
|------|-----------|
| 1. Classify Issue | 0.2 |
| 2. Choose Solution | 0.3 |
| 3. Escalation | 0.3 |
| 4. Close Ticket | 0.2 |
| **Total** | **1.0** |

## Action Format for /step

```json
{
  "action": {
    "action_type": "classify_issue|choose_solution|escalate_decision|close_ticket",
    "classification": "billing|account|bug|feature",
    "category": "category_name",
    "solution": "solution_name",
    "should_escalate": true
  }
}
```

**OpenEnv Team** - [Website](https://openenv.dev)  
**License:** BSD-3-Clause"""

# ============================================================
# CRITICAL FIX: Override OpenEnv routes with singleton pattern
# ============================================================

# Remove default OpenEnv routes to replace with singleton versions
routes_to_remove = [r for r in app.routes if hasattr(r, 'path') and r.path in ['/reset', '/step', '/state']]
for route in routes_to_remove:
    app.routes.remove(route)

# Singleton environment instance
_env_instance = None

def get_environment() -> CustomerSupportEnvironment:
    """Get or create singleton environment."""
    global _env_instance
    if _env_instance is None:
        _env_instance = CustomerSupportEnvironment()
    return _env_instance

# Add custom routes that use singleton
@app.post("/reset")
async def reset_endpoint():
    """Reset environment and load a new ticket."""
    env = get_environment()
    obs = env.reset()
    return {
        "observation": obs.__dict__,
        "reward": 0.0,
        "done": False
    }

@app.post("/step")
async def step_endpoint(request_body: Dict[str, Any] = Body(..., embed=False)):
    """Execute action in environment."""
    env = get_environment()
    
    try:
        # Handle both wrapped {"action": {...}} and direct {...} formats
        action_data = request_body
        if "action" in request_body and isinstance(request_body.get("action"), dict):
            action_data = request_body["action"]
        
        # Check if it has action_type (valid action)
        if "action_type" not in action_data:
            return {
                "observation": {
                    "message": "Invalid request",
                    "error": "Missing 'action_type' field in request body"
                },
                "reward": -0.5,
                "done": True
            }
        
        # Parse action from the action data
        action = SupportAction(**action_data)
        
        # Store current total reward BEFORE step
        reward_before = env.total_reward
        
        # Execute step
        obs = env.step(action)
        
        # Calculate reward for THIS step only
        step_reward = env.total_reward - reward_before
        
        return {
            "observation": obs.__dict__,
            "reward": step_reward,
            "done": obs.done
        }
    except Exception as e:
        return {
            "observation": {
                "message": "Error executing action",
                "resolution_message": f"ERROR: {str(e)}",
                "error": True
            },
            "reward": -0.5,
            "done": True
        }

@app.get("/state")
async def state_endpoint():
    """Get current environment state."""
    env = get_environment()
    
    if env.current_ticket is None:
        return {
            "status": "no_ticket_loaded",
            "message": "Call /reset first to load a ticket",
            "current_ticket": None
        }
    else:
        # Build observation from current state
        obs = env._observation(
            status="active",
            reward=0.0,
            done=False
        )
        return obs.__dict__


# ============================================================
# CORS & Gradio Setup
# ============================================================

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Spaces/demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build custom Gradio UI
gradio_app = build_gradio_app(
    app,
    web_manager=None,
    action_fields=[],
    metadata=None,
    is_chat_env=False,
    title="Customer Support OpenEnv Playground",
    quick_start_md="",
)

# CRITICAL FIX: Mount Gradio at root AFTER all API routes are already registered
# This ensures /reset, /step, /state are handled by FastAPI BEFORE Gradio intercepts them
# The key is that FastAPI routes take precedence over the Gradio ASGI mount
gr.mount_gradio_app(app, gradio_app, path="/")

# Add a simple health check to verify the API is accessible
@app.get("/api/status", include_in_schema=False)
async def api_status():
    """Verify API is accessible separate from Gradio."""
    return {"status": "ok"}




def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


# Make main() callable at module level for openenv validator
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
