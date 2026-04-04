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
app.description = """
🤖 **Customer Support Environment** – An AI training environment for handling support tickets.

Agents learn to:
- **Classify issues** (billing, account, bug, feature)
- **Choose solutions** (pick the right action for each category)
- **Make escalation decisions** (when to escalate vs. close)
- **Close tickets** (finalize with proper rewards)

## How to Use
1. Call `POST /reset` to start a new episode and load a random support ticket
2. Call `POST /step` repeatedly with your agent's actions
3. Episode ends when the observation returns `done: true`
4. Call `GET /state` anytime to check current state

## Reward Structure

| Step                | Max Reward |
|---------------------|------------|
| 1. Classify Issue   | 0.2        |
| 2. Choose Solution  | 0.3        |
| 3. Escalation       | 0.3        |
| 4. Close Ticket     | 0.2        |
| **Total**           | **1.0**    |

## Action Format for `/step`
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
"""
app.version = "1.0.0"
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
