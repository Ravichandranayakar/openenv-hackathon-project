"""
Customer Support OpenEnv Server - HTTP/WebSocket interface.

Provides OpenEnv-compliant REST API for customer support environment.
Standard endpoints (/reset, /step, /state, /schema, /ws) provided by create_app().
"""

from openenv.core.env_server.http_server import create_app
from models import SupportAction, SupportObservation
from customer_support_environment import CustomerSupportEnvironment


# Create app - provides /reset, /step, /state, /schema, /ws automatically
app = create_app(
    env_class=CustomerSupportEnvironment,
    action_type=SupportAction,
    observation_type=SupportObservation
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/tasks")
async def list_tasks():
    """List available tasks."""
    return {
        "tasks": [
            {"id": 1, "name": "Easy", "description": "Simple ticket classification"},
            {"id": 2, "name": "Medium", "description": "Mixed ticket types with escalation"},
            {"id": 3, "name": "Hard", "description": "Complex cases requiring expertise"}
        ]
    }



