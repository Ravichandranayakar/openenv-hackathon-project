"""
Customer Support OpenEnv Server - HTTP/WebSocket interface with Web Dashboard.

Provides OpenEnv-compliant REST API + interactive web interface for customer support environment.
Endpoints (/reset, /step, /state, /schema, /ws) + web dashboard at /web provided by create_web_interface_app().
"""

from openenv.core.env_server.http_server import create_web_interface_app
from models import SupportAction, SupportObservation
from customer_support_environment import CustomerSupportEnvironment


# Create app with built-in web interface dashboard
# Provides /reset, /step, /state, /schema, /ws + /web dashboard automatically
app = create_web_interface_app(
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



