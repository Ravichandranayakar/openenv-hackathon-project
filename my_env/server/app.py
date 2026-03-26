"""
Customer Support OpenEnv Server - HTTP/WebSocket interface.

Provides OpenEnv-compliant REST API for customer support environment.
Standard endpoints (/reset, /step, /state, /schema, /ws) provided by create_app().
"""

from openenv.core.env_server.http_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports
    from ..models import SupportAction, SupportObservation
    from .customer_support_environment import CustomerSupportEnvironment
except ImportError:
    # Standalone imports (Docker deployment)
    from models import SupportAction, SupportObservation
    from server.customer_support_environment import CustomerSupportEnvironment


# Create app - provides /reset, /step, /state, /schema, /ws automatically
# Use POSITIONAL arguments, not named
app = create_app(
    CustomerSupportEnvironment,
    SupportAction,
    SupportObservation
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



