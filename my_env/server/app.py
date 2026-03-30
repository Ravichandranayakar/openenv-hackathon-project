"""
Customer Support OpenEnv Server - HTTP/WebSocket interface.

Provides OpenEnv-compliant REST API for customer support environment.
Standard endpoints (/reset, /step, /state, /schema, /ws) provided by create_app().
"""

from openenv.core.env_server.http_server import create_app
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

# Support both in-repo and standalone imports
try:
    # In-repo imports (my_env package structure)
    from ..models import SupportAction, SupportObservation
    from .customer_support_environment import CustomerSupportEnvironment
except ImportError:
    # Standalone/Docker imports (models.py at root)
    import sys
    from pathlib import Path
    root = Path(__file__).parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models import SupportAction, SupportObservation
    from my_env.server.customer_support_environment import CustomerSupportEnvironment


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


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """
    Convert raw Pydantic validation errors into readable, user-friendly format.
    Helps both humans and agents understand what went wrong.
    """
    errors = []
    for error in exc.errors():
        field = error.get("loc", [])[-1] if error.get("loc") else "unknown"
        error_type = error.get("type", "unknown")
        
        # Create readable error messages
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



