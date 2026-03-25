"""
OpenEnv FastAPI Server - HTTP/WebSocket interface to environment.

Provides OpenEnv-compliant REST API with standard and custom endpoints.
"""

from openenv.core.env_server.http_server import create_app
from models import TicTacToeAction, TicTacToeObservation
from my_env_environment import TicTacToeEnvironment


#Create base app - provides /reset, /step, /state, /schema, /ws automatically
app = create_app(
    env_class=TicTacToeEnvironment,
    action_type=TicTacToeAction,
    observation_type=TicTacToeObservation
)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/tasks")
async def list_tasks():
    """List available tasks."""
    return {
        "tasks": [
            {"id": 1, "name": "Easy", "opponent": "random"},
            {"id": 2, "name": "Medium", "opponent": "strategic"},
            {"id": 3, "name": "Hard", "opponent": "optimal"}
        ]
    }
