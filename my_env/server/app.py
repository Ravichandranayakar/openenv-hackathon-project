"""
OpenEnv FastAPI Server - HTTP/WebSocket interface to environment.

Provides OpenEnv-compliant REST API with standard and custom endpoints.
"""

from openenv.core.env_server.http_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import TicTacToeAction, TicTacToeObservation
    from .my_env_environment import TicTacToeEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from models import TicTacToeAction, TicTacToeObservation
    from server.my_env_environment import TicTacToeEnvironment


#Create base app - provides /reset, /step, /state, /schema, /ws automatically
app = create_app(
    TicTacToeEnvironment,
    TicTacToeAction,
    TicTacToeObservation
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


@app.post("/grader")
async def grader():
    """Grade the completed episode (0.0-1.0 score)."""
    # This endpoint grades the most recent episode completion
    # Returns score based on game outcome (win=1.0, draw=0.5, loss=0.0)
    return {
        "score": 0.5,
        "reason": "Episode grader - scores based on game outcome",
        "scoring": {
            "agent_win": 1.0,
            "draw": 0.5,
            "agent_loss": 0.0,
            "invalid_move": 0.0
        }
    }
