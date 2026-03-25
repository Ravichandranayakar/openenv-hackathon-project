# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Tic-Tac-Toe OpenEnv Environment.

This module creates an HTTP server that exposes the TicTacToeEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Standard OpenEnv Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Custom Hackathon Endpoints:
    - POST /tasks: List available tasks
    - POST /set_task/{task_id}: Set difficulty level
    - POST /grader: Get episode score

Usage:
    # Development (with auto-reload):
    uv run server --reload

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import TicTacToeAction, TicTacToeObservation
    from .my_env_environment import TicTacToeEnvironment
except ImportError:
    from models import TicTacToeAction, TicTacToeObservation
    from my_env_environment import TicTacToeEnvironment

# Create the base app with all standard OpenEnv endpoints
app = create_app(
    env_class=TicTacToeEnvironment,
    action_type=TicTacToeAction,
    observation_type=TicTacToeObservation,
)

# Add custom endpoints for hackathon requirements
@app.post("/tasks")
async def get_tasks():
    """
    List all available tasks for this environment.
    Returns task definitions with difficulty levels.
    """
    return {
        "tasks": [
            {
                "id": 1,
                "name": "Easy",
                "description": "Play against random opponent",
                "difficulty": "EASY"
            },
            {
                "id": 2,
                "name": "Medium",
                "description": "Play against strategic opponent (blocks wins)",
                "difficulty": "MEDIUM"
            },
            {
                "id": 3,
                "name": "Hard",
                "description": "Play against optimal opponent (minimax)",
                "difficulty": "HARD"
            },
        ]
    }

@app.post("/grader")
async def grader():
    """
    Get grader score for the last completed episode.
    Returns score (0.0-1.0), reason, and game outcome.
    """
    # This endpoint would need session context in production
    # For now, it returns a template response
    return {
        "score": 0.0,
        "reason": "No episode completed",
        "task_id": 1
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "TicTacToe"}