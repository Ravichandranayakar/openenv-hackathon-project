"""
OpenEnv Environment Models - Define Action and Observation types.

These Pydantic models define the interface between agents and environment.
All fields must be JSON-serializable for HTTP communication.
"""

from typing import List
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


class TicTacToeAction(Action):
    """Agent action: place mark at (row, col) on 3x3 board."""
    row: int = Field(..., ge=0, le=2, description="Row: 0-2")
    col: int = Field(..., ge=0, le=2, description="Column: 0-2")


class TicTacToeObservation(Observation):
    """Environment observation: game state and metrics."""
    board: List[List[int]] = Field(..., description="3x3 board state")
    done: bool = Field(..., description="Episode terminated")
    reward: float = Field(..., description="Reward for last action")
    message: str = Field(..., description="Status message")
    winner: int = Field(0, description="0=none, 1=agent, 2=opponent, 3=draw")
    task_id: int = Field(0, description="Task level: 1-3")
    opponent_strength: str = Field("random", description="Opponent: random/strategic/optimal") 