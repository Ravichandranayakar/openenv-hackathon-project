# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Tic-Tac-Toe OpenEnv Environment.

Typed Pydantic models for actions, observations, and task definitions.
"""
from pydantic import BaseModel, Field
from typing import List
from openenv.core.env_server.types import Action, Observation


class TicTacToeAction(Action):
    """Action: Agent places mark at (row, col)."""
    row: int = Field(..., ge=0, le=2, description="Row index (0-2)")
    col: int = Field(..., ge=0, le=2, description="Column index (0-2)")


class TicTacToeObservation(Observation):
    """Observation: Current board state and game info."""
    board: List[List[int]] = Field(..., description="3x3 board: 0=empty, 1=agent, 2=opponent")
    done: bool = Field(..., description="Is game over?")
    reward: float = Field(..., description="Reward for this step")
    message: str = Field(..., description="Status message")
    winner: int = Field(0, description="0=none, 1=agent_wins, 2=opponent_wins, 3=draw")
    task_id: int = Field(0, description="Current task level (1-3)")
    opponent_strength: str = Field("random", description="Opponent AI level: random/strategic/optimal")