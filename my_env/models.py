# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the My Env Environment.

The my_env environment is a tic-tac-toe  for simple test 
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List

class TicTacToeAction(Action):
    """Action for the My Env environment - just a message to echo."""

    row: int = Field(..., description="Row index (0-2)")
    col: int = Field(... , description="coloumn index (0-2)")


class TicTacToeObservation(Observation):
    """Observation from the My Env environment - the echoed message."""

    board: List[List[int]] = Field(... , description="3x3 board: 0=empty , 1=agent ,2=opponent")
    done: bool = Field(... , description="Is the game over")
    reward: float = Field(... , description="feedback or status message")
    winner: int  = Field(0 , description="0=non , 1=agent , 2=opponent , 3=draw" )
    
