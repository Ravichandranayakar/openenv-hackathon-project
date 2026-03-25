# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TicTacToe OpenEnv Environment Client."""

from openenv.core import EnvClient
from .models import TicTacToeAction, TicTacToeObservation


class TicTacToeEnv(EnvClient[TicTacToeAction, TicTacToeObservation, dict]):
    """
    Client for the Tic-Tac-Toe Environment.

    Maintains a persistent WebSocket connection to the environment server
    for efficient multi-step interactions.

    Example:
        >>> with TicTacToeEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()
        ...     action = TicTacToeAction(row=0, col=0)
        ...     result = env.step(action)
        ...     print(f"Board: {result.observation.board}")
        ...     print(f"Reward: {result.reward}")
    """
    pass
        """
        Convert MyAction to JSON payload for step message.

        Args:
            action: MyAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MyObservation]:
        """
        Parse server response into StepResult[MyObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MyObservation
        """
        obs_data = payload.get("observation", {})
        observation = MyObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
