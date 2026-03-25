"""
OpenEnv Environment - Core game logic and RL interface.

Implements Environment interface with reset/step/state for Tic-Tac-Toe.
"""

from uuid import uuid4
from typing import List, Tuple
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import TicTacToeAction, TicTacToeObservation


class TicTacToeEnvironment(Environment):
    """Tic-Tac-Toe environment with 3 difficulty levels."""
    
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    TASKS = {
        1: {"name": "Easy", "opponent": "random"},
        2: {"name": "Medium", "opponent": "strategic"},
        3: {"name": "Hard", "opponent": "optimal"},
    }
    
    def __init__(self):
        self.board = [[0] * 3 for _ in range(3)]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_task = 1
        self.game_over = False
        self.winner = 0
    
    def reset(self) -> TicTacToeObservation:
        """Reset environment to initial state."""
        self.board = [[0] * 3 for _ in range(3)]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.game_over = False
        self.winner = 0
        return self._observation("Game reset", reward=0.0)
    
    def set_task(self, task_id: int) -> TicTacToeObservation:
        """Switch to different difficulty task."""
        if task_id not in self.TASKS:
            raise ValueError(f"Invalid task: {task_id}")
        self.current_task = task_id
        return self.reset()
    
    def step(self, action: TicTacToeAction) -> TicTacToeObservation:
        """Execute agent action and return observation."""
        try:
            self._state.step_count += 1
            
            # Validate move
            if not (0 <= action.row < 3 and 0 <= action.col < 3):
                return self._observation("Invalid: out of bounds", reward=-0.1, done=True)
            if self.board[action.row][action.col] != 0:
                return self._observation("Invalid: cell occupied", reward=-0.1, done=True)
            
            # Agent move
            self.board[action.row][action.col] = 1
            if self._check_winner() == 1:
                return self._observation("Agent wins", reward=1.0, done=True, winner=1)
            if self._is_full():
                return self._observation("Draw", reward=0.5, done=True, winner=3)
            
            # Opponent move
            opp_move = self._get_opponent_move()
            if opp_move:
                self.board[opp_move[0]][opp_move[1]] = 2
                if self._check_winner() == 2:
                    return self._observation("Opponent wins", reward=-1.0, done=True, winner=2)
                if self._is_full():
                    return self._observation("Draw", reward=0.5, done=True, winner=3)
            
            # Game continues
            return self._observation("Move accepted", reward=0.1)
        
        except Exception as e:
            return self._observation(f"Error: {e}", reward=-0.5, done=True)
    
    def _get_opponent_move(self) -> Tuple[int, int]:
        """Get opponent move based on task difficulty."""
        task = self.TASKS[self.current_task]
        if task["opponent"] == "random":
            return self._random_move()
        elif task["opponent"] == "strategic":
            return self._strategic_move()
        else:
            return self._optimal_move()
    
    def _random_move(self) -> Tuple[int, int]:
        """Random legal move."""
        available = [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]
        return random.choice(available) if available else None
    
    def _strategic_move(self) -> Tuple[int, int]:
        """Block agent win or try to win."""
        # Block agent
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    self.board[i][j] = 1
                    if self._check_winner() == 1:
                        self.board[i][j] = 0
                        return (i, j)
                    self.board[i][j] = 0
        
        # Try to win
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    self.board[i][j] = 2
                    if self._check_winner() == 2:
                        self.board[i][j] = 0
                        return (i, j)
                    self.board[i][j] = 0
        
        return self._random_move()
    
    def _optimal_move(self) -> Tuple[int, int]:
        """Try to win first, then block."""
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    self.board[i][j] = 2
                    if self._check_winner() == 2:
                        self.board[i][j] = 0
                        return (i, j)
                    self.board[i][j] = 0
        return self._strategic_move()
    
    def _check_winner(self) -> int:
        """Check win condition: 0=none, 1=agent, 2=opponent."""
        # Rows
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return self.board[i][0]
        
        # Columns
        for j in range(3):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != 0:
                return self.board[0][j]
        
        # Diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]
        
        return 0
    
    def _is_full(self) -> bool:
        """Check if board is full."""
        return all(cell != 0 for row in self.board for cell in row)
    
    def _observation(self, message: str, reward: float = 0.0, done: bool = False, 
                     winner: int = 0) -> TicTacToeObservation:
        """Create observation."""
        return TicTacToeObservation(
            board=self.board,
            done=done,
            reward=reward,
            message=message,
            winner=winner,
            task_id=self.current_task,
            opponent_strength=self.TASKS[self.current_task]["opponent"]
        )
    
    @property
    def state(self) -> State:
        """Get current state."""
        return self._state
    
    def grade_episode(self) -> dict:
        """Score the episode: 0.0-1.0."""
        if self.winner == 1:
            return {"score": 1.0}
        elif self.winner == 3:
            return {"score": 0.5}
        else:
            return {"score": 0.0}