"""
Tic-Tac-Toe OpenEnv Environment Implementation.

Complete RL environment with 3 difficulty tasks, opponent AI, graders, and reward shaping.
"""

from uuid import uuid4
from typing import List, Tuple
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TicTacToeAction, TicTacToeObservation
except ImportError:
    from models import TicTacToeAction, TicTacToeObservation

class TicTacToeEnvironment(Environment):
    """Tic-Tac-Toe environment with 3 difficulty-based tasks."""
    
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    # Task Definitions
    TASKS = {
        1: {"name": "Easy", "opponent": "random", "max_steps": 9, "difficulty": "EASY"},
        2: {"name": "Medium", "opponent": "strategic", "max_steps": 9, "difficulty": "MEDIUM"},
        3: {"name": "Hard", "opponent": "optimal", "max_steps": 9, "difficulty": "HARD"},
    }
    
    def __init__(self):
        self.board = [[0]*3 for _ in range(3)]
        self.current_player = 1
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_task = 1
        self.total_reward = 0.0
        self.game_history = []
        
    def reset(self) -> TicTacToeObservation:
        """Reset the game to a new state."""
        self.board = [[0]*3 for _ in range(3)]
        self.current_player = 1
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.total_reward = 0.0
        self.game_history = []
        return self._get_observation("New game started. Agent is X (1), Opponent is O (2).")
    
    def set_task(self, task_id: int):
        """Set the difficulty task (1, 2, or 3)."""
        if task_id not in self.TASKS:
            raise ValueError(f"Task {task_id} not found. Available: {list(self.TASKS.keys())}")
        self.current_task = task_id
        return self.reset()
    
    def step(self, action: TicTacToeAction) -> TicTacToeObservation:
        try:
            row, col = action.row, action.col
            self._state.step_count += 1
            
            # Validate move
            if not (0 <= row < 3 and 0 <= col < 3):
                return self._get_observation("Invalid: out of bounds.", penalty=-0.1)
            if self.board[row][col] != 0:
                return self._get_observation("Invalid: cell occupied.", penalty=-0.1)
            
            # Agent makes move
            self.board[row][col] = 1
            winner = self._check_winner()
            
            if winner == 1:
                reward = 1.0  # Agent wins
                self.game_history.append(("agent_win", self._state.step_count))
                return self._get_observation("Agent wins!", done=True, reward=reward, winner=1)
            
            if self._is_board_full():
                reward = 0.5  # Draw
                self.game_history.append(("draw", self._state.step_count))
                return self._get_observation("Draw!", done=True, reward=reward, winner=3)
            
            # Opponent makes move
            opponent_move = self._get_opponent_move()
            if opponent_move:
                self.board[opponent_move[0]][opponent_move[1]] = 2
                winner = self._check_winner()
                
                if winner == 2:
                    reward = -1.0  # Opponent wins
                    self.game_history.append(("opponent_win", self._state.step_count))
                    return self._get_observation("Opponent wins!", done=True, reward=reward, winner=2)
                
                if self._is_board_full():
                    reward = 0.5  # Draw
                    self.game_history.append(("draw", self._state.step_count))
                    return self._get_observation("Draw!", done=True, reward=reward, winner=3)
            
            # Game continues
            reward = 0.1  # Small reward for each valid move (partial progress signal)
            self.total_reward += reward
            return self._get_observation("Move accepted.", reward=reward)
            
        except Exception as e:
            return self._get_observation(f"Error: {str(e)}", penalty=-0.5)
    
    def _get_opponent_move(self) -> Tuple[int, int]:
        """Get opponent move based on task difficulty."""
        task = self.TASKS[self.current_task]
        opponent_type = task["opponent"]
        
        if opponent_type == "random":
            return self._get_random_move()
        elif opponent_type == "strategic":
            return self._get_strategic_move()
        else:  # optimal
            return self._get_optimal_move()
    
    def _get_random_move(self) -> Tuple[int, int]:
        """Random legal move."""
        available = [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]
        return random.choice(available) if available else None
    
    def _get_strategic_move(self) -> Tuple[int, int]:
        """Strategic move: block agent win or take center/corners."""
        # Check if agent can win next move
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    self.board[i][j] = 1
                    if self._check_winner() == 1:
                        self.board[i][j] = 0
                        return (i, j)  # Block
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
        
        # Otherwise, random move
        return self._get_random_move()
    
    def _get_optimal_move(self) -> Tuple[int, int]:
        """Optimal move using minimax algorithm (simplified)."""
        # Try to win
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    self.board[i][j] = 2
                    if self._check_winner() == 2:
                        self.board[i][j] = 0
                        return (i, j)
                    self.board[i][j] = 0
        
        # Block agent from winning
        return self._get_strategic_move()
    
    def _check_winner(self) -> int:
        """Returns: 1=agent, 2=opponent, 0=none."""
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                return self.board[0][i]
        
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]
        
        return 0
    
    def _is_board_full(self) -> bool:
        return all(cell != 0 for row in self.board for cell in row)
    
    def _get_observation(self, message: str, done: bool = False, reward: float = 0.0, 
                        winner: int = 0, penalty: float = 0.0) -> TicTacToeObservation:
        actual_reward = reward + penalty
        self.total_reward += actual_reward
        return TicTacToeObservation(
            board=self.board,
            done=done,
            reward=actual_reward,
            message=message,
            winner=winner,
            task_id=self.current_task,
            opponent_strength=self.TASKS[self.current_task]["opponent"]
        )
    
    @property
    def state(self) -> State:
        return self._state
    
    # GRADER METHODS (for scoring)
    def grade_episode(self) -> dict:
        """Grade the completed episode (0.0-1.0)."""
        if not self.game_history:
            return {"score": 0.0, "reason": "No game played"}
        
        outcome, steps = self.game_history[-1]
        
        if outcome == "agent_win":
            return {"score": 1.0, "reason": "Agent won"}
        elif outcome == "draw":
            return {"score": 0.5, "reason": "Draw"}
        else:
            return {"score": 0.0, "reason": "Agent lost"}