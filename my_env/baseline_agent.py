#!/usr/bin/env python3
"""
Baseline Agent - Example inference script showing how to use environment.

Runs a simple agent and measures performance across all tasks.
"""

import random
import argparse
from typing import List

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from .client import TicTacToeEnv
    from .models import TicTacToeAction
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from client import TicTacToeEnv
    from models import TicTacToeAction


class RandomAgent:
    """Agent that plays random legal moves."""
    
    def get_action(self, board: List[List[int]]) -> TicTacToeAction:
        """Select random legal move."""
        available = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
        if not available:
            return TicTacToeAction(row=0, col=0)
        row, col = random.choice(available)
        return TicTacToeAction(row=row, col=col)


def run_episode(env: TicTacToeEnv, agent: RandomAgent) -> dict:
    """Play one complete episode."""
    try:
        result = env.reset()
        steps = 0
        total_reward = 0.0
        
        while not result.done and steps < 20:
            action = agent.get_action(result.observation.board)
            result = env.step(action)
            total_reward += result.reward
            steps += 1
        
        # Determine outcome
        if result.observation.winner == 1:
            score = 1.0
            outcome = "win"
        elif result.observation.winner == 3:
            score = 0.5
            outcome = "draw"
        else:
            score = 0.0
            outcome = "loss"
        
        return {
            "score": score,
            "reward": total_reward,
            "steps": steps,
            "outcome": outcome
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {"score": 0.0, "reward": 0.0, "steps": 0, "outcome": "error"}


def run_baseline(env_url: str, num_episodes: int, seed: int):
    """Run baseline on all 3 tasks."""
    random.seed(seed)
    agent = RandomAgent()
    
    print(f"\nBaseline Agent Run")
    print(f"URL: {env_url}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    for task_id in [1, 2, 3]:
        task_names = {1: "EASY", 2: "MEDIUM", 3: "HARD"}
        print(f"\nTask {task_id} ({task_names[task_id]})")
        
        episodes_data = []
        with TicTacToeEnv(base_url=env_url).sync() as env:
            for i in range(num_episodes):
                ep = run_episode(env, agent)
                episodes_data.append(ep)
                status = "[+]" if ep["outcome"] == "win" else "[-]"
                print(f"  Ep {i+1}: {status} score={ep['score']:.2f} outcome={ep['outcome']}")
        
        avg_score = sum(e["score"] for e in episodes_data) / len(episodes_data)
        win_rate = sum(e["outcome"] == "win" for e in episodes_data) / len(episodes_data)
        
        print(f"  Summary: avg_score={avg_score:.3f}, win_rate={win_rate*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    try:
        run_baseline(args.url, args.episodes, args.seed)
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
