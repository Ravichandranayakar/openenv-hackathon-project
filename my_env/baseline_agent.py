#!/usr/bin/env python3
"""
Baseline Agent for Tic-Tac-Toe OpenEnv Environment.

This script runs a simple random-action agent against all 3 task levels
and produces reproducible baseline scores for the hackathon.

Usage:
    python baseline_agent.py --url http://localhost:8000
    python baseline_agent.py --episodes 50
"""

import os
import random
import argparse
from typing import Dict, List

try:
    from client import TicTacToeEnv
    from models import TicTacToeAction
except ImportError:
    from my_env.client import TicTacToeEnv
    from my_env.models import TicTacToeAction


class RandomAgent:
    """Baseline agent that plays random legal moves."""
    
    def get_action(self, board: List[List[int]]) -> TicTacToeAction:
        """Select a random legal move."""
        available_moves = [
            (i, j) for i in range(3) for j in range(3) if board[i][j] == 0
        ]
        if not available_moves:
            return TicTacToeAction(row=0, col=0)
        row, col = random.choice(available_moves)
        return TicTacToeAction(row=row, col=col)


def run_episode(env: TicTacToeEnv, task_id: int, agent: RandomAgent) -> Dict:
    """Run one episode and return score and stats."""
    try:
        # Reset environment
        result = env.reset()
        total_reward = result.reward
        steps = 1
        
        # Play until done
        while not result.done and steps < 20:
            action = agent.get_action(result.observation.board)
            result = env.step(action)
            total_reward += result.reward
            steps += 1
        
        # Score based on outcome
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
            "outcome": outcome,
            "task_id": task_id
        }
    except Exception as e:
        print(f"Error in episode: {e}")
        return {"score": 0.0, "reward": 0.0, "steps": 0, "outcome": "error", "task_id": task_id}


def run_baseline(
    env_url: str = "http://localhost:8000",
    num_episodes: int = 10,
    seed: int = 42
) -> Dict:
    """Run baseline agent against all 3 tasks."""
    
    random.seed(seed)
    results = {}
    agent = RandomAgent()
    
    print(f"\n🎮 Running Tic-Tac-Toe Baseline Agent")
    print(f"   Environment: {env_url}")
    print(f"   Episodes per task: {num_episodes}")
    print(f"   Seed: {seed}")
    print("=" * 60)
    
    for task_id in [1, 2, 3]:
        print(f"\n📊 Task {task_id} (Difficulty: ['EASY', 'MEDIUM', 'HARD'][{task_id-1}])")
        task_episodes = []
        
        with TicTacToeEnv(base_url=env_url).sync() as env:
            for episode in range(num_episodes):
                episode_result = run_episode(env, task_id, agent)
                task_episodes.append(episode_result)
                
                status = "✓" if episode_result["outcome"] == "win" else "✗"
                print(f"   Episode {episode+1:2d}: {status} score={episode_result['score']:.2f} "
                      f"outcome={episode_result['outcome']:6s} steps={episode_result['steps']}")
        
        avg_score = sum(e["score"] for e in task_episodes) / len(task_episodes)
        avg_reward = sum(e["reward"] for e in task_episodes) / len(task_episodes)
        win_rate = sum(1 for e in task_episodes if e["outcome"] == "win") / len(task_episodes)
        
        results[f"task_{task_id}"] = {
            "avg_score": avg_score,
            "avg_reward": avg_reward,
            "win_rate": win_rate,
            "episodes": task_episodes
        }
        
        print(f"   └─ Average Score: {avg_score:.3f} | Win Rate: {win_rate*100:.1f}% | Avg Reward: {avg_reward:.3f}")
    
    return results


def print_summary(results: Dict):
    """Print a summary of baseline results."""
    print("\n" + "=" * 60)
    print("📈 BASELINE RESULTS SUMMARY")
    print("=" * 60)
    
    for task_name, task_data in results.items():
        task_num = task_name.split("_")[1]
        print(f"\n{task_name.upper()}:")
        print(f"  Average Score:  {task_data['avg_score']:.4f}")
        print(f"  Average Reward: {task_data['avg_reward']:.4f}")
        print(f"  Win Rate:       {task_data['win_rate']*100:.1f}%")
    
    # Overall summary
    overall_score = sum(
        task_data["avg_score"] 
        for task_data in results.values()
    ) / len(results)
    
    print(f"\n{'Overall Average Score:':<25} {overall_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline agent for Tic-Tac-Toe OpenEnv")
    parser.add_argument("--url", default="http://localhost:8000", help="environment URL")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per task")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    args = parser.parse_args()
    
    try:
        results = run_baseline(
            env_url=args.url,
            num_episodes=args.episodes,
            seed=args.seed
        )
        print_summary(results)
    except Exception as e:
        print(f"\n❌ Error running baseline: {e}")
        import traceback
        traceback.print_exc()
