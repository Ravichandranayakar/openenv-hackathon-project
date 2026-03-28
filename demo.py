#!/usr/bin/env python3
"""
[DEMO] LIVE DEMO - Watch the Agent Learn!

Quick demo for judges:
    python demo.py          # Quick 5-episode demo
    python demo.py --episodes 20  # Longer demo (see more learning)

What this shows:
1. Creates a fresh agent (no training)
2. Runs it on real customer support tickets
3. Shows live accuracy metrics
4. Demonstrates real learning in action!
"""

import sys
import os
import argparse

# Ensure my_env is importable
sys.path.insert(0, os.path.dirname(__file__))

from my_env import CustomerSupportEnvironment, SupportAction
from my_env.agents import CurriculumLearningAgent

def run_demo(episodes=5):
    """Run a quick demo of the agent learning"""
    print("\n" + "="*80)
    print("[DEMO] LIVE DEMO - Agent Learning in Real-Time")
    print("="*80)
    print(f"\nRunning {episodes} episodes on real customer support tickets...\n")
    
    env = CustomerSupportEnvironment()
    agent = CurriculumLearningAgent()
    
    total_reward = 0
    for ep in range(1, episodes + 1):
        env.set_task(2)  # Medium difficulty - good for demo
        observation = env.reset()
        reward = agent.step(env, observation)
        total_reward += reward
        
        accuracy = len([r for r in agent.action_accuracy['classification']['rewards'] if r > 0]) / max(1, len(agent.action_accuracy['classification']['rewards']))
        
        print(f"Episode {ep:2d}: Reward={reward:+.2f} | Classification Accuracy: {accuracy*100:.1f}% | Escalation Keywords Learned: {len(agent.escalation_keywords)}")
    
    avg_reward = total_reward / episodes
    print(f"\n{'='*80}")
    print(f"Average Reward: {avg_reward:+.3f}")
    print(f"Escalation Keywords Learned: {sorted(list(agent.escalation_keywords))}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Customer Support OpenEnv - Live Training Demo",
        epilog="Example: python demo.py --episodes 10"
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    
    args = parser.parse_args()
    
    try:
        run_demo(args.episodes)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        import traceback
        traceback.print_exc()
        sys.exit(1)
