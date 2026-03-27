#!/usr/bin/env python3
"""
Customer Support OpenEnv - Demo Script

Quick start for reviewers:
    python demo.py --url http://localhost:8000 --episodes 5 --task 1

This script demonstrates how agents interact with the environment by:
1. Connecting to the OpenEnv server
2. Running multiple episodes (tasks)
3. Showing learning progress through ground truth feedback

Under the hood, this runs my_env/baseline_agent.py with command-line arguments.
"""

import sys
import os
import argparse

# Ensure my_env is importable
sys.path.insert(0, os.path.dirname(__file__))

from my_env.baseline_agent import run_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Customer Support OpenEnv - Baseline Agent Demo",
        epilog="Example: python demo.py --url http://localhost:8000 --episodes 5 --task 1"
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Environment server URL")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--task", type=int, default=1, help="Task ID: 1=Easy, 2=Medium, 3=Hard")
    
    args = parser.parse_args()
    
    try:
        run_baseline(args.url, args.episodes, args.seed, args.task)
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
