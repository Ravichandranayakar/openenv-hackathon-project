#!/usr/bin/env python3
"""
 TRAIN 4 SPECIALIZED LLM AGENTS - TRL GRPO APPROACH
──────────────────────────────────────────────────────
Theme #1: Multi-Agent Interactions (Hackathon)

This script trains 4 specialized LLM agents:
- Router Agent: Classify support tickets
- Resolver Agent: Propose solutions
- Manager Agent: Make escalation decisions
- Quality Agent: Assess satisfaction

Usage:
  python scripts/train_multi_agent.py

Training uses:
- Model: Llama-3.2-1B-Instruct
- Optimizer: TRL GRPO
- Quantization: Unsloth 4-bit
- Training examples per agent: 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from my_env.pytorch.training.trl_multi_agent_trainer import MultiAgentGRPOTrainer


def main():
  """Train all 4 agents with TRL GRPO."""
  
  print("\n" + "="*70)
  print(" MULTI-AGENT TRL GRPO TRAINING PIPELINE".center(70))
  print("="*70)
  print("\nTraining 4 Specialized LLM Agents:")
  print(" 1. Router Agent (classify tickets)")
  print(" 2. Resolver Agent (propose solutions)")
  print(" 3. Manager Agent (escalation decisions)")
  print(" 4. Quality Agent (satisfaction assessment)")
  print("\nExpected time: 20-45 minutes on GPU")
  print("\n" + "="*70 + "\n")
  
  # Create trainer
  trainer = MultiAgentGRPOTrainer(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    num_agents=4,
    learning_rate=1e-4,
    batch_size=8,
    num_train_epochs=3,
  )
  
  # Train all 4 agents
  trainer.train_all_agents(output_dir="./checkpoints")
  
  print("\n" + "="*70)
  print("✅ TRAINING COMPLETE - All 4 agents trained!")
  print("="*70)
  print("\nNext steps:")
  print(" 1. Start environment server:")
  print("   python -m uvicorn my_env.server.app:app --port 8000")
  print(" 2. Run inference demo:")
  print("   python scripts/inference_demo.py")
  print("="*70 + "\n")


if __name__ == "__main__":
  main()
