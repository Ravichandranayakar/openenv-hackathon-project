#!/usr/bin/env python3
"""Evaluate multi-agent system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from my_env.pytorch.agents.multi_agent_system import MultiAgentSystem
from my_env.pytorch.inference.inference_engine import InferenceEngine


def main():
  """Run evaluation."""
  system = MultiAgentSystem(device="cpu")
  engine = InferenceEngine(system)

  test_tickets = [
    "I forgot my password",
    "I was charged twice",
    "API returns errors",
  ]

  print(f"\n{'='*60}\nEvaluation\n{'='*60}\n")

  for ticket in test_tickets:
    result = engine.infer(ticket, return_timing=True)
    print(f"Ticket: {ticket}")
    print(f"Routing: {result['final_routing']} | Time: {result['inference_time_ms']:.1f}ms\n")


if __name__ == "__main__":
  main()
