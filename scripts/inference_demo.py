#!/usr/bin/env python3
"""Demo inference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from my_env.pytorch.agents.multi_agent_system import MultiAgentSystem
from my_env.pytorch.inference.inference_engine import InferenceEngine


def main():
  """Run demo."""
  system = MultiAgentSystem(device="cpu")
  engine = InferenceEngine(system)

  tickets = [
    {"text": "I forgot my password", "category": "account"},
    {"text": "I was charged twice", "category": "billing"},
    {"text": "API returns 500", "category": "technical"},
  ]

  print(f"\n{'='*70}\nDemo: Multi-Agent Customer Support\n{'='*70}\n")

  for ticket in tickets:
    result = engine.infer(ticket["text"], return_timing=True)
    print(f"Ticket: {ticket['text']}")
    print(f"Expected: {ticket['category']} | Routed: {result['final_routing']}")
    print(f"Confidence: {result['confidence']:.2%} | Time: {result['inference_time_ms']:.1f}ms\n")


if __name__ == "__main__":
  main()
