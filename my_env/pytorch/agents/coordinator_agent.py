"""Coordinator agent for intelligent routing and escalation decisions."""

from typing import Any, Dict
import torch
import torch.nn as nn

from my_env.pytorch.agents.base_agent import BaseAgent
from my_env.pytorch.models.dqn_network import CoordinatorDQN


class CoordinatorAgent(BaseAgent):
  """Coordinator agent: Routes tickets to specialists or escalates.
  
  - Receives classification and confidence from responder
  - Routes to [billing_specialist, account_specialist, technical_specialist, 
        escalate_to_human]
  - Learns routing policy via DQN
  - Maintains history of successful routings
  """

  def __init__(self, agent_id: str = "coordinator", device: str = "cpu"):
    """Initialize coordinator agent.
    
    Args:
      agent_id: Unique identifier
      device: 'cpu' or 'cuda'
    """
    super().__init__(agent_id, agent_type="coordinator")
    self.device = device

    # DQN for routing decisions
    # Input: (768 embedding + 4 class logits + 1 confidence) = 773 dims
    self.dqn = CoordinatorDQN(input_dim=773, action_dim=5, hidden_dims=[256, 128, 64])

    # Routing history: class -> specialist success rate
    self.routing_history = {i: {} for i in range(4)}

    self.to(device)

  def forward(self, observation: Dict[str, Any]) -> Dict[str, Any]:
    """Decide routing based on responder's output.
    
    Args:
      observation: Dict with:
        - classification: int
        - solution: int
        - confidence: float
        - embedding: tensor (768,)
        
    Returns:
      Dict with:
        - action: int (0-4, routing decision)
        - q_values: tensor (5,)
        - reasoning: str
    """
    # Extract inputs
    embedding = observation.get("embedding", torch.zeros(768))
    classification = observation.get("classification", 0)
    confidence = observation.get("confidence", 0.5)

    # Build DQN input
    # Convert classification to one-hot
    class_one_hot = torch.zeros(4)
    class_one_hot[classification] = 1.0
    confidence_tensor = torch.tensor([confidence])

    dqn_input = torch.cat([embedding, class_one_hot, confidence_tensor])

    # Get Q-values
    q_values = self.dqn(dqn_input.unsqueeze(0).to(self.device)) # (1, 5)

    # Choose action (greedy if not training, epsilon-greedy if training)
    if self.is_training and torch.rand(1).item() < 0.1: # 10% exploration
      action = torch.randint(0, 5, (1,)).item()
    else:
      action = torch.argmax(q_values, dim=1).item()

    # Reasoning
    action_names = [
      "route_to_billing",
      "route_to_account",
      "route_to_technical",
      "route_to_specialist",
      "escalate_to_human",
    ]

    return {
      "action": action,
      "action_name": action_names[action],
      "q_values": q_values.detach().cpu(),
      "reasoning": f"Routed based on DQN policy (Q={q_values.max().item():.3f})",
    }

  def to_device(self, device: str) -> None:
    """Move to device."""
    self.device = device
    self.dqn.to(device)

  @property
  def trainable_parameters(self) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in self.dqn.parameters())
