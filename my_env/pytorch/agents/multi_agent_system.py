"""Multi-agent system orchestrator."""

from typing import Any, Dict, List
import torch

from my_env.pytorch.agents.base_agent import BaseAgent
from my_env.pytorch.agents.responder_agent import ResponderAgent
from my_env.pytorch.agents.coordinator_agent import CoordinatorAgent
from my_env.pytorch.agents.specialist_agent import SpecialistAgent


class MultiAgentSystem(BaseAgent):
    """Orchestrates interactions between responder, coordinator, and specialist agents.
    
    Manages message passing, state synchronization, and episode execution.
    """

    def __init__(self, device: str = "cpu"):
        """Initialize multi-agent system.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        super().__init__("system", agent_type="orchestrator")
        self.device = device

        # Create agents
        self.responder = ResponderAgent(device=device)
        self.coordinator = CoordinatorAgent(device=device)

        # Create specialist agents
        self.specialists = {
            "billing": SpecialistAgent("specialist_billing", "billing", device=device),
            "account": SpecialistAgent("specialist_account", "account", device=device),
            "technical": SpecialistAgent(
                "specialist_technical", "technical", device=device
            ),
        }

        # Communication log
        self.message_log: List[Dict[str, Any]] = []

    def forward(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full multi-agent episode.
        
        Args:
            observation: Dict with 'ticket_text' key
            
        Returns:
            Dict with full episode trace and final decision
        """
        self.message_log = []

        # Step 1: Responder classifies
        responder_output = self.responder.forward(observation)
        self.message_log.append(
            {
                "step": 1,
                "agent": "responder",
                "output": responder_output,
            }
        )

        # Step 2: Coordinator routes
        coordinator_input = {
            "classification": responder_output["classification"],
            "solution": responder_output["solution"],
            "confidence": responder_output["confidence"],
            "embedding": responder_output["embedding"],
        }
        coordinator_output = self.coordinator.forward(coordinator_input)
        self.message_log.append(
            {
                "step": 2,
                "agent": "coordinator",
                "output": coordinator_output,
            }
        )

        # Step 3: Specialist processes if routed
        routing_action = coordinator_output["action"]
        specialist_output = None

        if routing_action < 3:  # Route to specialist
            specialist_names = ["billing", "account", "technical"]
            specialist_name = specialist_names[routing_action]
            specialist = self.specialists[specialist_name]

            specialist_output = specialist.forward(observation)
            self.message_log.append(
                {
                    "step": 3,
                    "agent": f"specialist_{specialist_name}",
                    "output": specialist_output,
                }
            )

        return {
            "message_log": self.message_log,
            "final_routing": coordinator_output["action_name"],
            "specialist_used": (
                self.message_log[-1]["agent"] if len(self.message_log) >= 3 else None
            ),
            "confidence": responder_output["confidence"],
        }

    def reset(self) -> None:
        """Reset all agents for new episode."""
        self.responder.reset()
        self.coordinator.reset()
        for specialist in self.specialists.values():
            specialist.reset()
        self.message_log = []

    def to_device(self, device: str) -> None:
        """Move all agents to device."""
        self.device = device
        self.responder.to_device(device)
        self.coordinator.to_device(device)
        for specialist in self.specialists.values():
            specialist.to_device(device)

    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Return all agents in system."""
        agents = {"responder": self.responder, "coordinator": self.coordinator}
        agents.update({f"specialist_{k}": v for k, v in self.specialists.items()})
        return agents

    @property
    def total_parameters(self) -> int:
        """Count total trainable parameters across all agents."""
        total = 0
        total += self.responder.trainable_parameters
        total += self.coordinator.trainable_parameters
        for specialist in self.specialists.values():
            total += specialist.trainable_parameters
        return total
