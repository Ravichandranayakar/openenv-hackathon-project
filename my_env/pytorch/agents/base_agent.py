"""Base agent class for multi-agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class BaseAgent(ABC):
    """Abstract base class for all agents in multi-agent system.
    
    Provides interface for agent interactions, message passing, and learning.
    """

    def __init__(self, agent_id: str, agent_type: str):
        """Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (e.g., 'responder', 'coordinator')
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.message_queue: list[Dict[str, Any]] = []
        self.is_training = True

    @abstractmethod
    def forward(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Compute agent action given observation.
        
        Args:
            observation: Current environment observation
            
        Returns:
            Action dictionary containing agent's decision
        """
        pass

    def send_message(self, recipient: str, message: Dict[str, Any]) -> None:
        """Send message to another agent.
        
        Args:
            recipient: ID of recipient agent
            message: Message payload
        """
        msg = {
            "sender": self.agent_id,
            "recipient": recipient,
            "payload": message,
        }
        # Message routing handled by MultiAgentSystem
        pass

    def receive_message(self, sender: str, message: Dict[str, Any]) -> None:
        """Receive message from another agent.
        
        Args:
            sender: ID of sender agent
            message: Message payload
        """
        self.message_queue.append({"sender": sender, "payload": message})

    def get_action(self) -> Any:
        """Extract executable action from agent state."""
        pass

    def update(self, reward: float, done: bool, info: Optional[Dict] = None) -> None:
        """Update agent with experience (learning step).
        
        Args:
            reward: Immediate reward received
            done: Whether episode finished
            info: Additional information dict
        """
        pass

    def reset(self) -> None:
        """Reset agent for new episode."""
        self.message_queue = []

    def to_device(self, device: str) -> None:
        """Move agent to device (CPU/GPU)."""
        pass

    @property
    def trainable_parameters(self) -> int:
        """Return count of trainable parameters."""
        return 0
