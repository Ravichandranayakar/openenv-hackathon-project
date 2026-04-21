"""Specialist agents for domain-specific ticket resolution."""

from typing import Any, Dict
import torch
import torch.nn as nn

from my_env.pytorch.agents.base_agent import BaseAgent
from my_env.pytorch.models.transformer_encoder import TicketEncoder


class SpecialistAgent(BaseAgent):
    """Specialist agent for domain-specific ticket handling.
    
    Each specialist focuses on one domain (billing, account, technical).
    """

    def __init__(self, agent_id: str, specialist_type: str, device: str = "cpu"):
        """Initialize specialist agent.
        
        Args:
            agent_id: Unique identifier
            specialist_type: 'billing', 'account', or 'technical'
            device: 'cpu' or 'cuda'
        """
        super().__init__(agent_id, agent_type=specialist_type)
        self.device = device
        self.specialist_type = specialist_type

        # Encoder (shared/frozen)
        self.encoder = TicketEncoder(embedding_dim=768, frozen=True)

        # Specialist-specific solution head
        self.solution_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 8),  # 8 specialist-specific solutions
        )

        # Confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.to(device)

    def forward(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Process ticket with specialist expertise.
        
        Args:
            observation: Dict with 'ticket_text' key
            
        Returns:
            Dict with:
                - solution: int (0-7)
                - confidence: float
                - explanation: str
        """
        ticket_text = observation.get("ticket_text", "")

        # Encode
        embedding = self.encoder(ticket_text)

        # Specialist solution
        logits = self.solution_head(embedding.unsqueeze(0))
        solution = torch.argmax(logits, dim=1).item()

        # Confidence
        confidence = self.confidence_head(embedding.unsqueeze(0)).item()

        return {
            "solution": solution,
            "confidence": confidence,
            "specialist_type": self.specialist_type,
            "explanation": f"Specialist {self.specialist_type} recommends solution {solution}",
        }

    def to_device(self, device: str) -> None:
        """Move to device."""
        self.device = device
        self.encoder.to(device)
        self.solution_head.to(device)
        self.confidence_head.to(device)

    @property
    def trainable_parameters(self) -> int:
        """Count trainable parameters."""
        count = sum(p.numel() for p in self.solution_head.parameters())
        count += sum(p.numel() for p in self.confidence_head.parameters())
        return count
