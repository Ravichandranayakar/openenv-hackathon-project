"""Responder agent for ticket classification and initial solution selection."""

from typing import Any, Dict
import torch
import torch.nn as nn

from my_env.pytorch.agents.base_agent import BaseAgent
from my_env.pytorch.models.transformer_encoder import TicketEncoder


class ResponderAgent(BaseAgent):
    """Responder agent: Classifies tickets and suggests initial solutions.
    
    - Encodes ticket text using distilbert
    - Classifies into issue category (4 classes)
    - Suggests initial solution (16 options)
    - Outputs (classification, solution, confidence)
    """

    def __init__(self, agent_id: str = "responder", device: str = "cpu"):
        """Initialize responder agent.
        
        Args:
            agent_id: Unique identifier
            device: 'cpu' or 'cuda'
        """
        super().__init__(agent_id, agent_type="responder")
        self.device = device

        # Encoder: distilbert-base-uncased (frozen initially)
        self.encoder = TicketEncoder(embedding_dim=768, frozen=True)

        # Classification head: 768 -> 4 classes
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
        )

        # Solution selector: 768 -> 16 solutions
        self.solution_selector = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 16),
        )

        # Confidence predictor: 768 -> 1
        self.confidence_predictor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.to(device)

    def forward(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Process ticket and return classification + solution.
        
        Args:
            observation: Dict with 'ticket_text' key
            
        Returns:
            Dict with:
                - classification: int (0-3)
                - solution: int (0-15)
                - confidence: float (0-1)
                - embedding: tensor (768,)
        """
        ticket_text = observation.get("ticket_text", "")

        # Encode ticket
        embedding = self.encoder(ticket_text)  # (768,)

        # Classify
        logits_class = self.classifier(embedding.unsqueeze(0))  # (1, 4)
        classification = torch.argmax(logits_class, dim=1).item()  # int

        # Select solution
        logits_solution = self.solution_selector(embedding.unsqueeze(0))  # (1, 16)
        solution = torch.argmax(logits_solution, dim=1).item()  # int

        # Predict confidence
        confidence = self.confidence_predictor(embedding.unsqueeze(0)).item()  # float

        return {
            "classification": classification,
            "solution": solution,
            "confidence": confidence,
            "embedding": embedding.detach().cpu(),
        }

    def to_device(self, device: str) -> None:
        """Move to device."""
        self.device = device
        self.encoder.to(device)
        self.classifier.to(device)
        self.solution_selector.to(device)
        self.confidence_predictor.to(device)

    @property
    def trainable_parameters(self) -> int:
        """Count trainable parameters (excluding frozen encoder)."""
        count = 0
        count += sum(p.numel() for p in self.classifier.parameters())
        count += sum(p.numel() for p in self.solution_selector.parameters())
        count += sum(p.numel() for p in self.confidence_predictor.parameters())
        return count
