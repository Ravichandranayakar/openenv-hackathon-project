"""Transformer-based ticket encoder using distilbert."""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TicketEncoder(nn.Module):
    """Encode ticket text into 768-dimensional embeddings using distilbert.
    
    Uses distilbert-base-uncased for fast, efficient encoding.
    Can be frozen for transfer learning or fine-tuned for the task.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 768,
        frozen: bool = True,
    ):
        """Initialize encoder.
        
        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Output embedding dimension (must match model)
            frozen: If True, freeze encoder weights (no gradients)
        """
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.frozen = frozen

        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text to embedding.
        
        Args:
            text: Input ticket text
            max_length: Maximum token length for truncation
            
        Returns:
            Embedding tensor of shape (embedding_dim,)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # Forward pass
        with torch.no_grad() if self.frozen else torch.enable_grad():
            outputs = self.model(**inputs)

        # Extract [CLS] token embedding (first token)
        # outputs.last_hidden_state shape: (batch_size, seq_length, embedding_dim)
        cls_embedding = outputs.last_hidden_state[0, 0, :]  # (embedding_dim,)

        return cls_embedding

    def encode_batch(self, texts: list, max_length: int = 512) -> torch.Tensor:
        """Encode batch of texts.
        
        Args:
            texts: List of ticket texts
            max_length: Maximum token length
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        with torch.no_grad() if self.frozen else torch.enable_grad():
            outputs = self.model(**inputs)

        # Extract CLS embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, embedding_dim)

        return cls_embeddings

    def set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze encoder."""
        self.frozen = frozen
        for param in self.model.parameters():
            param.requires_grad = not frozen

    def to(self, device: str):
        """Move to device."""
        super().to(device)
        self.model.to(device)
        return self
