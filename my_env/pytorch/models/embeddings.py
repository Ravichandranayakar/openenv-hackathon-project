"""Embedding utilities for multi-agent system."""

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """Generic embedding layer for categorical features."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        """Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length) or (batch_size,)
            
        Returns:
            Embedded tensor
        """
        return self.embedding(x)
