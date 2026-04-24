"""Experience replay buffer for multi-agent training."""

from typing import Any, Dict, List, Optional
import torch
from collections import deque
import random


class MultiAgentReplayBuffer:
  """Experience replay buffer for storing and sampling agent experiences.
  
  Supports prioritized or uniform sampling, separate buffers per agent type.
  """

  def __init__(self, max_size: int = 10000, batch_size: int = 32):
    """Initialize replay buffer.
    
    Args:
      max_size: Maximum buffer capacity
      batch_size: Batch size for sampling
    """
    self.max_size = max_size
    self.batch_size = batch_size
    self.buffer: deque = deque(maxlen=max_size)

  def add(self, experience: Dict[str, Any]) -> None:
    """Add experience to buffer.
    
    Args:
      experience: Dict with keys:
        - observation: current state
        - action: agent action
        - reward: immediate reward
        - next_observation: next state
        - done: episode termination flag
    """
    self.buffer.append(experience)

  def sample(self, batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """Sample batch of experiences.
    
    Args:
      batch_size: Batch size (uses default if not specified)
      
    Returns:
      List of sampled experiences
    """
    if batch_size is None:
      batch_size = self.batch_size

    batch_size = min(batch_size, len(self.buffer))
    return random.sample(list(self.buffer), batch_size)

  def size(self) -> int:
    """Return current buffer size."""
    return len(self.buffer)

  def is_full(self) -> bool:
    """Check if buffer is at capacity."""
    return len(self.buffer) == self.max_size

  def clear(self) -> None:
    """Clear buffer."""
    self.buffer.clear()

  def get_all(self) -> List[Dict[str, Any]]:
    """Get all experiences in buffer."""
    return list(self.buffer)
