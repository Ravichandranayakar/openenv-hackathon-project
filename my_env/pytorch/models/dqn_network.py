"""DQN network for coordinator routing decisions."""

from typing import List
import torch
import torch.nn as nn


class CoordinatorDQN(nn.Module):
  """Deep Q-Network for multi-agent coordination routing.
  
  Takes ticket representation and outputs Q-values for each routing action.
  Supports dueling architecture for improved learning stability.
  """

  def __init__(
    self,
    input_dim: int = 773,
    action_dim: int = 5,
    hidden_dims: List[int] = [256, 128, 64],
    use_dueling: bool = True,
    dropout_rate: float = 0.1,
  ):
    """Initialize DQN network.
    
    Args:
      input_dim: Input dimension (embedding + classification + confidence)
      action_dim: Number of routing actions
      hidden_dims: Hidden layer dimensions
      use_dueling: Whether to use dueling architecture
      dropout_rate: Dropout rate for regularization
    """
    super().__init__()
    self.input_dim = input_dim
    self.action_dim = action_dim
    self.use_dueling = use_dueling

    # Build feature extraction layers
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(prev_dim, hidden_dim))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(dropout_rate))
      prev_dim = hidden_dim

    self.feature_net = nn.Sequential(*layers)

    if use_dueling:
      # Dueling architecture: separate value and advantage streams
      self.value_stream = nn.Sequential(
        nn.Linear(hidden_dims[-1], 64),
        nn.ReLU(),
        nn.Linear(64, 1),
      )

      self.advantage_stream = nn.Sequential(
        nn.Linear(hidden_dims[-1], 64),
        nn.ReLU(),
        nn.Linear(64, action_dim),
      )
    else:
      # Standard DQN: single output layer
      self.q_layer = nn.Linear(hidden_dims[-1], action_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute Q-values for actions.
    
    Args:
      x: Input tensor of shape (batch_size, input_dim)
      
    Returns:
      Q-values tensor of shape (batch_size, action_dim)
    """
    features = self.feature_net(x)

    if self.use_dueling:
      # Dueling combination
      value = self.value_stream(features) # (batch_size, 1)
      advantage = self.advantage_stream(features) # (batch_size, action_dim)

      # Normalize advantage
      advantage = advantage - advantage.mean(dim=1, keepdim=True)

      # Combine: Q = V + (A - mean(A))
      q_values = value + advantage
    else:
      q_values = self.q_layer(features)

    return q_values

  def get_action(self, x: torch.Tensor, epsilon: float = 0.0) -> int:
    """Get action using epsilon-greedy policy.
    
    Args:
      x: Input tensor
      epsilon: Exploration probability (0 = greedy)
      
    Returns:
      Action index
    """
    if torch.rand(1).item() < epsilon:
      return torch.randint(0, self.action_dim, (1,)).item()
    else:
      q_values = self.forward(x.unsqueeze(0))
      return torch.argmax(q_values, dim=1).item()
