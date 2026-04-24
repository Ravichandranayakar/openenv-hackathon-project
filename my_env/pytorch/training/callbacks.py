"""Training callbacks for logging and monitoring."""

from typing import Any, Dict, Optional
from datetime import datetime
import json


class TrainingCallback:
  """Base callback for training events."""

  def on_episode_start(self, episode: int) -> None:
    """Called at episode start."""
    pass

  def on_episode_end(self, episode: int, metrics: Dict[str, Any]) -> None:
    """Called at episode end."""
    pass

  def on_step(self, step: int, metrics: Dict[str, Any]) -> None:
    """Called at each training step."""
    pass


class LoggingCallback(TrainingCallback):
  """Logs training progress to file."""

  def __init__(self, log_file: str):
    """Initialize logging callback.
    
    Args:
      log_file: Path to log file
    """
    self.log_file = log_file
    self.logs = []

  def on_episode_end(self, episode: int, metrics: Dict[str, Any]) -> None:
    """Log episode metrics."""
    log_entry = {
      "timestamp": datetime.now().isoformat(),
      "episode": episode,
      "metrics": metrics,
    }
    self.logs.append(log_entry)

    # Write to file
    with open(self.log_file, "a") as f:
      f.write(json.dumps(log_entry) + "\n")
