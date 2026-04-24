"""Episode evaluator for multi-agent system."""

from typing import Any, Dict, Optional, List
from my_env.pytorch.evaluation.metrics import CooperationMetrics


class EpisodeEvaluator:
  """Evaluates multi-agent episodes and computes rewards.
  
  Tracks individual and team performance.
  """

  def __init__(self):
    """Initialize evaluator."""
    self.metrics = CooperationMetrics()
    self.episode_history = []

  def evaluate_episode(
    self, environment_result: Dict[str, Any], agent_messages: List[Dict]
  ) -> Dict[str, Any]:
    """Evaluate one episode.
    
    Args:
      environment_result: Result from environment.step()
      agent_messages: Message log from multi-agent system
      
    Returns:
      Evaluation dictionary with metrics
    """
    episode_data = {
      "environment_result": environment_result,
      "agent_messages": agent_messages,
    }

    # Record metrics
    # TODO: Parse actual metrics from episode

    self.episode_history.append(episode_data)

    return {
      "episode_num": len(self.episode_history),
      "metrics": self.metrics.get_summary(),
    }

  def get_cumulative_metrics(self) -> Dict[str, Any]:
    """Get cumulative metrics across all episodes.
    
    Returns:
      Summary metrics
    """
    return self.metrics.get_summary()

  def reset(self) -> None:
    """Reset evaluator."""
    self.metrics.reset()
    self.episode_history.clear()
