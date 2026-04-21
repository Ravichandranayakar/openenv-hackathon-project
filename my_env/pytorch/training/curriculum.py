"""Curriculum learning scheduler for task difficulty progression."""

from typing import List, Dict


class CurriculumScheduler:
    """Manages difficulty progression during training.
    
    Starts with easy tasks, gradually progresses to harder ones.
    """

    def __init__(self, stages: List[str] = None):
        """Initialize curriculum scheduler.
        
        Args:
            stages: List of difficulty stages (default: [easy, medium, hard])
        """
        self.stages = stages or ["easy", "medium", "hard"]
        self.current_stage_idx = 0
        self.episodes_in_stage = 0
        self.threshold = 100  # Episodes before progression

    def get_current_stage(self) -> str:
        """Get current difficulty stage."""
        return self.stages[self.current_stage_idx]

    def update(self, episode_reward: float) -> bool:
        """Update curriculum based on episode reward.
        
        Args:
            episode_reward: Reward from last episode
            
        Returns:
            True if progressed to next stage
        """
        self.episodes_in_stage += 1

        # Progress if enough episodes completed or reward threshold exceeded
        if (
            self.current_stage_idx < len(self.stages) - 1
            and self.episodes_in_stage >= self.threshold
        ):
            self.current_stage_idx += 1
            self.episodes_in_stage = 0
            return True

        return False

    def get_config(self) -> Dict[str, any]:
        """Get curriculum configuration."""
        return {
            "stages": self.stages,
            "current_stage": self.get_current_stage(),
            "episodes_in_stage": self.episodes_in_stage,
            "total_episodes": self.episodes_in_stage + (self.current_stage_idx * self.threshold),
        }
