"""
Task Graders for Customer Support Environment

Graders evaluate the performance of the agent on different task difficulties.
"""


class BaseTaskGrader:
    """Base grader class for all tasks."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
    
    def grade(self, steps: int, rewards: list, success: bool) -> dict:
        """Grade a task execution.
        
        Args:
            steps: Number of steps taken
            rewards: List of rewards from each step
            success: Whether the task completed successfully
        
        Returns:
            dict with score, feedback, etc.
        """
        raise NotImplementedError


class EasyTaskGrader(BaseTaskGrader):
    """Grader for easy customer support tasks."""
    
    def __init__(self):
        super().__init__("easy_task")
    
    def grade(self, steps: int, rewards: list, success: bool) -> dict:
        """Easy tasks focus on single-step classification.
        
        Score based on:
        - Completing quickly (1-2 steps) = +0.3
        - Getting positive first reward = +0.4
        - Success = +0.3
        """
        score = 0.0
        feedback = []
        
        # Quick completion bonus
        if steps <= 2:
            score += 0.3
            feedback.append("Completed efficiently")
        elif steps <= 3:
            score += 0.15
        
        # Reward quality
        if rewards and rewards[0] > 0:
            score += 0.4
            feedback.append("Good initial classification")
        else:
            score += 0.2
        
        # Success bonus
        if success:
            score += 0.3
            feedback.append("Task completed")
        else:
            score += 0.1
        
        # Normalize to (0.01, 0.99)
        final_score = max(0.01, min(0.99, score / 1.0))
        
        return {
            "task_id": self.task_id,
            "score": final_score,
            "steps": steps,
            "feedback": ", ".join(feedback),
            "success": success
        }


class MediumTaskGrader(BaseTaskGrader):
    """Grader for medium difficulty customer support tasks."""
    
    def __init__(self):
        super().__init__("medium_task")
    
    def grade(self, steps: int, rewards: list, success: bool) -> dict:
        """Medium tasks involve multi-step reasoning.
        
        Score based on:
        - Completing all 4 steps = +0.4
        - Average reward > 0 = +0.4
        - Success = +0.2
        """
        score = 0.0
        feedback = []
        
        # Step completion
        if steps >= 4:
            score += 0.4
            feedback.append("All steps completed")
        elif steps >= 3:
            score += 0.3
            feedback.append("Most steps completed")
        else:
            score += 0.1
        
        # Reward quality
        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            if avg_reward > 0:
                score += 0.4
                feedback.append("Positive average reward")
            elif avg_reward > -0.2:
                score += 0.2
                feedback.append("Mixed rewards")
            else:
                score += 0.05
        
        # Success bonus
        if success:
            score += 0.2
            feedback.append("Task succeeded")
        else:
            score += 0.05
        
        # Normalize to (0.01, 0.99)
        final_score = max(0.01, min(0.99, score / 1.0))
        
        return {
            "task_id": self.task_id,
            "score": final_score,
            "steps": steps,
            "feedback": ", ".join(feedback),
            "success": success
        }


class HardTaskGrader(BaseTaskGrader):
    """Grader for hard customer support tasks."""
    
    def __init__(self):
        super().__init__("hard_task")
    
    def grade(self, steps: int, rewards: list, success: bool) -> dict:
        """Hard tasks require optimal decision-making.
        
        Score based on:
        - Completing all 4 steps with high rewards = +0.5
        - Total reward > 0.5 = +0.3
        - Success = +0.2
        """
        score = 0.0
        feedback = []
        
        # Step completion with quality
        if steps >= 4:
            if rewards and sum(rewards) > 0.5:
                score += 0.5
                feedback.append("High quality completion")
            else:
                score += 0.3
                feedback.append("All steps completed")
        elif steps >= 3:
            score += 0.2
            feedback.append("Partial completion")
        else:
            score += 0.05
        
        # Total reward quality
        if rewards:
            total_reward = sum(rewards)
            if total_reward > 0.5:
                score += 0.3
                feedback.append("Strong overall performance")
            elif total_reward > 0:
                score += 0.15
                feedback.append("Positive overall reward")
            else:
                score += 0.05
        
        # Success bonus
        if success:
            score += 0.2
            feedback.append("Task succeeded")
        
        # Normalize to (0.01, 0.99)
        final_score = max(0.01, min(0.99, score / 1.0))
        
        return {
            "task_id": self.task_id,
            "score": final_score,
            "steps": steps,
            "feedback": ", ".join(feedback),
            "success": success
        }
