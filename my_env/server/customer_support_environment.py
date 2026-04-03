"""
Customer Support Environment - OpenEnv Implementation

Implements the Environment interface for customer support ticket handling.
Agents learn to classify issues, provide solutions, and escalate when needed.
"""

from uuid import uuid4
from typing import Optional
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Support both in-repo and standalone imports
try:
    # In-repo imports (my_env package structure)
    from ..models import SupportAction, SupportObservation
    from .data.tickets import get_random_ticket, get_ticket_by_id
    from .logic.ticket_resolver import TicketResolver, RewardCalculator
except ImportError:
    # Standalone/Docker imports (models.py at root)
    import sys
    from pathlib import Path
    root = Path(__file__).parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models import SupportAction, SupportObservation
    from my_env.server.data.tickets import get_random_ticket, get_ticket_by_id
    from my_env.server.logic.ticket_resolver import TicketResolver, RewardCalculator


class CustomerSupportEnvironment(Environment):
    """
    Customer Support OpenEnv Environment
    
    Agent learns to handle support tickets by:
    1. Classifying issue type
    2. Providing solutions
    3. Deciding when to escalate
    4. Closing tickets appropriately
    """
    
    SUPPORTS_CONCURRENT_SESSIONS = False  # Single environment instance for all requests
    
    TASKS = {
        1: {"name": "Easy - Simple ticket classification", "difficulty": "easy"},
        2: {"name": "Medium - Mixed ticket types with escalation", "difficulty": "medium"},
        3: {"name": "Hard - Complex cases requiring expertise", "difficulty": "hard"},
    }
    
    def __init__(self):
        """Initialize environment."""
        self.resolver = TicketResolver()
        self.current_ticket = None
        self.current_task_id = 1
        # Initialize state with default values - will be updated on reset
        self._state = State(episode_id="not-started", step_count=0)
        self.step_count = 0
        self.total_reward = 0.0
        self.classification_done = False
        self.solution_done = False
        self.escalation_handled = False
        self.agent_classification = None  # <-- Track agent's choice
        self.agent_escalation_decision = None  # <-- Track agent's escalation choice!
    
    def reset(self) -> SupportObservation:
        """
        Reset environment and load a new ticket.
        
        Returns:
            Initial observation with ticket details
        """
        # Load random ticket based on task difficulty
        self.current_ticket = get_random_ticket(self.current_task_id)
        
        # Initialize state
        self._state = State(
            episode_id=str(uuid4()),
            step_count=0
        )
        
        # Reset flags
        self.step_count = 0
        self.total_reward = 0.0
        self.classification_done = False
        self.solution_done = False
        self.escalation_handled = False
        self.agent_classification = None  # <-- Reset agent's choice
        self.agent_escalation_decision = None  # <-- Reset agent's escalation choice
        
        return self._observation(
            status="open",
            reward=0.0,
            done=False,
            resolution_message="Ticket loaded. Please classify the issue type."
        )
    
    def set_task(self, task_id: int) -> SupportObservation:
        """
        Switch to a different task difficulty.
        
        Args:
            task_id: 1 (Easy), 2 (Medium), 3 (Hard)
        
        Returns:
            Observation with new task
        """
        if task_id not in self.TASKS:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")
        self.current_task_id = task_id
        return self.reset()
    
    def step(self, action: SupportAction) -> SupportObservation:
        """
        Process agent action and return updated observation.
        
        Args:
            action: SupportAction with action_type and parameters
        
        Returns:
            Updated SupportObservation with reward
        """
        try:
            # Validate action exists
            if action is None:
                return self._error_observation("Action cannot be None")
            
            # Convert dict to SupportAction if needed (for compatibility)
            if isinstance(action, dict):
                try:
                    action = SupportAction(**action)
                except Exception as e:
                    return self._error_observation(f"Invalid action format: {str(e)}")
            
            # Validate action has required fields
            if not hasattr(action, 'action_type'):
                return self._error_observation("Action must have action_type field")
            
            # Validate action_type is recognized
            action_type = action.action_type
            if not self.resolver.is_valid_action_type(action_type):
                return self._error_observation(
                    f"Invalid action_type: {action_type}. "
                    f"Must be one of: {', '.join(self.resolver.VALID_ACTIONS)}"
                )
            
            # Increment step count
            self._state.step_count += 1
            self.step_count += 1
            
            # Process action based on type
            if action.action_type == "classify_issue":
                return self._handle_classify(action)
            
            elif action.action_type == "choose_solution":
                return self._handle_choose_solution(action)
            
            elif action.action_type == "escalate_decision":
                return self._handle_escalation_decision(action)
            
            elif action.action_type == "close_ticket":
                return self._handle_close(action)
            
            else:
                return self._error_observation(f"Unknown action: {action.action_type}")
        
        except Exception as e:
            return self._error_observation(f"Error processing action: {str(e)}")
    
    def _handle_classify(self, action: SupportAction) -> SupportObservation:
        """Handle classify_issue action - Step 1 of 4."""
        # Check if episode was started (reset called)
        if self.current_ticket is None:
            return self._error_observation(
                "No episode in progress. Call POST /reset first to load a ticket."
            )
        
        if self.classification_done:
            return self._error_observation("Classification already done")
        
        # Safely access classification field
        classification = getattr(action, 'classification', None)
        if classification is None:
            return self._error_observation("classification field required for classify_issue")
        
        # Validate classification is one of the allowed types
        if not self.resolver.is_valid_classification(classification):
            return self._error_observation(
                f"Invalid classification: {classification}. "
                f"Must be one of: {', '.join(self.resolver.VALID_CLASSIFICATIONS)}"
            )
        
        # Check if classification is correct
        is_correct = self.resolver.is_classification_correct(
            self.current_ticket["id"],
            action.classification
        )
        
        # Calculate reward step-by-step
        classification_reward = RewardCalculator.classify_step(
            self.current_ticket["id"],
            action.classification
        )
        
        self.total_reward += classification_reward
        self.classification_done = True
        self.agent_classification = action.classification  # <-- Store agent's choice!
        
        # Create feedback message with ground truth
        correct_type = self.current_ticket["correct_type"]
        if is_correct:
            message = (
                f"Classified as '{action.classification}'. "
                f"[OK] CORRECT! (+{classification_reward:.1f}) -> Next: Select solution category."
            )
        else:
            message = (
                f"Classified as '{action.classification}'. "
                f"[FAIL] INCORRECT. Correct answer: '{correct_type}' "
                f"(+{classification_reward:.1f}) Learn: '{correct_type}' issues are about {correct_type} problems."
            )
        
        return self._observation(
            status="classified",
            reward=classification_reward,
            done=False,
            classification=action.classification,
            correct_classification=is_correct,
            classification_reward=classification_reward,
            resolution_message=message
        )
    
    def _handle_choose_solution(self, action: SupportAction) -> SupportObservation:
        """Handle choose_solution action - Step 2 of 4."""
        if not self.classification_done:
            return self._error_observation(
                "Must classify issue first (classify_issue action)"
            )
        
        if self.solution_done:
            return self._error_observation("Solution already chosen")
        
        # Safely access fields
        solution = getattr(action, 'solution', None)
        category = getattr(action, 'category', None)
        
        if solution is None:
            return self._error_observation("solution field required for choose_solution")
        
        if category is None:
            return self._error_observation("category field required for choose_solution")
        
        # Use the agent's CHOSEN classification from step 1, not ground truth!
        classification_for_validation = self.agent_classification
        if classification_for_validation is None:
            return self._error_observation(
                "Internal error: Classification not recorded. Must call classify_issue first."
            )
        
        # For this step, use the AGENT'S classification to validate solutions
        is_category_valid = self.resolver.is_category_valid_for_type(
            classification_for_validation,
            category
        )
        
        if not is_category_valid:
            return self._error_observation(
                f"Invalid category '{category}' for type '{classification_for_validation}'"
            )
        
        # Check category correctness
        is_category_correct = self.resolver.is_category_correct(
            self.current_ticket["id"],
            category
        )
        
        # Check solution correctness
        is_solution_correct = self.resolver.is_solution_correct(
            self.current_ticket["id"],
            solution
        )
        
        # Calculate reward for this step (using AGENT'S classification, not ground truth)
        solution_reward = RewardCalculator.solution_step(
            self.current_ticket["id"],
            classification_for_validation,
            category,
            solution
        )
        
        self.total_reward += solution_reward
        self.solution_done = True
        
        # Build base message
        if is_category_correct and is_solution_correct:
            message = (
                f"Selected category '{category}', solution '{solution}'. "
                f"[OK] CORRECT! (+{solution_reward:.1f}) -> Next: Make escalation decision."
            )
        else:
            # Build ground truth feedback for incorrect answers
            feedback_parts = []
            
            if not is_category_correct:
                correct_category = self.current_ticket["correct_category"]
                feedback_parts.append(f"Category - Correct: '{correct_category}'")
            
            if not is_solution_correct:
                correct_solution = self.current_ticket["correct_solution"]
                feedback_parts.append(f"Solution - Correct: '{correct_solution}'")
            
            ground_truth = " | ".join(feedback_parts)
            
            message = (
                f"Selected category '{category}', solution '{solution}'. "
                f"[FAIL] INCORRECT. {ground_truth} (+{solution_reward:.1f}) -> Next: Make escalation decision."
            )
        
        return self._observation(
            status="solution_selected",
            reward=solution_reward,
            done=False,
            category=category,
            correct_category=is_category_correct,
            solution=solution,
            correct_solution=is_solution_correct,
            solution_reward=solution_reward,
            resolution_message=message
        )
    
    def _handle_escalation_decision(self, action: SupportAction) -> SupportObservation:
        """Handle escalate_decision action - Step 3 of 4."""
        if not self.classification_done or not self.solution_done:
            return self._error_observation(
                "Must classify and choose solution first"
            )
        
        if self.escalation_handled:
            return self._error_observation("Escalation decision already made")
        
        # Safely access should_escalate field
        should_escalate = getattr(action, 'should_escalate', None)
        if should_escalate is None:
            return self._error_observation("should_escalate field required for escalate_decision")
        
        # Check if escalation decision is correct
        is_escalation_correct = self.resolver.is_escalation_correct(
            self.current_ticket["id"],
            should_escalate
        )
        
        # Calculate reward for escalation decision
        escalation_reward = RewardCalculator.escalation_step(
            self.current_ticket["id"],
            should_escalate
        )
        
        self.total_reward += escalation_reward
        self.escalation_handled = True
        self.agent_escalation_decision = should_escalate  # <-- Store agent's decision!
        
        escalation_text = "escalate" if should_escalate else "close"
        
        # Build message with ground truth if incorrect
        if is_escalation_correct:
            message = (
                f"Decision: {escalation_text.upper()}. [OK] CORRECT! "
                f"(+{escalation_reward:.1f}) -> Next: Close ticket."
            )
        else:
            correct_decision = self.current_ticket["needs_escalation"]
            correct_text = "escalate" if correct_decision else "close"
            message = (
                f"Decision: {escalation_text.upper()}. [FAIL] INCORRECT. "
                f"Correct decision: {correct_text.upper()} "
                f"(+{escalation_reward:.1f}) -> Next: Close ticket."
            )
        
        return self._observation(
            status="escalation_decided",
            reward=escalation_reward,
            done=False,
            escalation_decision=should_escalate,
            correct_escalation=is_escalation_correct,
            escalation_reward=escalation_reward,
            resolution_message=message
        )
    
    def _handle_close(self, action: SupportAction) -> SupportObservation:
        """Handle close_ticket action - Step 4 of 4."""
        if not self.classification_done:
            return self._error_observation("Cannot close without classification")
        
        if not self.solution_done:
            return self._error_observation("Cannot close without choosing solution")
        
        if not self.escalation_handled:
            return self._error_observation("Must make escalation decision first")
        
        # Award closure reward based on whether escalation was correct
        # Use the agent's escalation decision from the previous step
        escalation_correct = self.resolver.is_escalation_correct(
            self.current_ticket["id"],
            self.agent_escalation_decision  # Use stored agent decision
        )
        
        closure_reward = RewardCalculator.closure_step(escalation_correct)
        self.total_reward += closure_reward
        
        message = (
            f"Ticket closed. Episode complete. "
            f"Total reward: {self.total_reward:.2f}/1.0 "
            f"= {(self.total_reward/1.0)*100:.0f}%"
        )
        
        return self._observation(
            status="resolved",
            reward=closure_reward,
            done=True,
            closure_reward=closure_reward if closure_reward > 0 else 0.0,
            episode_reward=self.total_reward,
            episode_score=max(0.0, min(1.0, self.total_reward / 1.0)),
            resolution_message=message
        )
    
    def _observation(
        self,
        status: str,
        resolution_message: str = "",
        reward: float = 0.0,
        done: bool = False,
        truncated: bool = False,
        classification: Optional[str] = None,
        correct_classification: Optional[bool] = None,
        classification_reward: Optional[float] = None,
        category: Optional[str] = None,
        correct_category: Optional[bool] = None,
        solution: Optional[str] = None,
        correct_solution: Optional[bool] = None,
        solution_reward: Optional[float] = None,
        escalation_decision: Optional[bool] = None,
        correct_escalation: Optional[bool] = None,
        escalation_reward: Optional[float] = None,
        closure_reward: Optional[float] = None,
        episode_reward: Optional[float] = None,
        episode_score: Optional[float] = None,
    ) -> SupportObservation:
        """
        Create a SupportObservation with Gymnasium-style returns.
        
        Args:
            status: Current ticket status (open/classified/solution_selected/resolved/error)
            resolution_message: Feedback message from environment
            reward: Reward for THIS step (Gymnasium-style)
            done: Whether episode is complete (Gymnasium-style)
            truncated: Whether episode was truncated (Gymnasium-style)
            classification: Agent's classification for current step
            correct_classification: Whether classification was correct
            classification_reward: Reward for classification step
            category: Agent's category choice
            correct_category: Whether category was correct
            solution: Agent's solution choice
            correct_solution: Whether solution was correct
            solution_reward: Reward for solution step
            escalation_decision: Agent's escalation decision
            correct_escalation: Whether escalation decision was correct
            escalation_reward: Reward for escalation step
            closure_reward: Reward for closure step
            episode_reward: Total reward for episode
            episode_score: Normalized score
        
        Returns:
            SupportObservation with Gymnasium-style rewards
        """
        final_done = done
        final_reward = reward
        final_episode_reward = episode_reward if episode_reward is not None else self.total_reward
        
        return SupportObservation(
            # AGENT INPUT (only what agent needs to analyze):
            message=self.current_ticket["message"],
            severity=self.current_ticket["severity"],
            
            # HIDDEN FROM AGENT (prevent cheating):
            # [X] ticket_id - (would enable memorization instead of learning)
            # [X] status - (would hint which step we're on)
            # [X] task_id - (would hint difficulty level)
            # [X] task_name - (would hint difficulty level)
            # [X] step_count - (would hint progress)
            ticket_id="",  # Hidden
            status="",  # Hidden
            task_id=0,  # Hidden
            task_name="",  # Hidden
            step_count=0,  # Hidden
            
            # FEEDBACK (how agent learns):
            classification=classification,
            correct_classification=correct_classification,
            classification_reward=classification_reward,
            category=category,
            correct_category=correct_category,
            solution=solution,
            correct_solution=correct_solution,
            solution_reward=solution_reward,
            escalation_decision=escalation_decision,
            correct_escalation=correct_escalation,
            escalation_reward=escalation_reward,
            closure_reward=closure_reward,
            
            # GYMNASIUM-STYLE (episode tracking):
            reward=final_reward,
            done=final_done,
            truncated=truncated,
            resolution_message=resolution_message,
            episode_reward=final_episode_reward,
            episode_score=episode_score if episode_score is not None else 0.0,
        )
    
    def _error_observation(self, error_message: str) -> SupportObservation:
        """Return error observation and mark episode done."""
        penalty = -0.5
        self.total_reward += penalty
        return SupportObservation(
            # AGENT INPUT (only what agent needs):
            message=self.current_ticket["message"] if self.current_ticket else "No ticket loaded",
            severity=self.current_ticket.get("severity", "low") if self.current_ticket else "low",
            
            # HIDDEN FROM AGENT (prevent cheating):
            ticket_id="",  # Hidden
            status="",  # Hidden
            task_id=0,  # Hidden
            task_name="",  # Hidden
            step_count=0,  # Hidden
            
            # GYMNASIUM-STYLE
            reward=penalty,
            done=True,
            truncated=False,
            episode_reward=self.total_reward,
            episode_score=max(0.0, self.total_reward / 1.0),
            resolution_message=f"ERROR: {error_message}",
        )
    
    @property
    def state(self) -> State:
        """Get current episode state."""
        return self._state
    
    def grade_episode(self) -> dict:
        """
        Grade the episode on a 0.0-1.0 scale.
        
        Grading breakdown (Max reward per episode: 1.0):
        - Step 1 - Classification: 0.2
        - Step 2 - Solution/Category: 0.3
        - Step 3 - Escalation decision: 0.3
        - Step 4 - Closure: 0.2
        
        Score = total_reward / 1.0
        
        Returns:
            Dict with score (0.0-1.0)
        """
        max_reward = RewardCalculator.MAX_REWARD  # 1.0
        score = max(0.0, min(1.0, self.total_reward / max_reward))
        return {"score": score}
