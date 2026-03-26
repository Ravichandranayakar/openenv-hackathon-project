"""
Ticket Resolver - Validation and Reward Calculation Engine (Improved)

Provides all ground truth rules for ticket resolution:
- Classification validation
- Category validation (within an issue type)
- Solution validation (against resolution policies)  
- Escalation decision validation
- Step-by-step reward calculation

Max reward per episode: 1.0

Backward compatible with TicketResolver class interface.
"""

from ..data.tickets import RESOLUTION_POLICIES, get_ticket_by_id


def is_valid_action_type(action_type: str) -> bool:
    """Check if action type is valid."""
    VALID_ACTIONS = [
        "classify_issue",
        "choose_solution", 
        "escalate_decision",
        "close_ticket"
    ]
    return action_type in VALID_ACTIONS


def is_valid_classification(classification: str) -> bool:
    """Check if classification is valid."""
    VALID_TYPES = ["billing", "account", "bug", "feature"]
    return classification in VALID_TYPES


def is_classification_correct(ticket_id: str, classification: str) -> bool:
    """Check if agent's classification matches ground truth."""
    if not is_valid_classification(classification):
        return False
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        return False
    return ticket["correct_type"] == classification


def is_category_valid_for_type(classification: str, category: str) -> bool:
    """Check if a category belongs to an issue type."""
    if classification not in RESOLUTION_POLICIES:
        return False
    return category in RESOLUTION_POLICIES[classification]


def is_category_correct(ticket_id: str, category: str) -> bool:
    """Check if agent's category selection matches ground truth."""
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        return False
    return ticket["correct_category"] == category


def is_solution_valid_for_category(classification: str, category: str, solution: str) -> bool:
    """Check if solution is valid for a given category."""
    if classification not in RESOLUTION_POLICIES:
        return False
    if category not in RESOLUTION_POLICIES[classification]:
        return False
    return solution in RESOLUTION_POLICIES[classification][category]


def is_solution_correct(ticket_id: str, solution: str) -> bool:
    """Check if agent's solution matches ground truth."""
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        return False
    return ticket["correct_primary_solution"] == solution


def get_escalation_flag(ticket_id: str) -> bool:
    """Get whether a ticket should be escalated per ground truth."""
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        return False
    return ticket["needs_escalation"]


def is_escalation_correct(ticket_id: str, should_escalate: bool) -> bool:
    """Check if agent's escalation decision matches ground truth."""
    correct = get_escalation_flag(ticket_id)
    return should_escalate == correct


class RewardCalculator:
    """
    Step-by-step reward calculation.
    
    Max reward per episode: 1.0
    
    Breakdown:
    - Classification: 0.2 (20%)
    - Solution: 0.3 (30%)
    - Escalation: 0.3 (30%)
    - Closure: 0.2 (20%)
    """
    
    CLASSIFICATION_REWARD = 0.2
    SOLUTION_REWARD = 0.3
    ESCALATION_REWARD = 0.3
    CLOSURE_REWARD = 0.2
    MAX_REWARD = 1.0
    
    PENALTY_CLASSIFICATION = -0.2
    PENALTY_SOLUTION = -0.3
    PENALTY_ESCALATION = -0.3
    
    @staticmethod
    def classify_step(ticket_id: str, classification: str) -> float:
        """Calculate reward for classification step."""
        if not is_valid_classification(classification):
            return 0.0
        return RewardCalculator.CLASSIFICATION_REWARD if is_classification_correct(ticket_id, classification) else RewardCalculator.PENALTY_CLASSIFICATION
    
    @staticmethod
    def solution_step(ticket_id: str, classification: str, category: str, solution: str) -> float:
        """Calculate reward for solution step."""
        if not is_category_valid_for_type(classification, category):
            return 0.0
        if not is_solution_valid_for_category(classification, category, solution):
            return RewardCalculator.PENALTY_SOLUTION
        return RewardCalculator.SOLUTION_REWARD if is_solution_correct(ticket_id, solution) else RewardCalculator.PENALTY_SOLUTION
    
    @staticmethod
    def escalation_step(ticket_id: str, should_escalate: bool) -> float:
        """Calculate reward for escalation decision."""
        return RewardCalculator.ESCALATION_REWARD if is_escalation_correct(ticket_id, should_escalate) else RewardCalculator.PENALTY_ESCALATION
    
    @staticmethod
    def closure_step(correct_escalation: bool) -> float:
        """Calculate reward for proper closure."""
        return RewardCalculator.CLOSURE_REWARD if correct_escalation else 0.0


# Backward compatibility wrapper class for existing code
class TicketResolver:
    """Determines correctness of agent actions and calculates rewards."""
    
    VALID_CLASSIFICATIONS = ["billing", "account", "bug", "feature"]
    
    VALID_ACTIONS = [
        "classify_issue",
        "choose_solution",
        "escalate_decision",
        "close_ticket",
    ]
    
    def __init__(self):
        """Initialize resolver."""
        pass
    
    def is_valid_action_type(self, action_type: str) -> bool:
        """Check if action_type is valid."""
        return is_valid_action_type(action_type)
    
    def is_valid_classification(self, classification: str) -> bool:
        """Check if classification is one of the allowed types."""
        return is_valid_classification(classification)
    
    def is_classification_correct(self, ticket_id: str, classification: str) -> bool:
        """Check if agent's classification matches ground truth."""
        return is_classification_correct(ticket_id, classification)
    
    def is_category_correct(self, ticket_id: str, category: str) -> bool:
        """Check if agent's category matches ground truth."""
        return is_category_correct(ticket_id, category)
    
    def is_category_valid_for_type(self, classification: str, category: str) -> bool:
        """Check if category is valid for this classification."""
        return is_category_valid_for_type(classification, category)
    
    def is_solution_correct(self, ticket_id: str, solution: str) -> bool:
        """Check if agent's solution matches ground truth."""
        return is_solution_correct(ticket_id, solution)
    
    def is_solution_valid_for_category(self, classification: str, category: str, solution: str) -> bool:
        """Check if solution is valid for this category."""
        return is_solution_valid_for_category(classification, category, solution)
    
    def get_escalation_flag(self, ticket_id: str) -> bool:
        """Get whether ticket should be escalated."""
        return get_escalation_flag(ticket_id)
    
    def is_escalation_correct(self, ticket_id: str, should_escalate: bool) -> bool:
        """Check if escalation decision is correct."""
        return is_escalation_correct(ticket_id, should_escalate)
    
    def get_max_reward_for_ticket(self, ticket_id: str) -> float:
        """Get max possible reward per episode."""
        return RewardCalculator.MAX_REWARD


def get_max_reward():
    """Get the maximum possible reward for an episode."""
    return RewardCalculator.MAX_REWARD
