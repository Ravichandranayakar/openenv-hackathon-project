"""
Customer Support Environment - Type Definitions

Defines the Action and Observation contracts between the agent and environment.
These are the ONLY fields the agent will see and send.
"""

from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class SupportAction(Action):
    """
    Action the agent sends to the environment.
    
    Fields:
    - action_type: classify_issue, choose_solution, escalate_decision, close_ticket
    - classification: billing, account, bug, feature (for classify_issue)
    - category: Specific category like duplicate_charge, password, app_crash
    - solution: Proposed solution ID (e.g., refund_duplicate_charge)
    - should_escalate: Whether to escalate
    - escalate_reason: Why escalating
    """
    
    action_type: str = Field(
        ...,
        description="Type of action: 'classify_issue', 'choose_solution', 'escalate_decision', 'close_ticket'"
    )
    
    classification: Optional[str] = Field(
        None,
        description="Issue type for classify_issue action: 'billing', 'account', 'bug', 'feature'"
    )
    
    category: Optional[str] = Field(
        None,
        description="Specific category within issue type: 'duplicate_charge', 'password', 'app_crash', etc."
    )
    
    solution: Optional[str] = Field(
        None,
        description="Solution ID for choose_solution action: 'refund_duplicate_charge', 'reset_password_link', etc."
    )
    
    should_escalate: Optional[bool] = Field(
        None,
        description="Escalation decision for escalate_decision action"
    )
    
    escalate_reason: Optional[str] = Field(
        None,
        description="Reason for escalation"
    )


class SupportObservation(Observation):
    """
    Observation the environment returns to the agent.
    
    Step-by-step feedback on:
    1. Classification (billing/account/bug/feature) - 0.2 points
    2. Category selection (duplicate_charge/password/app_crash/etc.) - no direct points but gates solution validation
    3. Solution selection - 0.3 points
    4. Escalation decision - 0.3 points
    5. Closure feedback - 0.2 points
    
    Max episode reward: 1.0
    """
    
    # Ticket information
    ticket_id: str = Field(..., description="Unique ticket ID")
    message: str = Field(..., description="Customer's message/problem")
    severity: str = Field(..., description="Severity level: low, medium, high")
    
    # Episode tracking
    task_id: int = Field(..., description="Task difficulty: 1 (Easy), 2 (Medium), 3 (Hard)")
    task_name: str = Field(..., description="Human-readable task name")
    step_count: int = Field(default=0, description="Number of steps taken")
    
    # Status
    status: str = Field(
        ..., 
        description="Current episode status: open, classified, category_selected, solution_selected, escalation_decided, resolved, error"
    )
    
    # Step 1: Classification feedback (0.2 points max)
    classification: Optional[str] = Field(None, description="Agent's classification")
    correct_classification: Optional[bool] = Field(None, description="Whether classification is correct")
    classification_reward: Optional[float] = Field(None, description="Points awarded for classification step (0.0-0.2)")
    
    # Step 2: Category feedback (gates solution validation but no direct points)
    category: Optional[str] = Field(None, description="Agent's chosen category")
    correct_category: Optional[bool] = Field(None, description="Whether category is correct for this issue type")
    
    # Step 3: Solution feedback (0.3 points max)
    solution: Optional[str] = Field(None, description="Solution attempt ID")
    correct_solution: Optional[bool] = Field(None, description="Whether solution is correct for this category")
    solution_reward: Optional[float] = Field(None, description="Points awarded for solution step (0.0-0.3)")
    
    # Step 4: Escalation feedback (0.3 points max)
    escalation_decision: Optional[bool] = Field(None, description="Agent's escalation decision (True=escalate, False=close)")
    correct_escalation: Optional[bool] = Field(None, description="Whether escalation decision matches ground truth")
    escalation_reward: Optional[float] = Field(None, description="Points awarded for escalation step (0.0-0.3)")
    
    # Step 5: Closure feedback (0.2 points max)
    closure_reward: Optional[float] = Field(None, description="Points awarded for proper closure (0.0-0.2)")
    
    # Resolution summary
    resolution_message: Optional[str] = Field(None, description="Feedback message from environment")
    episode_done: bool = Field(False, description="Whether episode is complete")
    episode_reward: float = Field(0.0, description="Total reward for episode")
    episode_score: float = Field(0.0, description="Normalized score: episode_reward / 1.0 (0.0-1.0)")
    
    # Ground truth for analysis/debugging
    ground_truth_type: Optional[str] = Field(None, description="[Hidden] Correct issue type")
    ground_truth_category: Optional[str] = Field(None, description="[Hidden] Correct category")
    ground_truth_solution: Optional[str] = Field(None, description="[Hidden] Ground truth solution")
    ground_truth_escalation: Optional[bool] = Field(None, description="[Hidden] Whether escalation needed") 