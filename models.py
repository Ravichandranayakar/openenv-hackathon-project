"""
Customer Support Environment - Type Definitions

API CONTRACT (Gymnasium-Style):
================================

INTERACTION PATTERN:
    1. Agent calls:  obs = env.reset()
    2. Agent reads: obs.ticket_id, obs.message, obs.severity, obs.status
    3. Agent sends: action = SupportAction(action_type="classify_issue", classification="bug")
    4. Environment returns: obs = env.step(action)
    5. Agent reads:
       - State: obs.ticket_id, obs.message, obs.status, obs.step_count, obs.task_id
       - Feedback: obs.resolution_message, obs.correct_classification, obs.correct_solution, etc.
       - Reward: obs.reward (latest step reward)
       - Episode: obs.done (is episode complete?), obs.episode_reward (total), obs.episode_score (0.0-1.0)

ACTION FORMAT (Agent -> Environment):
    {
      "action_type": "classify_issue" | "choose_solution" | "escalate_decision" | "close_ticket",
      "classification": "billing" | "account" | "bug" | "feature",        [if classify_issue]
      "category": "duplicate_charge" | "password" | "app_crash" | ...,    [if choose_solution]
      "solution": "refund_duplicate_charge" | "send_reset" | ...,         [if choose_solution]
      "should_escalate": true | false,                                     [if escalate_decision]
      "escalate_reason": "requires_manager" | ...                         [optional]
    }

OBSERVATION FORMAT (Environment -> Agent):
    State (what agent needs to know):
      - ticket_id: str           (unique ID)
      - message: str             (customer's problem)
      - severity: str            (low/medium/high)
      - status: str              (open/classified/solution_selected/resolved/error)
      - step_count: int          (steps taken so far)
      - task_id: int             (1=Easy, 2=Medium, 3=Hard)
      
    Feedback (how agent did):
      - resolution_message: str  (feedback with ground truth if wrong)
      - correct_classification: bool | null
      - correct_category: bool | null
      - correct_solution: bool | null
      - correct_escalation: bool | null
      
    Reward (Gymnasium-style):
      - reward: float            (points for THIS step: 0.0-0.3)
      - done: bool               (is episode complete?)
      - truncated: bool          (was episode cut short?)
      - episode_reward: float    (total for entire episode: 0.0-1.0)
      - episode_score: float     (normalized: 0.0-1.0)

REWARD SCHEME:
    Phase 1: Classify issue        -> 0.2 points max
    Phase 2: Choose solution       -> 0.3 points max
    Phase 3: Escalation decision   -> 0.3 points max
    Phase 4: Close ticket          -> 0.2 points max
    ----
    Total per episode: 1.0 point (normalized 0.0-1.0 score)
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
    Observation the environment returns to the agent (Gymnasium-style).
    
    Maps to Gymnasium return: (observation, reward, done, truncated, info)
    """
    
    # STATE (What the agent needs to know about the ticket/episode)
    
    # Ticket information
    ticket_id: str = Field(..., description="Unique ticket ID")
    message: str = Field(..., description="Customer's message/problem description")
    severity: str = Field(..., description="Severity level: low, medium, high")
    
    # Episode tracking
    task_id: int = Field(..., description="Task difficulty: 1 (Easy), 2 (Medium), 3 (Hard)")
    task_name: str = Field(..., description="Human-readable task name")
    step_count: int = Field(default=0, description="Number of steps taken in this episode")
    
    # Current status
    status: str = Field(
        ..., 
        description="Current state: open, classified, solution_selected, escalation_decided, resolved, error"
    )
    
    # FEEDBACK (How well the agent is doing - step by step)
    
    # Step 1: Classification feedback
    classification: Optional[str] = Field(None, description="Agent's issue type classification")
    correct_classification: Optional[bool] = Field(None, description="Whether classification is correct")
    classification_reward: Optional[float] = Field(None, description="Points for this step (0.0-0.2)")
    
    # Step 2: Category & Solution feedback
    category: Optional[str] = Field(None, description="Agent's chosen category")
    correct_category: Optional[bool] = Field(None, description="Whether category matches issue type")
    solution: Optional[str] = Field(None, description="Agent's proposed solution ID")
    correct_solution: Optional[bool] = Field(None, description="Whether solution is correct")
    solution_reward: Optional[float] = Field(None, description="Points for this step (0.0-0.3)")
    
    # Step 3: Escalation feedback
    escalation_decision: Optional[bool] = Field(None, description="Agent's decision: True=escalate, False=close")
    correct_escalation: Optional[bool] = Field(None, description="Whether escalation decision is correct")
    escalation_reward: Optional[float] = Field(None, description="Points for this step (0.0-0.3)")
    
    # Step 4: Closure feedback
    closure_reward: Optional[float] = Field(None, description="Points for proper closure (0.0-0.2)")
    
    # GYMNASIUM-STYLE RETURNS (reward, done, truncated, info)
    
    reward: float = Field(default=0.0, description="Reward for THIS step (latest action)")
    done: bool = Field(default=False, description="Whether episode is complete (reached terminal state)")
    truncated: bool = Field(default=False, description="Whether episode was truncated (max steps reached)")
    
    # MESSAGE & SUMMARY
    resolution_message: Optional[str] = Field(
        None, 
        description="Feedback message with ground truth answers when agent is wrong"
    )
    
    # EPISODE SUMMARY
    episode_reward: float = Field(0.0, description="Total reward accumulated (0.0-1.0)")
    episode_score: float = Field(0.0, description="Normalized score: episode_reward / 1.0") 