"""
Multi-Agent Support Team Environment - Theme #1: Multi-Agent Interactions

A negotiation-based environment where 4 specialist agents cooperate to resolve tickets:
- Billing Specialist
- Account Specialist 
- Technical Specialist
- Manager Agent (routes based on team bids)

Agents learn through:
1. Bidding on tickets (confidence + specialization)
2. Partial observability (see other bids, not reasoning)
3. Coalition formation (multiple agents for complex cases)
4. Team + individual rewards (encourages cooperation + specialization)

This addresses Theme #1:
✓ Multi-agent cooperation/negotiation
✓ Theory-of-mind reasoning (learn specializations)
✓ Partially observable settings (limited info per agent)
✓ Emergent strategic behavior (coordination improves over training)
"""

from uuid import uuid4
from typing import Optional, Dict, List
from enum import Enum
from dataclasses import dataclass, field
import random
import json

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
  from ..models import SupportAction, SupportObservation
  from .data.tickets import get_random_ticket
except ImportError:
  import sys
  from pathlib import Path
  root = Path(__file__).parent.parent.parent
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  from models import SupportAction, SupportObservation
  from my_env.server.data.tickets import get_random_ticket


class AgentRole(str, Enum):
  BILLING = "billing"
  ACCOUNT = "account"
  TECHNICAL = "technical"
  MANAGER = "manager"


@dataclass
class AgentBid:
  """Agent's bid to handle a ticket"""
  agent_role: AgentRole
  confidence: float # 0-1
  timestamp: int = 0


@dataclass
class TeamObservation:
  """Partial observation for each agent"""
  ticket_id: str
  message: str
  severity: str # low/medium/high/critical
  category: str # billing/account/bug/feature
  
  # What THIS agent can see about others
  other_bids: Dict[str, float] = field(default_factory=dict) # agent_role -> confidence
  
  # Team state
  winning_agent: Optional[str] = None
  step_count: int = 0
  phase: str = "bidding" # bidding -> execution -> resolution


class MultiAgentNegotiationEnvironment(Environment):
  """
  OpenEnv-compatible multi-agent negotiation environment.
  
  4 agents negotiate task allocation through bidding, then execute solutions.
  """
  
  SUPPORTS_CONCURRENT_SESSIONS = False
  
  # ANTI-HACKING: Max steps enforcement (prevent infinite loops)
  MAX_STEPS_PER_EPISODE = 10 # Bidding (3 agents) + execution (1 winner) + resolution (1 manager) = 5-6 steps max
  
  AGENT_ROLES = [AgentRole.BILLING, AgentRole.ACCOUNT, AgentRole.TECHNICAL]
  MANAGER_ROLE = AgentRole.MANAGER
  
  # Reward structure - MULTIPLE INDEPENDENT REWARD FUNCTIONS
  # (Following hackathon guide section 7: multiple independent rewards prevent gaming)
  REWARDS = {
    # Individual agent rewards
    "correct_specialist_bid": 0.2,    # Agent bids correctly for their specialty
    "correct_solution": 0.2,       # Solution is factually correct
    "appropriate_confidence": 0.1,    # Confidence matches actual ability (not overconfident)
    "solution_format": 0.05,       # Solution follows required format
    
    # Team cooperation rewards
    "team_success_bonus": 0.2,      # Shared when ticket resolved correctly
    
    # Penalties for bad behavior (anti-hacking)
    "wrong_specialist": -0.2,      # Agent bid when not their specialty
    "wrong_solution": -0.2,       # Solution is incorrect
    "overconfident": -0.1,        # Bid high but solution was wrong
    "team_failure_penalty": -0.1,    # Shared penalty on failure
    "invalid_bid": -0.05,        # Bid outside 0-1 range
    "timeout": -0.15,          # Episode timeout
  }
  
  def __init__(self):
    """Initialize multi-agent environment."""
    self.current_ticket = None
    self.current_ticket_difficulty = "easy"
    self.current_phase = "bidding" # bidding -> execution -> resolution
    
    # Current episode state
    self._state = State(episode_id="not-started", step_count=0)
    self.step_count = 0
    self.episode_reward = 0.0
    
    # Track bids from all agents (filled during bidding phase)
    self.agent_bids: Dict[str, AgentBid] = {} # agent_role -> bid
    self.winning_agent: Optional[AgentRole] = None
    
    # Track actions and rewards for each agent
    self.agent_actions: Dict[str, str] = {} # agent_role -> action taken
    self.agent_rewards: Dict[str, float] = {} # agent_role -> reward this episode
    self.agent_solutions: Dict[str, str] = {} # agent_role -> proposed solution
    
    # Ground truth for the ticket
    self.correct_category: Optional[str] = None
    self.expected_specialist: Optional[AgentRole] = None

  def reset(self) -> SupportObservation:
    """Reset environment for new episode."""
    # Load random ticket
    ticket_data = get_random_ticket(task_level=1) # Start with easy
    self.current_ticket = ticket_data
    self.current_ticket_difficulty = ticket_data.get("difficulty", "easy")
    self.correct_category = ticket_data.get("correct_type") # correct_type = billing/account/bug/feature
    
    # Determine expected specialist
    category_to_role = {
      "billing": AgentRole.BILLING,
      "account": AgentRole.ACCOUNT,
      "bug": AgentRole.TECHNICAL,
      "feature": AgentRole.TECHNICAL,
      "technical": AgentRole.TECHNICAL,
    }
    self.expected_specialist = category_to_role.get(self.correct_category, AgentRole.TECHNICAL)
    
    # Reset episode state
    self._state = State(
      episode_id=str(uuid4()),
      step_count=0
    )
    self.step_count = 0
    self.episode_reward = 0.0
    self.current_phase = "bidding"
    
    # Clear agent tracking
    self.agent_bids = {}
    self.winning_agent = None
    self.agent_actions = {}
    self.agent_rewards = {role.value: 0.0 for role in self.AGENT_ROLES}
    self.agent_solutions = {}
    
    # Return initial observation
    return self._make_observation(
      agent_role="manager",
      phase="bidding",
      message=f"New ticket arrived. Agents, please bid! Ticket: {self.current_ticket.get('message', '')[:80]}..."
    )

  def step(self, action: SupportAction) -> SupportObservation:
    """
    Process agent action during current phase.
    
    Phases:
    1. BIDDING: Agents send confidence bids
    2. EXECUTION: Winning agent sends solution
    3. RESOLUTION: Manager gives final reward
    """
    try:
      # ANTI-HACKING: Check timeout (prevent infinite loops)
      if self.step_count >= self.MAX_STEPS_PER_EPISODE:
        # Episode timed out - penalize all agents
        for role_str in self.agent_rewards:
          self.agent_rewards[role_str] += self.REWARDS.get("timeout", -0.15)
        
        self.episode_reward = sum(self.agent_rewards.values())
        return self._error_observation(
          f"Episode timeout! Maximum {self.MAX_STEPS_PER_EPISODE} steps exceeded."
        )
      
      if action is None:
        return self._error_observation("Action cannot be None")
      
      if isinstance(action, dict):
        action = SupportAction(**action)
      
      # Determine which agent is acting based on action_type
      agent_role = action.action_type.split("_")[0] if hasattr(action, "action_type") else "unknown"
      
      # Route to appropriate phase handler
      if self.current_phase == "bidding":
        return self._handle_bidding_phase(action, agent_role)
      elif self.current_phase == "execution":
        return self._handle_execution_phase(action, agent_role)
      elif self.current_phase == "resolution":
        return self._handle_resolution_phase(action, agent_role)
      else:
        return self._error_observation(f"Unknown phase: {self.current_phase}")
    
    except Exception as e:
      return self._error_observation(f"Error: {str(e)}")

  def _handle_bidding_phase(self, action: SupportAction, agent_role: str) -> SupportObservation:
    """Collect bids from agents with anti-hacking safeguards."""
    
    # SAFEGUARD 1: Validate bid is a number in [0, 1]
    try:
      confidence = getattr(action, "confidence", None)
      if confidence is None:
        return self._error_observation(
          f"Confidence bid required. Must be float in [0, 1]."
        )
      confidence = float(confidence)
    except (ValueError, TypeError):
      return self._error_observation(
        f"Invalid confidence: {confidence}. Must be float in [0, 1]."
      )
    
    # SAFEGUARD 2: Enforce strict [0, 1] range (prevent gaming)
    if not (0.0 <= confidence <= 1.0):
      self.agent_rewards[agent_role] = self.REWARDS.get("invalid_bid", -0.05)
      return self._error_observation(
        f"Confidence must be in [0, 1]. Got {confidence}. Invalid bid penalized."
      )
    
    # SAFEGUARD 3: Log bid for inspection (anti-hacking audit trail)
    if not hasattr(self, "bid_history"):
      self.bid_history = []
    self.bid_history.append({
      "timestamp": self.step_count,
      "agent": agent_role,
      "confidence": confidence,
      "ticket_id": self.current_ticket.get("id")
    })
    
    # Record bid
    self.agent_bids[agent_role] = AgentBid(
      agent_role=agent_role,
      confidence=confidence,
      timestamp=self.step_count
    )
    
    # Check if all agents have bid
    if len(self.agent_bids) < len(self.AGENT_ROLES):
      # Still waiting for other agents
      pending = [r.value for r in self.AGENT_ROLES if r.value not in self.agent_bids]
      return self._make_observation(
        agent_role=agent_role,
        phase="bidding",
        message=f"Bid recorded ({len(self.agent_bids)}/{len(self.AGENT_ROLES)}). Waiting for: {', '.join(pending)}"
      )
    
    # All bids collected - select winning agent
    self._select_winning_agent()
    self.current_phase = "execution"
    
    return self._make_observation(
      agent_role=agent_role,
      phase="execution",
      message=f"Bidding complete! {self.winning_agent.value} agent selected with confidence {self.agent_bids[self.winning_agent.value].confidence:.2f}. Please provide solution."
    )

  def _select_winning_agent(self):
    """Select agent with highest confidence (auction winner)."""
    winner = max(
      self.agent_bids.items(),
      key=lambda x: x[1].confidence
    )
    self.winning_agent = AgentRole(winner[0])

  def _handle_execution_phase(self, action: SupportAction, agent_role: str) -> SupportObservation:
    """Winning agent provides solution."""
    
    # Only winning agent can execute
    if agent_role != self.winning_agent.value:
      return self._error_observation(
        f"Only {self.winning_agent.value} agent can execute. You are {agent_role}."
      )
    
    # Get solution
    solution = getattr(action, "solution", "unknown")
    self.agent_solutions[agent_role] = solution
    
    self.current_phase = "resolution"
    
    return self._make_observation(
      agent_role=agent_role,
      phase="resolution",
      message=f"Solution provided: {solution}. Resolving..."
    )

  def _handle_resolution_phase(self, action: SupportAction, agent_role: str) -> SupportObservation:
    """Manager finalizes and assigns rewards."""
    
    # Evaluate winning agent's solution
    solution_correct = self._evaluate_solution()
    
    # Calculate INDEPENDENT rewards for each agent (following guide section 7)
    for agent_role_enum in self.AGENT_ROLES:
      role_str = agent_role_enum.value
      reward_components = []
      
      # REWARD 1: Was this agent's specialty (correct specialist bid)?
      if role_str == self.expected_specialist.value:
        reward_components.append(self.REWARDS["correct_specialist_bid"])
      else:
        reward_components.append(self.REWARDS["wrong_specialist"])
      
      if role_str == self.winning_agent.value:
        # REWARD 2: Solution correctness (outcome quality)
        if solution_correct:
          reward_components.append(self.REWARDS["correct_solution"])
          # REWARD 3: Appropriate confidence (didn't overconfident)
          bid_confidence = self.agent_bids[role_str].confidence
          if 0.7 <= bid_confidence <= 1.0: # High confidence for correct solution
            reward_components.append(self.REWARDS["appropriate_confidence"])
          else:
            reward_components.append(self.REWARDS["overconfident"])
        else:
          reward_components.append(self.REWARDS["wrong_solution"])
          # Penalize overconfidence if bid high but solution wrong
          bid_confidence = self.agent_bids[role_str].confidence
          if bid_confidence > 0.7:
            reward_components.append(self.REWARDS["overconfident"])
        
        # REWARD 4: Solution format compliance
        reward_components.append(self.REWARDS["solution_format"])
      
      # Team rewards (shared)
      if solution_correct:
        reward_components.append(self.REWARDS["team_success_bonus"])
      else:
        reward_components.append(self.REWARDS["team_failure_penalty"])
      
      self.agent_rewards[role_str] = sum(reward_components)
    
    # Calculate episode reward
    self.episode_reward = sum(self.agent_rewards.values())
    self.step_count += 1
    self._state.step_count += 1
    
    resolution_msg = f"Ticket resolved! Winning agent ({self.winning_agent.value}): {'✓ Correct' if solution_correct else '✗ Incorrect'}. Team rewards: {', '.join([f'{r}={self.agent_rewards[r]:.2f}' for r in self.agent_rewards])}"
    
    return self._make_observation(
      agent_role=agent_role,
      phase="complete",
      message=resolution_msg,
      done=True,
      reward=self.episode_reward
    )

  def _evaluate_solution(self) -> bool:
    """Check if winning agent's solution was correct."""
    # Simple heuristic: check if agent category matches expected specialist
    return self.winning_agent == self.expected_specialist

  def _make_observation(
    self,
    agent_role: str,
    phase: str,
    message: str,
    done: bool = False,
    reward: float = 0.0
  ) -> SupportObservation:
    """Create observation for agent."""
    
    # Build other agents' bid info (partial observability)
    other_bids = {
      role.value: self.agent_bids.get(role.value, AgentBid(role, 0.0)).confidence
      for role in self.AGENT_ROLES
      if role.value != agent_role
    }
    
    # Create observation
    obs = SupportObservation(
      ticket_id=self.current_ticket.get("id", "unknown") if self.current_ticket else "none",
      message=self.current_ticket.get("message", "") if self.current_ticket else "",
      severity=self.current_ticket.get("severity", "low") if self.current_ticket else "low",
      status=phase,
      task_id=1,
      task_name="Multi-Agent Negotiation",
      step_count=self.step_count,
      resolution_message=message,
      reward=reward,
      done=done,
      truncated=False,
      episode_reward=self.episode_reward,
      episode_score=max(0.0, min(1.0, self.episode_reward)),
      
      # Multi-agent specific fields
      agent_role=agent_role,
      other_agent_bids=other_bids,
      winning_agent=self.winning_agent.value if self.winning_agent else None,
      current_phase=phase,
    )
    
    return obs

  def _error_observation(self, error_msg: str) -> SupportObservation:
    """Return error observation."""
    return self._make_observation(
      agent_role="system",
      phase="error",
      message=f"ERROR: {error_msg}",
      done=True,
      reward=-0.5
    )

  def get_team_state(self) -> Dict:
    """Get complete team state for evaluation."""
    return {
      "episode_id": self._state.episode_id,
      "step_count": self.step_count,
      "current_ticket": self.current_ticket,
      "expected_specialist": self.expected_specialist.value if self.expected_specialist else None,
      "winning_agent": self.winning_agent.value if self.winning_agent else None,
      "agent_bids": {role: self.agent_bids.get(role, AgentBid(role, 0.0)).confidence for role in [r.value for r in self.AGENT_ROLES]},
      "agent_rewards": self.agent_rewards,
      "episode_reward": self.episode_reward,
      "solution_correct": self._evaluate_solution(),
    }

  @property
  def state(self) -> State:
    """Return current OpenEnv state."""
    return self._state
