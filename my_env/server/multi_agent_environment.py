"""Multi-Agent Environment"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import random
from datetime import datetime

class Department(str, Enum):
  BILLING = "billing"
  ACCOUNT = "account"
  TECHNICAL = "technical"
  ESCALATE = "escalate"

class ResolutionDay(int, Enum):
  DAY_1 = 1
  DAY_2 = 2
  DAY_3 = 3
  DAY_4 = 4
  DAY_5 = 5

@dataclass
class SupportTicket:
  id: str
  customer_name: str
  issue_text: str
  severity: str
  category: str
  correct_department: Department
  difficulty: str
  created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AgentDecision:
  agent_name: str
  day: ResolutionDay
  decision: str
  confidence: float
  reasoning: str
  timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ResolutionState:
  ticket: SupportTicket
  current_day: ResolutionDay = ResolutionDay.DAY_1
  agent_decisions: List[AgentDecision] = field(default_factory=list)
  team_score: float = 0.0
  episode_complete: bool = False
  resolution_message: str = ""

class MultiAgentSupportEnvironment:
  def __init__(self, difficulty: str = "easy"):
    self.difficulty = difficulty
    self.current_state: Optional[ResolutionState] = None
    self.ticket_dataset = self._load_tickets()

  def reset(self, difficulty: Optional[str] = None) -> ResolutionState:
    if difficulty:
      self.difficulty = difficulty
    matching_tickets = [t for t in self.ticket_dataset if t.difficulty == self.difficulty]
    ticket = random.choice(matching_tickets)
    self.current_state = ResolutionState(ticket=ticket, current_day=ResolutionDay.DAY_1)
    return self.current_state

  def step_router_agent(self, predicted_department: Department, confidence: float) -> Dict:
    if not self.current_state:
      raise ValueError("Call reset()")
    is_correct = predicted_department == self.current_state.ticket.correct_department
    reward = 0.2 if is_correct else -0.2
    decision = AgentDecision("router", ResolutionDay.DAY_1, predicted_department.value, confidence, "Router")
    self.current_state.agent_decisions.append(decision)
    self.current_state.team_score += reward
    return {"agent": "router", "day": 1, "correct": is_correct, "reward": reward, "team_score": self.current_state.team_score}

  def step_resolver_agent(self, proposed_solution: str, confidence: float) -> Dict:
    if not self.current_state:
      raise ValueError("Call reset()")
    self.current_state.current_day = ResolutionDay.DAY_2
    is_plausible = len(proposed_solution) > 10
    reward = 0.15 if is_plausible else -0.15
    decision = AgentDecision("resolver", ResolutionDay.DAY_2, proposed_solution, confidence, "Resolver")
    self.current_state.agent_decisions.append(decision)
    self.current_state.team_score += reward
    return {"agent": "resolver", "day": 2, "plausible": is_plausible, "reward": reward, "team_score": self.current_state.team_score}

  def step_manager_agent(self, should_escalate: bool) -> Dict:
    if not self.current_state:
      raise ValueError("Call reset()")
    self.current_state.current_day = ResolutionDay.DAY_3
    should_actually_escalate = self.current_state.ticket.severity in ["high", "critical"] or self.current_state.ticket.difficulty == "hard"
    is_correct = should_escalate == should_actually_escalate
    reward = 0.25 if is_correct else -0.25
    decision = AgentDecision("manager", ResolutionDay.DAY_3, "escalate" if should_escalate else "proceed", 1.0, "Manager")
    self.current_state.agent_decisions.append(decision)
    self.current_state.team_score += reward
    return {"agent": "manager", "day": 3, "correct": is_correct, "reward": reward, "team_score": self.current_state.team_score}

  def step_quality_agent(self, satisfaction_score: float) -> Dict:
    if not self.current_state:
      raise ValueError("Call reset()")
    self.current_state.current_day = ResolutionDay.DAY_5
    quality_reward = satisfaction_score * 0.4
    decision = AgentDecision("quality", ResolutionDay.DAY_5, f"satisfaction_{satisfaction_score:.2f}", 1.0, "Quality")
    self.current_state.agent_decisions.append(decision)
    self.current_state.team_score += quality_reward
    self.current_state.episode_complete = True
    final_score = max(-1.0, min(1.0, self.current_state.team_score))
    return {"agent": "quality", "day": 5, "episode_complete": True, "final_team_score": final_score}

  def _load_tickets(self) -> List[SupportTicket]:
    return [
      SupportTicket("T001", "Alice", "I was charged twice", "low", "billing", Department.BILLING, "easy"),
      SupportTicket("T002", "Bob", "Forgot password", "low", "account", Department.ACCOUNT, "easy"),
      SupportTicket("T003", "Charlie", "App crashes", "high", "technical", Department.TECHNICAL, "medium"),
      SupportTicket("T004", "Diana", "Refund needed", "high", "billing", Department.BILLING, "medium"),
      SupportTicket("T005", "Eve", "Database down SLA breach", "critical", "technical", Department.TECHNICAL, "hard"),
      SupportTicket("T006", "Frank", "Complex billing dispute", "critical", "billing", Department.BILLING, "hard"),
    ]
