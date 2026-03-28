"""Customer Support OpenEnv Environment

Clean API for agents and reviewers:

    from my_env import SupportAction, SupportObservation, CustomerSupportEnvironment
    
    env = CustomerSupportEnvironment()
    obs = env.reset()
    action = SupportAction(action_type="classify_issue", classification="bug")
    obs = env.step(action)
"""

# Type definitions
from my_env.models import SupportAction, SupportObservation

# Environment (4-phase support ticket resolution)
from my_env.server.customer_support_environment import CustomerSupportEnvironment

# Data & Logic (internal)
from my_env.server.data.tickets import TICKETS, RESOLUTION_POLICIES, get_random_ticket
from my_env.server.logic.ticket_resolver import TicketResolver, RewardCalculator

# Client (HTTP wrapper for agents)
from my_env.client import CustomerSupportEnv

__all__ = [
    "SupportAction",
    "SupportObservation",
    "CustomerSupportEnvironment",
    "CustomerSupportEnv",
    "TICKETS",
    "RESOLUTION_POLICIES",
    "get_random_ticket",
    "TicketResolver",
    "RewardCalculator",
]
