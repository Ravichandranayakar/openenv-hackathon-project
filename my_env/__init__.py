"""Customer Support OpenEnv Environment

Clean API for agents and reviewers:

    from my_env import SupportAction, SupportObservation, CustomerSupportEnvironment
    
    env = CustomerSupportEnvironment()
    obs = env.reset()
    action = SupportAction(action_type="classify_issue", classification="bug")
    obs = env.step(action)
"""

# Type definitions (Gymnasium-style)
# Import from root level (models.py moved to root for OpenEnv validation)
import sys
from pathlib import Path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from models import SupportAction, SupportObservation

# Environment (4-phase support ticket resolution)
from my_env.server.customer_support_environment import CustomerSupportEnvironment

# Data & Logic (internal)
from my_env.server.data.tickets import TICKETS, RESOLUTION_POLICIES, get_random_ticket
from my_env.server.logic.ticket_resolver import TicketResolver, RewardCalculator

# Client (HTTP wrapper for agents - also moved to root for OpenEnv validation)
from client import CustomerSupportEnv

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
