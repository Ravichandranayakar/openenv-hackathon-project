"""Data package for customer support environment."""

# Import tickets data and utilities
# Handles both relative imports (normal Python) and absolute imports (Docker)
try:
    from .tickets import TICKETS, RESOLUTION_POLICIES, get_random_ticket, get_ticket_by_id
except (ImportError, ValueError):
    # Fallback for Docker/standalone environments
    from tickets import TICKETS, RESOLUTION_POLICIES, get_random_ticket, get_ticket_by_id

__all__ = [
    "TICKETS",
    "RESOLUTION_POLICIES",
    "get_random_ticket",
    "get_ticket_by_id",
]
