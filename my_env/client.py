"""
OpenEnv Environment Client - Agent interface to environment server.

Provides type-safe client for interacting with customer support environment over HTTP/WebSocket.
"""

from openenv.core import EnvClient

try:
    from models import SupportAction, SupportObservation
except ImportError:
    from my_env.models import SupportAction, SupportObservation


class CustomerSupportEnv(EnvClient[SupportAction, SupportObservation, dict]):
    """
    Client for Customer Support Environment.
    
    Usage:
        env = CustomerSupportEnv(base_url="http://localhost:8000")
        with env.sync() as client:
            obs = client.reset()
            action = SupportAction(action_type="classify_issue", classification="billing")
            obs = client.step(action)
    """
    pass
