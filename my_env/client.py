"""
OpenEnv Environment Client - Agent interface to environment server.

Provides type-safe client for interacting with environment over HTTP/WebSocket.
"""

from openenv.core import EnvClient

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from .models import TicTacToeAction, TicTacToeObservation
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from models import TicTacToeAction, TicTacToeObservation


class TicTacToeEnv(EnvClient[TicTacToeAction, TicTacToeObservation, dict]):
    """
    Client for Tic-Tac-Toe Environment.
    
    Usage:
        env = TicTacToeEnv(base_url="http://localhost:8000")
        with env.sync() as client:
            obs = client.reset()
            obs = client.step(TicTacToeAction(row=0, col=0))
    """
    pass
        