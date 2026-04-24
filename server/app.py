"""Compatibility adapter: provide server/app.py for OpenEnv validator.

The validator explicitly checks for server/app.py.
This file imports and re-exports from the actual implementation in my_env.server.app.
"""
from my_env.server.app import app, main as _main


def main(host: str = "0.0.0.0", port: int = 8000):
  """Entry point - delegates to my_env.server.app:main."""
  _main(host=host, port=port)


__all__ = ["app", "main"]


if __name__ == "__main__":
  main()
