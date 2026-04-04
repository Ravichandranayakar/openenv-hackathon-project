"""
Customer Support OpenEnv Server - HTTP/WebSocket interface.

Provides OpenEnv-compliant REST API for customer support environment.
Standard endpoints (/reset, /step, /state, /schema, /ws) provided by manual routing.
Uses SINGLE environment instance maintained across all requests (not using create_app).
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.requests import Request
import json
import gradio as gr

# Support both in-repo and standalone imports
try:
    from ..models import SupportAction, SupportObservation
    from .customer_support_environment import CustomerSupportEnvironment
except ImportError:
    import sys
    from pathlib import Path
    root = Path(__file__).parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # All rights reserved.
    #
    # This source code is licensed under the BSD-style license found in the
    # LICENSE file in the root directory of this source tree.

    """
    FastAPI application for the Customer Support OpenEnv Environment.
    This version uses the OpenEnv create_app pattern for robust, stable integration.
    """

    try:
        from openenv.core.env_server.http_server import create_app
    except ImportError as e:
        raise ImportError(
            "openenv-core is required for the web interface. Install dependencies with 'pip install openenv-core[core]'"
        ) from e

    try:
        from ..models import SupportAction, SupportObservation
        from .customer_support_environment import CustomerSupportEnvironment
        from .gradio_ui import build_gradio_app
    except ImportError:
        from models import SupportAction, SupportObservation
        from my_env.server.customer_support_environment import CustomerSupportEnvironment
        from my_env.server.gradio_ui import build_gradio_app

    # Create the app with web interface and Gradio integration
    app = create_app(
        CustomerSupportEnvironment,
        SupportAction,
        SupportObservation,
        env_name="customer_support_env",
        max_concurrent_envs=1,
        gradio_builder=build_gradio_app,
    )

    def main(host: str = "0.0.0.0", port: int = 8000):
        """
        Entry point for direct execution via uv run or python -m.
        """
        import uvicorn
        uvicorn.run(app, host=host, port=port)

    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8000)
        args = parser.parse_args()
        main(port=args.port)
