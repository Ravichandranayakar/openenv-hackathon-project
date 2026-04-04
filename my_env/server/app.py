"""
FastAPI application for the Customer Support OpenEnv Environment.
Uses OpenEnv create_app for stable REST API + manual Gradio mounting for visible UI.
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Install: pip install 'openenv-core[core]>=0.2.1'"
    ) from e

import gradio as gr

try:
    from ..models import SupportAction, SupportObservation
    from .customer_support_environment import CustomerSupportEnvironment
    from .gradio_ui import build_gradio_app
except ImportError:
    from models import SupportAction, SupportObservation
    from my_env.server.customer_support_environment import CustomerSupportEnvironment
    from my_env.server.gradio_ui import build_gradio_app

# Create the app with OpenEnv's REST API endpoints (/reset, /step, /state, /health, /schema, etc.)
app = create_app(
    CustomerSupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="customer_support_env",
    max_concurrent_envs=1,
)




# Build Gradio UI and mount at root path (after all API endpoints)
gradio_app = build_gradio_app(
    web_manager=None,
    action_fields=[],
    metadata=None,
    is_chat_env=False,
    title="Customer Support OpenEnv Playground",
    quick_start_md="",
)

gr.mount_gradio_app(app, gradio_app, path="/")


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
