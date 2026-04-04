"""
Gradio UI builder for Customer Support OpenEnv.
"""
import gradio as gr
import json

def build_gradio_app(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """Builds the Gradio UI for the Customer Support environment."""
    del action_fields, metadata, is_chat_env, quick_start_md

    def step_action(action_type, classification, category, solution, should_escalate, escalate_reason, message):
        action = {
            "action_type": action_type or "classify_issue",
            "classification": classification or "billing",
            "category": category or "general",
            "solution": solution or "refund",
            "should_escalate": should_escalate.lower() == "true",
            "escalation_reason": escalate_reason or "customer_request"
        }
        if message:
            action["message"] = message
        try:
            data = web_manager.step_environment({"action": action})
            return json.dumps(data, indent=2), "Step executed."
        except Exception as e:
            return "", f"Error: {e}"

    def reset_env():
        try:
            data = web_manager.reset_environment({})
            return json.dumps(data, indent=2), "Environment reset."
        except Exception as e:
            return "", f"Error: {e}"

    def get_state():
        try:
            data = web_manager.get_state()
            return json.dumps(data, indent=2), "State fetched."
        except Exception as e:
            return "", f"Error: {e}"

    with gr.Blocks(title=title) as demo:
        gr.Markdown("# Customer Support OpenEnv Playground")
        with gr.Row():
            with gr.Column(scale=1):
                action_type = gr.Textbox(label="Action Type", placeholder="e.g. classify_issue")
                classification = gr.Textbox(label="Classification", placeholder="e.g. billing")
                category = gr.Textbox(label="Category", placeholder="Enter category")
                solution = gr.Textbox(label="Solution", placeholder="Enter solution")
                should_escalate = gr.Textbox(label="Should Escalate", placeholder="true/false")
                escalate_reason = gr.Textbox(label="Escalate Reason", placeholder="Enter escalate reason")
                message = gr.Textbox(label="Message", placeholder="Enter message (optional)")
                step_btn = gr.Button("Step", variant="primary")
                reset_btn = gr.Button("Reset Environment", variant="secondary")
                state_btn = gr.Button("Get State", variant="secondary")
            with gr.Column(scale=1):
                output_json = gr.Textbox(label="Output / State", interactive=False, lines=12)
                status = gr.Textbox(label="Status", interactive=False)
        step_btn.click(step_action, inputs=[action_type, classification, category, solution, should_escalate, escalate_reason, message], outputs=[output_json, status])
        reset_btn.click(reset_env, outputs=[output_json, status])
        state_btn.click(get_state, outputs=[output_json, status])
    return demo
