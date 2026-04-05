import gradio as gr
import json
from datetime import datetime

def build_gradio_app(app, web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    del web_manager, action_fields, metadata, is_chat_env, quick_start_md

    def step_action(action_type, classification, category, solution, should_escalate, escalate_reason, history_state):
        # Build action payload based on action_type
        # Only include fields relevant to the current step
        action_dict = {"action_type": action_type}
        
        if action_type == "classify_issue":
            if not classification:
                return ("", "<div style='background:#fee; color:#c33; padding:10px; border-radius:4px;'>❌ Error: Classification required for classify_issue</div>", 
                        render_action_history(history_state), "Missing classification", history_state, "")
            action_dict["classification"] = classification
        
        elif action_type == "choose_solution":
            if not category or not solution:
                return ("", "<div style='background:#fee; color:#c33; padding:10px; border-radius:4px;'>❌ Error: Category and Solution required for choose_solution</div>", 
                        render_action_history(history_state), "Missing category or solution", history_state, "")
            action_dict["category"] = category
            action_dict["solution"] = solution
        
        elif action_type == "escalate_decision":
            if should_escalate is None or should_escalate.strip() == "":
                return ("", "<div style='background:#fee; color:#c33; padding:10px; border-radius:4px;'>❌ Error: Should Escalate required (true/false)</div>", 
                        render_action_history(history_state), "Missing escalation decision", history_state, "")
            action_dict["should_escalate"] = should_escalate.lower() == "true"
            if escalate_reason and escalate_reason.strip():
                action_dict["escalate_reason"] = escalate_reason
        
        elif action_type == "close_ticket":
            # close_ticket only needs action_type
            pass
        
        try:
            r = client.post("/step", json=action_dict)
            if r.status_code == 200:
                data = r.json()
                obs = data.get("observation", {})
                reward = data.get("reward", 0.0)
                history_state = history_state or []
                
                step_info = {
                    "step_num": len(history_state) + 1,
                    "action_type": action_type,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "reward": reward,
                    "status": obs.get("status", "unknown")
                }
                history_state.append(step_info)
                
                return (
                    render_ticket_view(obs),
                    render_scoreboard(obs, reward),
                    render_action_history(history_state),
                    render_formatted_feedback(obs),
                    history_state,
                    json.dumps(data, indent=2)
                )
            else:
                error_msg = f"Error {r.status_code}: {r.text[:200]}"
                return ("", f"<div style='background:#fee; color:#c33; padding:10px; border-radius:4px;'>❌ Backend Error: Check Raw JSON</div>", 
                        render_action_history(history_state), error_msg, history_state, f"Error {r.status_code}")
        except Exception as e:
            return ("", f"<div style='background:#fee; color:#c33; padding:10px; border-radius:4px;'>❌ Error: {str(e)}</div>", 
                    render_action_history(history_state), f"Exception: {str(e)}", history_state, str(e))

    def reset_env():
        try:
            r = client.post("/reset", json={})
            if r.status_code == 200:
                data = r.json()
                obs = data.get("observation", {})
                return (
                    render_ticket_view(obs),
                    render_scoreboard(obs, 0.0),
                    render_action_history([]),
                    render_formatted_feedback(obs),
                    [],
                    json.dumps(data, indent=2)
                )
            else:
                return ("", "<div style='background:#fee; color:#c33; padding:10px;'>❌ Reset failed</div>", 
                        "", "Error", [], "")
        except Exception as e:
            return ("", f"<div style='background:#fee; color:#c33; padding:10px;'>❌ Error: {str(e)}</div>", 
                    "", f"Error", [], str(e))

    def get_state(history_state):
        try:
            r = client.get("/state")
            if r.status_code == 200:
                data = r.json()
                obs = data.get("observation", {})
                reward = data.get("reward", 0.0)
                return (
                    render_ticket_view(obs),
                    render_scoreboard(obs, reward),
                    render_action_history(history_state),
                    render_formatted_feedback(obs),
                    history_state,
                    json.dumps(data, indent=2)
                )
            else:
                return ("", "", "", "Error retrieving state", history_state, "")
        except Exception as e:
            return ("", "", "", f"Error: {str(e)}", history_state, "")

    def render_scoreboard(obs, current_reward):
        """Render 4 metric boxes: Step Reward, Total Reward, Score, Status"""
        if not obs:
            obs = {}
        
        last_action_reward = current_reward or 0.0
        total_reward = obs.get("episode_reward", 0.0) or 0.0
        score = obs.get("episode_score", 0.0) or 0.0
        status = obs.get("status", "unknown").upper()
        done = obs.get("episode_done", False)
        
        status_text = "DONE" if done else "RUNNING" if status != "ERROR" else "ERROR"
        
        html = f"""
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; background: none; border: none;'>
            <div style='background: #0d6efd; color: white; padding: 20px; border-radius: 0; text-align: center; border: none;'>
                <div style='font-size: 12px; font-weight: 600;'>Last Action Reward</div>
                <div style='font-size: 28px; font-weight: 700; margin-top: 8px;'>{last_action_reward:+.2f}</div>
            </div>
            <div style='background: #fd7e14; color: white; padding: 20px; border-radius: 0; text-align: center; border: none;'>
                <div style='font-size: 12px; font-weight: 600;'>Total Reward</div>
                <div style='font-size: 28px; font-weight: 700; margin-top: 8px;'>{total_reward:.2f}</div>
            </div>
            <div style='background: #198754; color: white; padding: 20px; border-radius: 0; text-align: center; border: none;'>
                <div style='font-size: 12px; font-weight: 600;'>Score</div>
                <div style='font-size: 28px; font-weight: 700; margin-top: 8px;'>{score*100:.0f}%</div>
            </div>
            <div style='background: #6c757d; color: white; padding: 20px; border-radius: 0; text-align: center; border: none;'>
                <div style='font-size: 12px; font-weight: 600;'>Status</div>
                <div style='font-size: 18px; font-weight: 700; margin-top: 8px;'>{status_text}</div>
            </div>
        </div>
        """
        return html

    def render_ticket_view(obs):
        """Render current ticket with metadata tags and message"""
        if not obs:
            return "<div style='color: #bbb; padding: 30px; text-align: center; background: #23232b; border-radius: 4px;'>Reset to load ticket</div>"
        
        message = obs.get("message", "No message")
        severity = obs.get("severity", "unknown").upper()
        ticket_id = obs.get("ticket_id", "—")
        
        html = f"""
        <div style='background: #23232b; padding: 20px; border-radius: 4px;'>
            <div style='margin-bottom: 15px; display: flex; gap: 10px; flex-wrap: wrap;'>
                <span style='background: #353542; color: #eee; padding: 6px 12px; border-radius: 3px; font-size: 12px;'>
                    Severity: {severity}
                </span>
                <span style='background: #353542; color: #eee; padding: 6px 12px; border-radius: 3px; font-size: 12px;'>
                    ID: {ticket_id}
                </span>
            </div>
            <div style='background: #181820; padding: 15px; border-radius: 4px;'>
                <div style='font-size: 13px; color: #bbb; font-weight: 600; margin-bottom: 8px;'>Customer Message</div>
                <div style='font-size: 15px; color: #eee; line-height: 1.6;'>{message}</div>
            </div>
        </div>
        """
        return html

    def render_formatted_feedback(obs):
        """Render system feedback/resolution message"""
        if not obs:
            return ""
        
        resolution = obs.get("resolution_message", "")
        
        html = f"""
        <div style='background: #23232b; padding: 15px; border-radius: 4px; margin-top: 15px;'>
            <div style='font-size: 13px; color: #bbb; font-weight: 600; margin-bottom: 8px;'>System Feedback</div>
            <div style='font-size: 14px; color: #eee; line-height: 1.6;'>{resolution if resolution else "Waiting for action..."}</div>
        </div>
        """
        return html

    def render_action_history(history_state):
        """Render timeline of all actions"""
        if not history_state:
            return "<div style='color: #888; padding: 20px; text-align: center; font-size: 13px; background: #23232b; border-radius: 4px;'>No actions yet. Click EXECUTE STEP to start.</div>"
        
        html = "<div style='font-size: 13px; line-height: 2;'>"
        for h in history_state:
            step_num = h.get("step_num", 0)
            action = h.get("action_type", "unknown")
            reward = h.get("reward", 0.0)
            status = h.get("status", "unknown")
            html += f"""
            <div style='padding: 10px; background: #181820; border-radius: 3px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;'>
                <span style='flex: 1; color: #eee;'><b>Step {step_num}:</b> {action}</span>
                <span style='font-weight: 700; margin-right: 15px; color: #eee;'>{reward:+.2f}</span>
                <span style='color: #bbb; font-size: 12px; min-width: 80px;'>{status}</span>
            </div>
            """
        html += "</div>"
        return html

    # CSS for simple clean styling
    custom_css = """
    body {
        background: #181820;
        color: #eee;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .gr-textbox input, .gr-textbox textarea {
        background: #23232b;
        border: 1px solid #353542;
        color: #eee;
        padding: 10px 12px;
        font-size: 14px;
    }
    .gr-button {
        font-weight: 600;
        border-radius: 4px;
        padding: 12px 20px;
        font-size: 14px;
    }
    .gr-button-primary {
        background: #0d6efd !important;
        border: none;
    }
    .gr-group {
        border: none;
        background: #23232b;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .gr-form {
        background: #23232b;
        border: none;
        border-radius: 0;
        padding: 0;
    }
    .gr-markdown h3 {
        margin-top: 0;
        margin-bottom: 15px;
        font-size: 16px;
        font-weight: 600;
        color: #eee;
    }
    .gr-markdown h1 {
        margin-bottom: 10px;
        font-size: 24px;
        color: #eee;
    }
    """

    with gr.Blocks(css=custom_css, title="Customer Support OpenEnv") as demo:
        gr.Markdown("""
        # Customer Support Command Center
        Manage support tickets. Classify. Solve. Escalate. Close.
        """)
        
        # ======== ZONE 1: INPUT ========
        with gr.Group():
            gr.Markdown("### Step-by-Step Action Input")
            gr.Markdown("<div style='background: #1a3a3a; padding: 12px; border-radius: 4px; font-size: 12px; color: #aaa; margin-bottom: 15px;'><b>📝 Instructions:</b> Select action type → Fill in required field(s) → Click EXECUTE STEP → Repeat for next step</div>")
            with gr.Row():
                action_type = gr.Dropdown(
                    choices=["classify_issue", "choose_solution", "escalate_decision", "close_ticket"],
                    value="classify_issue",
                    label="Action Type",
                    info="Step 1→2→3→4",
                    scale=3
                )
            with gr.Row():
                classification = gr.Textbox(label="Classification (Step 1)", placeholder="billing, account, bug, feature", scale=1)
                category = gr.Textbox(label="Category (Step 2)", placeholder="duplicate_charge, password, api, etc.", scale=1)
            with gr.Row():
                solution = gr.Textbox(label="Solution (Step 2)", placeholder="refund_duplicate_charge, reset_password_link, etc.", scale=2)
            with gr.Row():
                should_escalate = gr.Textbox(label="Escalate? (Step 3)", placeholder="true or false", scale=1)
                escalate_reason = gr.Textbox(label="Reason (if true)", placeholder="fraud_suspected, security_breach, etc.", scale=1)
            
            with gr.Row():
                step_btn = gr.Button("EXECUTE STEP", variant="primary", scale=2)
                reset_btn = gr.Button("RESET", scale=1)
                state_btn = gr.Button("GET STATE", scale=1)
        
        # ======== ZONE 2: REWARD SCOREBOARD ========
        scoreboard_html = gr.HTML("<div style='padding: 20px; color: #999; text-align: center;'>Reset to see metrics</div>")
        
        # ======== ZONE 3: CURRENT TICKET ========
        with gr.Group():
            gr.Markdown("### Current Ticket")
            ticket_html = gr.HTML("<div style='padding: 30px; color: #999; text-align: center;'>Reset to load ticket</div>")
        
        # ======== ZONE 4: SYSTEM FEEDBACK ========
        feedback_html = gr.HTML()
        
        # ======== ZONE 5: ACTION HISTORY ========
        with gr.Group():
            gr.Markdown("### Action Timeline")
            history_html = gr.HTML("<div style='padding: 20px; color: #999; text-align: center;'>No actions yet</div>")
        
        # ======== ZONE 6: RAW JSON (Debug) ========
        with gr.Group():
            gr.Markdown("### Raw Observation (Debug)")
            raw_json = gr.Code(language="json", label="Full Observation JSON", interactive=False)
        
        # State management
        history_state = gr.State([])
        
        # Event handlers
        step_btn.click(
            step_action,
            inputs=[action_type, classification, category, solution, should_escalate, escalate_reason, history_state],
            outputs=[ticket_html, scoreboard_html, history_html, feedback_html, history_state, raw_json]
        )
        
        reset_btn.click(
            reset_env,
            outputs=[ticket_html, scoreboard_html, history_html, feedback_html, history_state, raw_json]
        )
        
        state_btn.click(
            get_state,
            inputs=[history_state],
            outputs=[ticket_html, scoreboard_html, history_html, feedback_html, history_state, raw_json]
        )
    
    return demo
