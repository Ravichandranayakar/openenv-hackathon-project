import gradio as gr
import json
from datetime import datetime

def build_gradio_app(app, web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    del web_manager, action_fields, metadata, is_chat_env, quick_start_md

    def step_action(agent_action, confidence, solution, history_state):
        # Build action payload based on agent_action
        action_dict = {}
        
        if "Bid" in agent_action:
            if not confidence:
                return ("", "<div style='background:#fee; color:#c33; padding:10px; border-radius:4px;'>❌ Error: Confidence required for bidding</div>", 
                        render_action_history(history_state), "Missing confidence", history_state, "")
            try:
                conf_val = float(confidence)
            except ValueError:
                return ("", "<div style='background:#fee; color:#c33; padding:10px; border-radius:4px;'>❌ Error: Confidence must be a number</div>", 
                        render_action_history(history_state), "Invalid confidence", history_state, "")
                
            agent_prefix = agent_action.split(" ")[0].lower()
            action_dict["action_type"] = f"{agent_prefix}_bid"
            action_dict["confidence"] = conf_val
            
        elif "Execute" in agent_action:
            if not solution:
                return ("", "<div style='background:#fee; color:#c33; padding:10px; border-radius:4px;'>❌ Error: Solution required for execution</div>", 
                        render_action_history(history_state), "Missing solution", history_state, "")
            agent_prefix = agent_action.split(" ")[0].lower()
            action_dict["action_type"] = f"{agent_prefix}_execute"
            action_dict["solution"] = solution
            
        elif "Evaluate" in agent_action:
            agent_prefix = agent_action.split(" ")[0].lower()
            action_dict["action_type"] = f"{agent_prefix}_evaluate"
            # Optional evaluate specific fields can go here
        
        try:
            r = client.post("/step", json={"action": action_dict})
            if r.status_code == 200:
                data = r.json()
                obs = data.get("observation", {})
                reward = data.get("reward", 0.0)
                history_state = history_state or []
                
                step_info = {
                    "step_num": len(history_state) + 1,
                    "action_type": action_dict["action_type"],
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
        """Render 4 metric boxes: Step Reward, Total Reward, Score, Phase"""
        if not obs:
            obs = {}
        
        last_action_reward = current_reward or 0.0
        total_reward = obs.get("episode_reward", 0.0) or 0.0
        score = obs.get("episode_score", 0.0) or 0.0
        phase = obs.get("phase", "unknown").upper()
        
        html = f"""
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; background: none; border: none;'>
            <div style='background: #0d6efd; color: white; padding: 20px; border-radius: 0; text-align: center; border: none;'>
                <div style='font-size: 12px; font-weight: 600;'>Last Action Reward</div>
                <div style='font-size: 28px; font-weight: 700; margin-top: 8px;'>{last_action_reward:+.2f}</div>
            </div>
            <div style='background: #fd7e14; color: white; padding: 20px; border-radius: 0; text-align: center; border: none;'>
                <div style='font-size: 12px; font-weight: 600;'>Total Team Reward</div>
                <div style='font-size: 28px; font-weight: 700; margin-top: 8px;'>{total_reward:.2f}</div>
            </div>
            <div style='background: #198754; color: white; padding: 20px; border-radius: 0; text-align: center; border: none;'>
                <div style='font-size: 12px; font-weight: 600;'>Score</div>
                <div style='font-size: 28px; font-weight: 700; margin-top: 8px;'>{score*100:.0f}%</div>
            </div>
            <div style='background: #6c757d; color: white; padding: 20px; border-radius: 0; text-align: center; border: none;'>
                <div style='font-size: 12px; font-weight: 600;'>Current Phase</div>
                <div style='font-size: 18px; font-weight: 700; margin-top: 8px;'>{phase}</div>
            </div>
        </div>
        """
        return html

    def render_ticket_view(obs):
        """Render current ticket with metadata tags and message"""
        if not obs:
            return "<div style='color: #bbb; padding: 30px; text-align: center; background: #23232b; border-radius: 4px;'>Reset to load ticket</div>"
        
        # New multi_agent_negotiation structure nests ticket inside openenv 'obs'
        ticket = obs.get("ticket", {})
        message = ticket.get("message", "No message") if isinstance(ticket, dict) else obs.get("message", "No message")
        severity = ticket.get("severity", "unknown").upper() if isinstance(ticket, dict) else obs.get("severity", "unknown").upper()
        ticket_id = ticket.get("id", "—") if isinstance(ticket, dict) else obs.get("ticket_id", "—")
        
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
        
        feedback = obs.get("message", "")
        
        html = f"""
        <div style='background: #23232b; padding: 15px; border-radius: 4px; margin-top: 15px;'>
            <div style='font-size: 13px; color: #bbb; font-weight: 600; margin-bottom: 8px;'>Environment System Feedback</div>
            <div style='font-size: 14px; color: #eee; line-height: 1.6;'>{feedback if feedback else "Waiting for action..."}</div>
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

    with gr.Blocks(css=custom_css, title="Customer Support Multi-Agent OpenEnv") as demo:
        gr.Markdown("""
        # Customer Support Command Center (ROUND 2)
        Manage support tickets manually using the 3-Phase Bidding Protocol.
        """)
        
        # ======== ZONE 1: INPUT ========
        with gr.Group():
            gr.Markdown("### Step-by-Step Action Input")
            gr.Markdown("<div style='background: #1a3a3a; padding: 12px; border-radius: 4px; font-size: 12px; color: #aaa; margin-bottom: 15px;'><b> Instructions:</b> Phase 1: Bidding → Phase 2: Winning Agent Executes → Phase 3: Manager Evaluates</div>")
            
            with gr.Row():
                agent_action = gr.Dropdown(
                    choices=[
                        "Technical Bid", "Billing Bid", "Account Bid", "Manager Bid",
                        "Technical Execute", "Billing Execute", "Account Execute",
                        "Manager Evaluate"
                    ],
                    value="Technical Bid",
                    label="Agent & Action Selection",
                    scale=2
                )
            
            with gr.Row(visible=True) as row_bid:
                confidence = gr.Textbox(label="Confidence (0.0 to 1.0)", placeholder="e.g. 0.95", scale=1)
            
            with gr.Row(visible=False) as row_execute:
                solution = gr.Textbox(label="Proposed Solution", placeholder="e.g. clear_cache_restart", scale=1)
            
            with gr.Row():
                step_btn = gr.Button("EXECUTE AGENT ACTION", variant="primary", scale=2)
                reset_btn = gr.Button("RESET EPISODE", scale=1)
                state_btn = gr.Button("GET STATE", scale=1)
            
            # Update form visibility based on action type
            def update_form_visibility(action):
                is_bid = "Bid" in action
                is_exec = "Execute" in action
                return [
                    gr.update(visible=is_bid),   # row_bid
                    gr.update(visible=is_exec)   # row_execute
                ]
            
            agent_action.change(
                update_form_visibility,
                inputs=[agent_action],
                outputs=[row_bid, row_execute]
            )
        
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
            gr.Markdown("### Agent Decison Timeline")
            history_html = gr.HTML("<div style='padding: 20px; color: #999; text-align: center;'>No actions yet</div>")
        
        # ======== ZONE 6: RAW JSON (Debug) ========
        with gr.Group():
            gr.Markdown("### Raw Observation Dump (Debug)")
            raw_json = gr.Code(language="json", label="Full Observation JSON", interactive=False)
        
        # State management
        history_state = gr.State([])
        
        # Event handlers
        step_btn.click(
            step_action,
            inputs=[agent_action, confidence, solution, history_state],
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
