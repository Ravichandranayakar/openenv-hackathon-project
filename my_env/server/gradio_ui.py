import gradio as gr
import json
from datetime import datetime


def build_gradio_app(app, web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    del web_manager, action_fields, metadata, is_chat_env, quick_start_md

    # ─────────────────────────────────────────────
    # HELPER: Map reward + context to human label
    # ─────────────────────────────────────────────
    def get_reward_reason(action_type, reward, status, resolution_message=""):
        """Return a short human-readable label explaining WHY this reward was given."""
        if resolution_message and "ERROR" in resolution_message.upper():
            return "❌ Invalid Action"
        if "bid" in action_type:
            if reward == 0.0:
                return "⏳ Bid Recorded"
            if reward > 0:
                return "✅ Correct Specialist Bid"
            return "⚠️ Wrong Specialist Bid"
        if "execute" in action_type:
            if reward == 0.0:
                return "⏳ Solution Recorded"
            if reward > 0:
                return "✅ Correct Solution"
            return "❌ Wrong Solution / Overconfident"
        if "evaluate" in action_type or status == "complete":
            if reward > 0:
                return "🏆 Team Success Bonus"
            return "💀 Team Failure Penalty"
        return "ℹ️ Action Processed"

    # ─────────────────────────────────────────────
    # RENDER HELPERS
    # ─────────────────────────────────────────────
    def render_scoreboard(obs, current_reward):
        if not obs:
            obs = {}
        step_reward = current_reward or 0.0
        total_reward = obs.get("episode_reward", 0.0) or 0.0
        score = obs.get("episode_score", 0.0) or 0.0
        phase = obs.get("current_phase", obs.get("status", "—")).upper()
        phase_color = {"BIDDING": "#0F766E", "EXECUTION": "#d97706", "RESOLUTION": "#7c3aed", "COMPLETE": "#16a34a", "ERROR": "#dc2626"}.get(phase, "#475569")

        return f"""
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:0;'>
            <div style='background:#0d1f1e; border:1px solid #0F766E33; border-radius:8px; padding:14px; text-align:center;'>
                <div style='font-size:11px; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;'>Step Reward</div>
                <div style='font-size:24px; font-weight:700; margin-top:4px; color:{"#22c55e" if step_reward >= 0 else "#ef4444"};'>{step_reward:+.2f}</div>
            </div>
            <div style='background:#0d1f1e; border:1px solid #0F766E33; border-radius:8px; padding:14px; text-align:center;'>
                <div style='font-size:11px; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;'>Team Reward</div>
                <div style='font-size:24px; font-weight:700; margin-top:4px; color:{"#22c55e" if total_reward >= 0 else "#ef4444"};'>{total_reward:.2f}</div>
            </div>
            <div style='background:#0d1f1e; border:1px solid #0F766E33; border-radius:8px; padding:14px; text-align:center;'>
                <div style='font-size:11px; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;'>Score</div>
                <div style='font-size:24px; font-weight:700; margin-top:4px; color:#0F766E;'>{score*100:.0f}%</div>
            </div>
            <div style='background:#0d1f1e; border:1px solid {phase_color}55; border-radius:8px; padding:14px; text-align:center;'>
                <div style='font-size:11px; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;'>Phase</div>
                <div style='font-size:14px; font-weight:700; margin-top:6px; color:{phase_color};'>{phase}</div>
            </div>
        </div>
        """

    def render_agent_status(obs):
        if not obs:
            obs = {}
        agent_bids = obs.get("other_agent_bids", {})
        winning = obs.get("winning_agent", None)
        current_phase = obs.get("current_phase", obs.get("status", "bidding"))
        agents = [
            ("technical", "⚙️", "#3b82f6"),
            ("billing", "💳", "#f59e0b"),
            ("account", "👤", "#8b5cf6"),
            ("manager", "🎯", "#0F766E"),
        ]
        rows = ""
        for name, icon, color in agents:
            bid_val = agent_bids.get(name)
            is_winner = winning == name
            badge = f"<span style='background:{color}22; color:{color}; font-size:10px; font-weight:700; padding:2px 7px; border-radius:4px; margin-left:6px;'>WIN</span>" if is_winner else ""
            bid_display = f"<span style='color:#9ca3af; font-size:11px;'>Bid: {bid_val:.2f}</span>" if bid_val is not None else "<span style='color:#4b5563; font-size:11px;'>Waiting...</span>"
            dot_color = color if is_winner else ("#22c55e" if bid_val is not None else "#374151")
            rows += f"""
            <div style='display:flex; align-items:center; padding:8px 10px; background:#111827; border-radius:6px; margin-bottom:6px; border:1px solid {"#0F766E44" if is_winner else "#1f2937"};'>
                <div style='width:8px; height:8px; border-radius:50%; background:{dot_color}; margin-right:10px; flex-shrink:0;'></div>
                <span style='font-size:13px; margin-right:4px;'>{icon}</span>
                <span style='color:#e5e7eb; font-size:13px; font-weight:600; text-transform:capitalize; flex:1;'>{name}{badge}</span>
                {bid_display}
            </div>"""
        return f"<div style='margin-top:4px;'>{rows}</div>"

    def render_ticket(obs):
        if not obs:
            return "<div style='color:#4b5563; padding:40px; text-align:center;'>Click NEW EPISODE to load a ticket.</div>"
        ticket = obs.get("ticket", {})
        message = ticket.get("message", obs.get("message", "No message"))
        severity = (ticket.get("severity", obs.get("severity", "low")) or "low").upper()
        ticket_id = ticket.get("id", obs.get("ticket_id", "—"))
        sev_color = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a", "CRITICAL": "#7c3aed"}.get(severity, "#6b7280")
        return f"""
        <div style='background:#111827; border-radius:8px; overflow:hidden;'>
            <div style='background:#0F766E22; border-bottom:1px solid #0F766E33; padding:10px 16px; display:flex; gap:10px; align-items:center;'>
                <span style='color:#9ca3af; font-size:12px; font-weight:600;'>TICKET</span>
                <span style='background:#1f2937; color:#e5e7eb; font-size:11px; font-weight:700; padding:2px 8px; border-radius:4px;'>{ticket_id}</span>
                <span style='background:{sev_color}22; color:{sev_color}; font-size:11px; font-weight:700; padding:2px 8px; border-radius:4px;'>{severity}</span>
            </div>
            <div style='padding:20px;'>
                <div style='color:#6b7280; font-size:11px; font-weight:600; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em;'>Customer Message</div>
                <div style='color:#e5e7eb; font-size:15px; line-height:1.7;'>{message}</div>
            </div>
        </div>
        """

    def render_feedback(obs):
        if not obs:
            return "<div style='color:#4b5563; font-size:13px; padding:16px; background:#111827; border-radius:8px;'>Waiting for action...</div>"
        msg = obs.get("resolution_message", obs.get("message", ""))
        is_error = "ERROR" in str(msg).upper() or obs.get("status") == "error"
        border = "#dc2626" if is_error else "#0F766E"
        icon = "❌" if is_error else "💬"
        return f"""
        <div style='background:#111827; border-left:3px solid {border}; border-radius:0 8px 8px 0; padding:14px 16px;'>
            <div style='color:#6b7280; font-size:11px; font-weight:600; margin-bottom:6px;'>ENVIRONMENT FEEDBACK {icon}</div>
            <div style='color:#e5e7eb; font-size:13px; line-height:1.6;'>{msg if msg else "Action processed."}</div>
        </div>
        """

    def render_history(history_state):
        if not history_state:
            return "<div style='color:#4b5563; font-size:13px; padding:20px; text-align:center;'>No actions yet. Start with Phase 1: Bidding.</div>"
        agent_colors = {"technical": "#3b82f6", "billing": "#f59e0b", "account": "#8b5cf6", "manager": "#0F766E"}
        html = ""
        for h in reversed(history_state):
            step_num = h.get("step_num", 0)
            action = h.get("action_type", "unknown")
            reward = h.get("reward", 0.0)
            reason = h.get("reason", "")
            agent = action.split("_")[0] if "_" in action else "system"
            color = agent_colors.get(agent, "#6b7280")
            r_color = "#22c55e" if reward >= 0 else "#ef4444"
            html += f"""
            <div style='background:#111827; border-radius:6px; padding:10px 12px; margin-bottom:6px; border:1px solid #1f2937;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='display:flex; align-items:center; gap:8px;'>
                        <span style='color:#4b5563; font-size:11px; font-weight:700;'>#{step_num}</span>
                        <span style='background:{color}22; color:{color}; font-size:10px; font-weight:700; padding:2px 7px; border-radius:4px; text-transform:uppercase;'>{agent}</span>
                        <span style='color:#9ca3af; font-size:12px;'>{action.split("_", 1)[1] if "_" in action else action}</span>
                    </div>
                    <span style='color:{r_color}; font-size:14px; font-weight:700;'>{reward:+.2f}</span>
                </div>
                {f'<div style="color:#6b7280; font-size:11px; margin-top:4px; padding-left:32px;">{reason}</div>' if reason else ""}
            </div>"""
        return html

    def render_reward_breakdown(obs):
        """Show the 11 reward functions that fired this episode — the key differentiator for judges."""
        if not obs:
            return ""
        # Only show after episode completes (done=True)
        done = obs.get("done", False)
        if not done:
            return ""

        # The 11 reward functions defined in the environment
        reward_signals = [
            ("correct_specialist_bid", +0.30, "Winning agent was the correct specialist"),
            ("correct_solution",       +0.30, "Solution matched expected resolution"),
            ("appropriate_confidence", +0.15, "High confidence + correct outcome"),
            ("solution_format",        +0.05, "Response followed required JSON schema"),
            ("team_success_bonus",     +0.20, "All agents share team win bonus"),
            ("wrong_specialist",       -0.20, "Non-specialist agent won the bid"),
            ("wrong_solution",         -0.20, "Winning agent proposed wrong solution"),
            ("overconfident",          -0.10, "High confidence bid + failed execution"),
            ("team_failure_penalty",   -0.10, "All agents share team failure penalty"),
            ("invalid_bid",            -0.05, "Bid value outside [0.0, 1.0] range"),
            ("timeout",               -0.15, "Episode exceeded max step limit"),
        ]

        rows = ""
        for name, value, description in reward_signals:
            color = "#22c55e" if value > 0 else "#ef4444"
            sign = f"+{value:.2f}" if value > 0 else f"{value:.2f}"
            rows += f"""
            <div style='display:flex; align-items:center; justify-content:space-between; padding:6px 10px;
                        background:#0d1117; border-radius:5px; margin-bottom:4px; border:1px solid #1f2937;'>
                <div>
                    <div style='font-size:11px; color:#e5e7eb; font-weight:600;'>{name}</div>
                    <div style='font-size:10px; color:#4b5563;'>{description}</div>
                </div>
                <span style='color:{color}; font-size:12px; font-weight:700; margin-left:8px; flex-shrink:0;'>{sign}</span>
            </div>"""

        total_reward = obs.get("episode_reward", 0.0)
        total_color = "#22c55e" if total_reward >= 0 else "#ef4444"
        return f"""
        <div style='border:1px solid #0F766E44; border-radius:8px; overflow:hidden;'>
            <div style='background:#0F766E22; padding:8px 12px; border-bottom:1px solid #0F766E33;'>
                <span style='font-size:11px; font-weight:700; color:#0d9488; text-transform:uppercase; letter-spacing:0.06em;'>11-Signal Reward Breakdown</span>
            </div>
            <div style='padding:10px;'>
                {rows}
                <div style='border-top:1px solid #1f2937; margin-top:8px; padding-top:8px; display:flex; justify-content:space-between;'>
                    <span style='color:#6b7280; font-size:12px; font-weight:700;'>EPISODE TOTAL</span>
                    <span style='color:{total_color}; font-size:14px; font-weight:700;'>{total_reward:+.3f}</span>
                </div>
            </div>
        </div>
        """

    # ─────────────────────────────────────────────
    # BACKEND CALLBACKS
    # ─────────────────────────────────────────────
    def step_action(agent_action, confidence, solution, history_state):
        action_dict = {}
        if "Bid" in agent_action:
            agent_prefix = agent_action.split(" ")[0].lower()
            action_dict["action_type"] = f"{agent_prefix}_bid"
            action_dict["confidence"] = float(confidence)
        elif "Execute" in agent_action:
            if not solution:
                return (render_ticket({}), render_scoreboard({}, 0), render_agent_status({}),
                        "<div style='color:#ef4444; padding:12px;'>❌ Solution required for Execute phase.</div>",
                        render_history(history_state), history_state, "")
            agent_prefix = agent_action.split(" ")[0].lower()
            action_dict["action_type"] = f"{agent_prefix}_execute"
            action_dict["solution"] = solution
        elif "Evaluate" in agent_action:
            action_dict["action_type"] = "manager_evaluate"

        try:
            r = client.post("/step", json={"action": action_dict})
            data = r.json()
            obs = data.get("observation", {})
            reward = data.get("reward", 0.0)
            history_state = history_state or []
            reason = get_reward_reason(action_dict.get("action_type", ""), reward, obs.get("status", ""), obs.get("resolution_message", ""))
            history_state.append({
                "step_num": len(history_state) + 1,
                "action_type": action_dict.get("action_type", "unknown"),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "reward": reward,
                "status": obs.get("status", "unknown"),
                "reason": reason
            })
            return (render_ticket(obs), render_scoreboard(obs, reward), render_agent_status(obs),
                    render_feedback(obs), render_history(history_state), history_state, json.dumps(data, indent=2))
        except Exception as e:
            return (render_ticket({}), render_scoreboard({}, 0), render_agent_status({}),
                    f"<div style='color:#ef4444; padding:12px;'>❌ Error: {str(e)}</div>",
                    render_history(history_state), history_state, str(e))

    def reset_env():
        try:
            r = client.post("/reset", json={})
            data = r.json()
            obs = data.get("observation", {})
            return (render_ticket(obs), render_scoreboard(obs, 0.0), render_agent_status(obs),
                    render_feedback(obs), render_history([]), [], json.dumps(data, indent=2))
        except Exception as e:
            return (render_ticket({}), render_scoreboard({}, 0), render_agent_status({}),
                    f"<div style='color:#ef4444; padding:12px;'>❌ Reset failed: {str(e)}</div>",
                    render_history([]), [], str(e))

    def get_state(history_state):
        try:
            r = client.get("/state")
            data = r.json()
            obs = data.get("observation", {})
            reward = data.get("reward", 0.0)
            return (render_ticket(obs), render_scoreboard(obs, reward), render_agent_status(obs),
                    render_feedback(obs), render_history(history_state), history_state, json.dumps(data, indent=2))
        except Exception as e:
            return (render_ticket({}), render_scoreboard({}, 0), render_agent_status({}),
                    f"<div style='color:#ef4444; padding:12px;'>❌ Error: {str(e)}</div>",
                    render_history(history_state), history_state, str(e))

    def update_form_visibility(action):
        is_bid = "Bid" in action
        is_exec = "Execute" in action
        return gr.update(visible=is_bid), gr.update(visible=is_exec)

    # ─────────────────────────────────────────────
    # CSS
    # ─────────────────────────────────────────────
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Base ── */
    body, .gradio-container {
        background: linear-gradient(135deg, #09090b 0%, #0f0f1a 50%, #1e1b4b 100%) !important;
        min-height: 100vh !important;
        font-family: 'Inter', sans-serif !important;
    }
    footer { display:none !important; }

    /* ── Glass Panel ── */
    .glass-panel {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,255,255,0.06) !important;
        padding: 20px !important;
        margin-bottom: 12px !important;
    }

    /* ── Gradio Group overrides ── */
    .gr-group {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5) !important;
    }

    /* ── Buttons ── */
    .gr-button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        letter-spacing: 0.03em !important;
        transition: all 0.2s ease !important;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #00C9FF 0%, #00f2c3 100%) !important;
        border: none !important;
        color: #0a0a12 !important;
        box-shadow: 0 0 20px rgba(0, 201, 255, 0.35) !important;
    }
    .gr-button-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 32px rgba(0, 201, 255, 0.55) !important;
    }
    .gr-button-secondary {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        color: #cbd5e1 !important;
    }
    .gr-button-secondary:hover {
        background: rgba(255,255,255,0.12) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Inputs ── */
    .gr-form { background: transparent !important; border: none !important; }
    .gr-label { color: #64748b !important; font-size: 11px !important; font-weight: 700 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
    input, textarea, select {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    /* ── Slider ── */
    .gr-slider input[type=range]::-webkit-slider-thumb { background: #00C9FF !important; }

    /* ── Code block ── */
    .gr-code { background: rgba(0,0,0,0.4) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 10px !important; }

    /* ── Dropdown ── */
    .gr-dropdown { background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px !important; color: #e2e8f0 !important; }
    """

    # ─────────────────────────────────────────────
    # LAYOUT
    # ─────────────────────────────────────────────
    with gr.Blocks(css=custom_css, title=" Multi-Agent Command Center") as demo:

        history_state = gr.State([])

        # ── Header ──
        gr.HTML("""
        <div style='padding:20px 0 16px; border-bottom:1px solid #21262d; margin-bottom:20px;'>
            <div style='display:flex; align-items:center; gap:12px;'>
                <div style='width:36px; height:36px; background:linear-gradient(135deg,#0F766E,#0d9488); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:18px;'>🤖</div>
                <div>
                    <div style='font-size:20px; font-weight:700; color:#f0f6fc; letter-spacing:-0.02em;'>BARO AI — Multi-Agent Command Center</div>
                    <div style='font-size:12px; color:#6b7280; margin-top:2px;'>3-Phase Bidding Protocol  ·  11 Independent Reward Functions  ·  OpenEnv</div>
                </div>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ═══════════════════════════
            # LEFT COLUMN — Control Panel
            # ═══════════════════════════
            with gr.Column(scale=1, min_width=200, elem_classes="glass-panel"):

                with gr.Group():
                    gr.HTML("<div style='font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;'>Episode Control</div>")
                    reset_btn = gr.Button(" NEW EPISODE", variant="primary", size="sm")
                    state_btn = gr.Button(" GET STATE", variant="secondary", size="sm")

                with gr.Group():
                    gr.HTML("<div style='font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;'>Live Scoreboard</div>")
                    scoreboard_html = gr.HTML("<div style='color:#4b5563; font-size:12px; text-align:center; padding:20px;'>Start an episode</div>")

                with gr.Group():
                    gr.HTML("<div style='font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;'>Agent Status</div>")
                    agent_status_html = gr.HTML("<div style='color:#4b5563; font-size:12px; text-align:center; padding:20px;'>Start an episode</div>")

            # ═══════════════════════════
            # CENTER COLUMN — Ticket + Action
            # ═══════════════════════════
            with gr.Column(scale=2, min_width=380, elem_classes="glass-panel"):

                with gr.Group():
                    gr.HTML("<div style='font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;'>Active Ticket</div>")
                    ticket_html = gr.HTML("<div style='color:#4b5563; font-size:13px; padding:40px; text-align:center;'>Click NEW EPISODE to load a ticket.</div>")

                with gr.Group():
                    gr.HTML("<div style='font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;'>Agent Action</div>")

                    agent_action = gr.Dropdown(
                        choices=[
                            "Technical Bid", "Billing Bid", "Account Bid",
                            "Technical Execute", "Billing Execute", "Account Execute",
                            "Manager Evaluate"
                        ],
                        value="Technical Bid",
                        label="Select Agent + Action",
                    )

                    # PHASE 1: Bidding — shows slider only
                    with gr.Row(visible=True) as row_bid:
                        confidence = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.85, step=0.01,
                            label="Confidence Score (0.0 = Not Sure → 1.0 = Certain)"
                        )

                    # PHASE 2: Execution — shows solution textbox only
                    with gr.Row(visible=False) as row_execute:
                        solution = gr.Textbox(
                            label="Proposed Solution",
                            placeholder="e.g. Reset password via email verification link",
                            lines=2
                        )

                    step_btn = gr.Button("▶  EXECUTE ACTION", variant="primary")

                with gr.Group():
                    gr.HTML("<div style='font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>Environment Feedback</div>")
                    feedback_html = gr.HTML("<div style='color:#4b5563; font-size:13px; padding:16px; background:#111827; border-radius:8px;'>Waiting for action...</div>")

            # ═══════════════════════════
            # RIGHT COLUMN — Decision Log
            # ═══════════════════════════
            with gr.Column(scale=1, min_width=260, elem_classes="glass-panel"):

                with gr.Group():
                    gr.HTML("<div style='font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;'>Decision Log</div>")
                    history_html = gr.HTML("<div style='color:#4b5563; font-size:12px; text-align:center; padding:20px;'>No actions yet.</div>")

                with gr.Group():
                    gr.HTML("<div style='font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>11-Signal Reward Breakdown</div>")
                    reward_breakdown_html = gr.HTML("<div style='color:#4b5563; font-size:11px; text-align:center; padding:16px;'>Appears after episode completes.</div>")

                with gr.Group():
                    gr.HTML("<div style='font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>Debug JSON</div>")
                    raw_json = gr.Code(language="json", label="", interactive=False)

        # ─────────────────────────────────────────────
        # WIRING
        # ─────────────────────────────────────────────
        all_outputs = [ticket_html, scoreboard_html, agent_status_html, feedback_html, history_html, reward_breakdown_html, history_state, raw_json]

        def step_with_breakdown(agent_action, confidence, solution, history_state):
            results = list(step_action(agent_action, confidence, solution, history_state))
            # results = [ticket_html, scoreboard_html, agent_status_html, feedback_html, history_html, history_state, raw_json]
            # We need to insert reward_breakdown_html at index 5
            obs_data = {}
            try:
                raw = results[6]  # raw_json
                obs_data = json.loads(raw).get("observation", {}) if raw else {}
            except Exception:
                pass
            results.insert(5, render_reward_breakdown(obs_data))
            return results

        def reset_with_breakdown():
            results = list(reset_env())
            results.insert(5, "<div style='color:#4b5563; font-size:11px; text-align:center; padding:16px;'>Appears after episode completes.</div>")
            return results

        def state_with_breakdown(history_state):
            results = list(get_state(history_state))
            obs_data = {}
            try:
                raw = results[6]
                obs_data = json.loads(raw).get("observation", {}) if raw else {}
            except Exception:
                pass
            results.insert(5, render_reward_breakdown(obs_data))
            return results

        agent_action.change(update_form_visibility, inputs=[agent_action], outputs=[row_bid, row_execute])
        step_btn.click(step_with_breakdown, inputs=[agent_action, confidence, solution, history_state], outputs=all_outputs)
        reset_btn.click(reset_with_breakdown, outputs=all_outputs)
        state_btn.click(state_with_breakdown, inputs=[history_state], outputs=all_outputs)

    return demo
