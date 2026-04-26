#!/usr/bin/env python3
"""
Customer Support OpenEnv - Inference Script for Hackathon
=========================================================

Multi-Agent GRPO-Trained Inference Pipeline.

4 specialist agents (Technical, Billing, Account, Manager) solve tickets
using the bidding protocol learned via GRPO training.

Trained LoRA adapters: RavichandraNayakar/openenv-multi-agent-grpo
Base model: unsloth/Meta-Llama-3.1-8B-Instruct

Environment variables:
 - API_BASE_URL: HuggingFace LLM API endpoint
 - MODEL_NAME:   LLM model name (defaults to 8B — same as trained)
 - API_KEY:      HuggingFace API token
"""

import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")

# Using the merged model with adapters already applied
MODEL_NAME = os.environ.get("MODEL_NAME", "RavichandraNayakar/openenv-grpo-merged")

API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

if not API_KEY:
    raise ValueError("API_KEY environment variable not set. Required for hackathon submission.")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
OPENENV_SERVER = os.environ.get("OPENENV_SERVER", "http://localhost:8000")

RESET_URL = f"{OPENENV_SERVER}/reset"
STEP_URL  = f"{OPENENV_SERVER}/step"

# ─────────────────────────────────────────────────────────────
# TRAINED AGENT SYSTEM PROMPTS
# These mirror the persona alignment learned via GRPO training.
# Fine-tuned adapters: RavichandraNayakar/openenv-multi-agent-grpo
# ─────────────────────────────────────────────────────────────

AGENT_PROMPTS = {
    "technical": """You are a Technical Support Specialist Agent trained via GRPO reinforcement learning.
Your role: Diagnose software bugs, crashes, API failures, and performance issues.
Specialization: You bid HIGH (0.85-0.95) when the ticket involves app crashes, errors, or technical failures.
You bid LOW (0.05-0.15) for billing or account-only issues.
When executing: Provide specific diagnostic steps, request logs, and escalate SEV1 issues.
Output ONLY valid JSON, no extra text.""",

    "billing": """You are a Billing Support Specialist Agent trained via GRPO reinforcement learning.
Your role: Resolve payment issues, refund requests, invoice disputes, and subscription changes.
Specialization: You bid HIGH (0.85-0.95) when the ticket involves charges, payments, or billing.
You bid LOW (0.05-0.15) for technical bugs or account-only issues.
When executing: Verify charges, process refunds according to policy, and document disputes.
Output ONLY valid JSON, no extra text.""",

    "account": """You are an Account Support Specialist Agent trained via GRPO reinforcement learning.
Your role: Handle login issues, password resets, account access, and profile management.
Specialization: You bid HIGH (0.85-0.95) when the ticket involves login failures or account access.
You bid LOW (0.05-0.15) for billing or technical bug issues.
When executing: Verify identity, send reset links, and check account status.
Output ONLY valid JSON, no extra text.""",

    "manager": """You are a Manager Agent trained via GRPO reinforcement learning.
Your role: Evaluate the team's solution quality and decide escalation.
You see the full ticket context and the winning agent's solution.
Decision criteria:
- should_escalate: true → if SEV1 issue, unresolved, or requires human override
- should_escalate: false → if solution is complete and customer-safe
Output ONLY valid JSON, no extra text.""",
}


def reset_environment() -> dict:
    response = requests.post(RESET_URL, json={})
    response.raise_for_status()
    return response.json().get("observation", {})


def send_action(action_payload: dict) -> dict:
    response = requests.post(STEP_URL, json={"action": action_payload})
    response.raise_for_status()
    data = response.json()
    return {
        "observation": data.get("observation", {}),
        "reward": data.get("reward", 0.0),
        "episode_reward": data.get("episode_reward", 0.0),
        "done": data.get("done", False),
    }


def get_agent_decision(agent_name: str, user_prompt: str) -> dict:
    """
    Call the 8B model (same as fine-tuned) via HF Serverless API
    with the trained agent's system prompt.
    """
    system_prompt = AGENT_PROMPTS[agent_name]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    except (json.JSONDecodeError, Exception):
        return {}


def solve_episode():
    print(f"[START] task=multi_agent_support model={MODEL_NAME}")

    obs = reset_environment()
    ticket_msg = obs.get("message", "Unknown issue")

    print(f"\n[TICKET] {ticket_msg}\n")

    step_num = 1
    user_context = f"Support ticket: {ticket_msg}\n\nRespond with JSON only."

    # ─────────────────────────────────────────────────────────
    # PHASE 1: BIDDING — 3 specialist agents bid confidence
    # ─────────────────────────────────────────────────────────
    # Technical Agent
    tech_payload = get_agent_decision(
        "technical",
        user_context + '\nOutput: {"action_type": "technical_bid", "confidence": <float 0-1>}'
    )
    if not tech_payload:
        tech_payload = {"action_type": "technical_bid", "confidence": 0.5}
    tech_payload["action_type"] = "technical_bid"
    res1 = send_action(tech_payload)
    print(f"[STEP] step={step_num} agent=technical bid={tech_payload.get('confidence', 0):.2f} reward={res1['reward']:.2f}")
    step_num += 1

    # Billing Agent
    bill_payload = get_agent_decision(
        "billing",
        user_context + '\nOutput: {"action_type": "billing_bid", "confidence": <float 0-1>}'
    )
    if not bill_payload:
        bill_payload = {"action_type": "billing_bid", "confidence": 0.3}
    bill_payload["action_type"] = "billing_bid"
    res2 = send_action(bill_payload)
    print(f"[STEP] step={step_num} agent=billing  bid={bill_payload.get('confidence', 0):.2f} reward={res2['reward']:.2f}")
    step_num += 1

    # Account Agent
    acc_payload = get_agent_decision(
        "account",
        user_context + '\nOutput: {"action_type": "account_bid", "confidence": <float 0-1>}'
    )
    if not acc_payload:
        acc_payload = {"action_type": "account_bid", "confidence": 0.3}
    acc_payload["action_type"] = "account_bid"
    res3 = send_action(acc_payload)
    print(f"[STEP] step={step_num} agent=account  bid={acc_payload.get('confidence', 0):.2f} reward={res3['reward']:.2f}")
    step_num += 1

    # Determine winner from observation message
    winning_msg = res3["observation"].get("message", "")
    if "technical" in winning_msg.lower():
        winner = "technical"
        win_action = "technical_execute"
    elif "billing" in winning_msg.lower():
        winner = "billing"
        win_action = "billing_execute"
    else:
        winner = "account"
        win_action = "account_execute"

    # ─────────────────────────────────────────────────────────
    # PHASE 2: EXECUTION — winning agent provides solution
    # ─────────────────────────────────────────────────────────
    exec_payload = get_agent_decision(
        winner,
        user_context + f'\nYou won the bid. Provide solution.\nOutput: {{"action_type": "{win_action}", "category": "bug|account|billing", "solution": "detailed solution text"}}'
    )
    if not exec_payload:
        exec_payload = {"action_type": win_action, "category": "bug", "solution": "Investigating the issue."}
    exec_payload["action_type"] = win_action
    res4 = send_action(exec_payload)
    print(f"[STEP] step={step_num} agent={winner}(winner) action=execute reward={res4['reward']:.2f}")
    step_num += 1

    # ─────────────────────────────────────────────────────────
    # PHASE 3: RESOLUTION — manager evaluates and closes
    # ─────────────────────────────────────────────────────────
    manager_context = (
        f"Ticket: {ticket_msg}\n"
        f"Winning agent: {winner}\n"
        f"Solution provided: {exec_payload.get('solution', 'N/A')}\n"
        'Output: {"action_type": "manager_evaluate", "should_escalate": false, "reason": "brief reason"}'
    )
    man_payload = get_agent_decision("manager", manager_context)
    if not man_payload:
        man_payload = {"action_type": "manager_evaluate", "should_escalate": False, "reason": "Solution adequate."}
    man_payload["action_type"] = "manager_evaluate"
    res5 = send_action(man_payload)
    print(f"[STEP] step={step_num} agent=manager action=evaluate escalate={man_payload.get('should_escalate')} reward={res5['reward']:.2f}")

    episode_score = res5["episode_reward"]
    print(f"\n[END] task=multi_agent_support score={episode_score:.2f} steps={step_num}")


if __name__ == "__main__":
    solve_episode()
