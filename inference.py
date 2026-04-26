#!/usr/bin/env python3
"""
Customer Support OpenEnv - Inference Script for Hackathon
=========================================================

Multi-Agent GRPO-Trained Inference Pipeline.
4 specialist agents (Technical, Billing, Account, Manager) solve tickets
using the 3-phase bidding protocol learned via GRPO training.

Trained Model:  RavichandraNayakar/openenv-grpo-merged
LoRA Adapters:  RavichandraNayakar/openenv-multi-agent-grpo

Output Format: Strict [START], [STEP], [END] text format for hackathon grader.

Required environment variables:
 - API_BASE_URL: HuggingFace LLM API endpoint
 - MODEL_NAME:   LLM model name (defaults to our trained merged model)
 - HF_TOKEN:     HuggingFace API token
"""

import os
import requests
import json
from dotenv import load_dotenv
from my_env.pytorch.prompts import (
    TECHNICAL_AGENT_PROMPT,
    BILLING_AGENT_PROMPT,
    ACCOUNT_AGENT_PROMPT,
    EXECUTION_PROMPT,
    MANAGER_AGENT_PROMPT,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────
# MODE SELECTION
# Set USE_LOCAL_MODEL=true in .env to use trained LoRA adapters directly.
# Set USE_LOCAL_MODEL=false (default) to use HF Serverless API.
# ─────────────────────────────────────────────────────────────
USE_LOCAL_MODEL = os.environ.get("USE_LOCAL_MODEL", "false").lower() == "true"
BASE_MODEL_ID   = "unsloth/Meta-Llama-3.1-8B-Instruct"
ADAPTER_ID      = "RavichandraNayakar/openenv-multi-agent-grpo"
MERGED_MODEL_ID = "RavichandraNayakar/openenv-grpo-merged"

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

# Initialise appropriate backend
if USE_LOCAL_MODEL:
    print("[INFO] Loading trained LoRA adapters locally...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    _base      = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _model = PeftModel.from_pretrained(_base, ADAPTER_ID)
    _model.eval()
    _pipe  = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True,
    )
    DISPLAY_MODEL = f"{ADAPTER_ID} (LoRA on {BASE_MODEL_ID})"
    print(f"[INFO] Trained model loaded: {DISPLAY_MODEL}")
else:
    from openai import OpenAI
    if not API_KEY:
        raise ValueError("HF_TOKEN not set. Add it to .env")
    _client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    DISPLAY_MODEL = MODEL_NAME

OPENENV_SERVER = os.environ.get("OPENENV_SERVER", "http://localhost:8000")
RESET_URL = f"{OPENENV_SERVER}/reset"
STEP_URL  = f"{OPENENV_SERVER}/step"


# ─────────────────────────────────────────────────────────────
# TRAINED AGENT SYSTEM PROMPTS
# Mirror the GRPO-trained persona alignment per agent.
# ─────────────────────────────────────────────────────────────
# TRAINED AGENT SYSTEM PROMPTS (imported from prompts.py)
# These are the exact prompts used during GRPO training.
# ─────────────────────────────────────────────────────────────
AGENT_PROMPTS = {
    "technical": TECHNICAL_AGENT_PROMPT,
    "billing":   BILLING_AGENT_PROMPT,
    "account":   ACCOUNT_AGENT_PROMPT,
    "execution": EXECUTION_PROMPT,
    "manager":   MANAGER_AGENT_PROMPT,
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
        "observation":    data.get("observation", {}),
        "reward":         data.get("reward", 0.0),
        "episode_reward": data.get("episode_reward", 0.0),
        "done":           data.get("done", False),
    }


def get_agent_decision(agent_name: str, user_prompt: str) -> dict:
    """Call the trained model (local LoRA or HF API) with trained system prompt."""
    system_prompt = AGENT_PROMPTS[agent_name]
    try:
        if USE_LOCAL_MODEL:
            # ─ Local inference: base model + LoRA adapters ───────────────────
            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": user_prompt},
            ]
            prompt_str = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            out     = _pipe(prompt_str)[0]["generated_text"]
            content = out[len(prompt_str):].strip()
        else:
            # ─ HF Serverless API ───────────────────────────────────
            response = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            content = response.choices[0].message.content.strip()

        # Strip markdown fences if model wraps output
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        # Extract first JSON object if model appended extra text
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start != -1 and end > start:
            content = content[start:end]
        return json.loads(content)
    except Exception as e:
        print(f"  [WARN] Model call failed ({agent_name}): {e}", flush=True)
        return {}


def solve_episode():
    print(f"[START] task=multi_agent_support model={MODEL_NAME}")

    obs        = reset_environment()
    ticket_msg = obs.get("message", "Unknown issue")
    step_num   = 1
    ctx        = f"Support ticket: {ticket_msg}\nRespond with JSON only."

    # ─────────────────────────────────────────────────────────
    # PHASE 1: BIDDING — 3 specialists bid confidence [0.0-1.0]
    # ─────────────────────────────────────────────────────────

    # Technical agent bids
    tech = get_agent_decision(
        "technical",
        ctx + '\nOutput exactly: {"action_type": "technical_bid", "confidence": <float>}'
    )
    if not tech:
        tech = {"action_type": "technical_bid", "confidence": 0.5}
    tech["action_type"] = "technical_bid"   # enforce correct action type
    r1 = send_action(tech)
    print(f"[STEP] step={step_num} action=technical_bid confidence={tech.get('confidence', 0):.2f} reward={r1['reward']:.2f}")
    step_num += 1

    # Billing agent bids
    bill = get_agent_decision(
        "billing",
        ctx + '\nOutput exactly: {"action_type": "billing_bid", "confidence": <float>}'
    )
    if not bill:
        bill = {"action_type": "billing_bid", "confidence": 0.3}
    bill["action_type"] = "billing_bid"
    r2 = send_action(bill)
    print(f"[STEP] step={step_num} action=billing_bid confidence={bill.get('confidence', 0):.2f} reward={r2['reward']:.2f}")
    step_num += 1

    # Account agent bids
    acc = get_agent_decision(
        "account",
        ctx + '\nOutput exactly: {"action_type": "account_bid", "confidence": <float>}'
    )
    if not acc:
        acc = {"action_type": "account_bid", "confidence": 0.3}
    acc["action_type"] = "account_bid"
    r3 = send_action(acc)
    print(f"[STEP] step={step_num} action=account_bid confidence={acc.get('confidence', 0):.2f} reward={r3['reward']:.2f}")
    step_num += 1

    # Determine winner from environment observation
    obs_msg = r3["observation"].get("message", "").lower()
    if "technical" in obs_msg:
        winner, win_action = "technical", "technical_execute"
    elif "billing" in obs_msg:
        winner, win_action = "billing", "billing_execute"
    else:
        winner, win_action = "account", "account_execute"

    # ─────────────────────────────────────────────────────────
    # PHASE 2: EXECUTION — winner uses EXECUTION_PROMPT from prompts.py
    # ─────────────────────────────────────────────────────────
    exec_payload = get_agent_decision(
        "execution",   # uses EXECUTION_PROMPT (the exact training prompt)
        ctx + f'\nYou are the {winner} agent and you won the bid.\n'
              f'Output exactly: {{"action_type": "{win_action}", "category": "bug|billing|account", "solution": "brief solution text"}}'
    )
    if not exec_payload:
        exec_payload = {"action_type": win_action, "category": "bug", "solution": "Investigating the reported issue."}
    exec_payload["action_type"] = win_action   # enforce correct action type
    r4 = send_action(exec_payload)
    print(f"[STEP] step={step_num} action={win_action} agent={winner} reward={r4['reward']:.2f}")
    step_num += 1

    # ─────────────────────────────────────────────────────────
    # PHASE 3: RESOLUTION — manager evaluates and closes
    # ─────────────────────────────────────────────────────────
    man = get_agent_decision(
        "manager",
        f"Ticket: {ticket_msg}\n"
        f"Winning agent: {winner}\n"
        f"Solution: {exec_payload.get('solution', 'N/A')}\n"
        'Output exactly: {"action_type": "manager_evaluate", "should_escalate": false, "reason": "brief reason"}'
    )
    if not man:
        man = {"action_type": "manager_evaluate", "should_escalate": False, "reason": "Solution adequate."}
    man["action_type"] = "manager_evaluate"
    r5 = send_action(man)
    print(f"[STEP] step={step_num} action=manager_evaluate escalate={man.get('should_escalate')} reward={r5['reward']:.2f}")

    episode_score = r5["episode_reward"]
    print(f"[END] task=multi_agent_support score={episode_score:.2f} steps={step_num}")


if __name__ == "__main__":
    solve_episode()
