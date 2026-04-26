#!/usr/bin/env python3
"""
Customer Support OpenEnv - Inference Script for Hackathon
=========================================================

An AI agent orchestrator that solves tickets using the 4-Agent Bidding Protocol.
Required environment variables:
 - API_BASE_URL: HuggingFace LLM API endpoint (e.g., https://router.huggingface.co/v1)
 - MODEL_NAME:   LLM model name (e.g., meta-llama/Llama-3.2-1B-Instruct)
 - API_KEY:      HuggingFace API token

Output Format: Strict [START], [STEP], [END] text format (NOT JSON) for hackathon grader
"""

import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

if not API_KEY:
    raise ValueError("API_KEY environment variable not set. Required for hackathon submission.")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
OPENENV_SERVER = os.environ.get("OPENENV_SERVER", "http://localhost:8000")

RESET_URL = f"{OPENENV_SERVER}/reset"
STEP_URL = f"{OPENENV_SERVER}/step"

def reset_environment():
    response = requests.post(RESET_URL, json={})
    response.raise_for_status()
    return response.json().get("observation", {})


def send_action(action_payload):
    response = requests.post(STEP_URL, json={"action": action_payload})
    response.raise_for_status()
    data = response.json()
    return {
        "observation": data.get("observation", {}),
        "reward": data.get("reward", 0.0),
        "episode_reward": data.get("episode_reward", 0.0),
        "done": data.get("done", False)
    }

def get_llm_json_decision(system_prompt: str, user_prompt: str) -> dict:
    """Invokes the HF Serverless Endpoint and extracts JSON"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        
        # Cleanup markdown json blocks if LLM outputs them
        if content.startswith("```json"):
            content = content.replace("```json", "", 1).strip()
            if content.endswith("```"):
                content = content[:-3].strip()
        
        return json.loads(content)
    except Exception as e:
        # Fallback dictionary if parsing completely fails
        return {}


def solve_episode():
    # ---------------------------------------------------------
    # PRINT [START] LOG FORMAT
    # ---------------------------------------------------------
    print(f"[START] task=multi_agent_support model={MODEL_NAME}")
    
    obs = reset_environment()
    ticket_msg = obs.get("message", "")
    
    step_num = 1
    episode_score = 0.0
    
    user_context = f"Ticket: {ticket_msg}"
    
    # ---------------------------------------------------------
    # PHASE 1: BIDDING
    # ---------------------------------------------------------
    # Agent 1 (Technical)
    tech_prompt = "You are the Technical Agent. Output JSON: {\"action_type\": \"technical_bid\", \"confidence\": <float 0-1>}"
    tech_payload = get_llm_json_decision(tech_prompt, user_context)
    if not tech_payload: tech_payload = {"action_type": "technical_bid", "confidence": 0.5}
    res1 = send_action(tech_payload)
    print(f"[STEP] step={step_num} action={tech_payload.get('action_type')} reward={res1['reward']:.2f}")
    step_num += 1

    # Agent 2 (Billing)
    bill_prompt = "You are the Billing Agent. Output JSON: {\"action_type\": \"billing_bid\", \"confidence\": <float 0-1>}"
    bill_payload = get_llm_json_decision(bill_prompt, user_context)
    if not bill_payload: bill_payload = {"action_type": "billing_bid", "confidence": 0.5}
    res2 = send_action(bill_payload)
    print(f"[STEP] step={step_num} action={bill_payload.get('action_type')} reward={res2['reward']:.2f}")
    step_num += 1

    # Agent 3 (Account)
    acc_prompt = "You are the Account Agent. Output JSON: {\"action_type\": \"account_bid\", \"confidence\": <float 0-1>}"
    acc_payload = get_llm_json_decision(acc_prompt, user_context)
    if not acc_payload: acc_payload = {"action_type": "account_bid", "confidence": 0.5}
    res3 = send_action(acc_payload)
    print(f"[STEP] step={step_num} action={acc_payload.get('action_type')} reward={res3['reward']:.2f}")
    step_num += 1

    # ---------------------------------------------------------
    # PHASE 2: EXECUTION
    # ---------------------------------------------------------
    exec_prompt = "You won the bet. Output JSON: {\"action_type\": \"execute\", \"category\": \"bug|account|billing\", \"solution\": \"brief text\"}"
    exec_payload = get_llm_json_decision(exec_prompt, user_context)
    if not exec_payload: exec_payload = {"action_type": "execute", "category": "bug", "solution": "Check logs"}
    # Because of our architecture, the env expects 'winner_execute' actually, the environment handles generic "execute" from the winner.
    if exec_payload.get("action_type") == "execute":
        winner = res3["observation"].get("message", "").split(" ")[0].lower() # Hacky extract if needed
        # Or safely fallback
        exec_payload["action_type"] = "execution"
    res4 = send_action(exec_payload)
    print(f"[STEP] step={step_num} action=execution reward={res4['reward']:.2f}")
    step_num += 1

    # ---------------------------------------------------------
    # PHASE 3: RESOLUTION
    # ---------------------------------------------------------
    man_prompt = "You are the Manager. Output JSON: {\"action_type\": \"evaluate\", \"should_escalate\": false}"
    man_payload = get_llm_json_decision(man_prompt, user_context)
    if not man_payload: man_payload = {"action_type": "evaluate", "should_escalate": False}
    man_payload["action_type"] = "resolution"
    
    res5 = send_action(man_payload)
    print(f"[STEP] step={step_num} action=resolution reward={res5['reward']:.2f}")
    
    # ---------------------------------------------------------
    # PRINT [END] LOG FORMAT
    # ---------------------------------------------------------
    episode_score = res5["episode_reward"]
    print(f"[END] task=multi_agent_support score={episode_score:.2f} steps={step_num}")
    

if __name__ == "__main__":
    solve_episode()
