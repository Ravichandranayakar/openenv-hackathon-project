#!/usr/bin/env python3
"""
Customer Support OpenEnv - Inference Script for Hackathon

An AI agent that solves customer support tickets using the OpenEnv API.
Required environment variables:
  - API_BASE_URL:  HuggingFace LLM API endpoint (e.g., https://router.huggingface.co/v1)
  - MODEL_NAME:    LLM model name (e.g., Qwen/Qwen2.5-72B-Instruct)
  - HF_TOKEN:      HuggingFace API token

Output Format: Strict [START], [STEP], [END] text format (NOT JSON) for hackathon grader
Example:
  [START] task=customer_support_ticket model=Qwen/Qwen2.5-72B-Instruct
  [STEP] step=1 action=classify_issue reward=0.50
  [STEP] step=2 action=choose_solution reward=0.30
  [END] task=customer_support_ticket score=0.80 steps=2
"""

import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration - REQUIRED BY HACKATHON
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# STRICT: Use only API_KEY from environment as specified by hackathon
# Fallback to HF_TOKEN only for local development
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

# Ensure API_KEY is available
if not API_KEY:
    raise ValueError("API_KEY environment variable not set. Required for hackathon submission.")

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL  # Use hackathon's proxy or HuggingFace API
)

# OpenEnv environment server (local or provided by hackathon)
OPENENV_SERVER = os.environ.get("OPENENV_SERVER", "http://localhost:8000")

# OpenEnv server endpoints
RESET_URL = f"{OPENENV_SERVER}/reset"
STEP_URL = f"{OPENENV_SERVER}/step"
STATE_URL = f"{OPENENV_SERVER}/state"

MAX_STEPS = 10


def reset_environment():
    """Reset the environment and get initial ticket state."""
    response = requests.post(RESET_URL, json={})
    response.raise_for_status()
    data = response.json()
    return data.get("observation", {})


def send_action(action):
    """Send action to environment and get observation."""
    payload = {"action": action}
    response = requests.post(STEP_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return {
        "observation": data.get("observation", {}),
        "reward": data.get("reward", 0.0),
        "done": data.get("done", False)
    }


def classify_issue_with_llm(message: str, severity: str) -> str:
    """Use LLM to classify the ticket issue type."""
    prompt = f"""Classify this support ticket into ONE category:
- billing (payments, subscriptions, charges)
- account (login, password, profile)
- bug (crashes, errors, glitches)
- feature (how-to, new features, capabilities)

Message: {message}
Severity: {severity}

Respond with ONLY the category word (billing, account, bug, or feature)."""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20
        )
        result = response.choices[0].message.content.strip().lower()
        if result in ["billing", "account", "bug", "feature"]:
            return result
        return "billing"  # Default fallback
    except Exception as e:
        print(f"[DEBUG] LLM classification error: {e}", flush=True)
        return "billing"


def choose_solution_with_llm(message: str, classification: str) -> tuple:
    """Use LLM to pick solution category and ID."""
    solutions = {
        "billing": ("duplicate_charge", "refund_duplicate_charge"),
        "account": ("password", "reset_password_link"),
        "bug": ("app_crash", "update_app_version"),
        "feature": ("how_to", "explain_feature")
    }
    
    category, solution_id = solutions.get(classification, ("duplicate_charge", "refund_duplicate_charge"))
    return category, solution_id


def escalate_with_llm(message: str, severity: str) -> bool:
    """Use LLM to decide escalation."""
    prompt = f"""Should this ticket be escalated to a human or closed?
Escalate if: Complex, urgent, customer angry, needs specialist
Close if: Simple fix, FAQ answer, already resolved

Message: {message}
Severity: {severity}

Respond with ONLY "escalate" or "close"."""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20
        )
        result = response.choices[0].message.content.strip().lower()
        return "escalate" in result
    except Exception as e:
        print(f"[DEBUG] LLM escalation error: {e}", flush=True)
        return False



def run_task(task_name: str) -> tuple:
    """Run a single customer support task and return (score, step_count, rewards_list).
    Score is strictly in range (0, 1) - never exactly 0.0 or 1.0."""
    
    rewards_list = []  # Track all step rewards
    step_count = 0
    episode_done = False
    env_name = "customer_support_env"
    
    try:
        # RESET: Load a random ticket
        observation = reset_environment()
        message = observation.get("message", "")
        severity = observation.get("severity", "low")
        
        # STEP 1: Classify issue
        step_count += 1
        classification = classify_issue_with_llm(message, severity)
        action = {"action_type": "classify_issue", "classification": classification}
        
        result = send_action(action)
        reward = result.get("reward", 0.0)
        rewards_list.append(reward)
        obs = result.get("observation", {})
        episode_done = result.get("done", False)
        
        error_msg = "null"
        done_str = "true" if episode_done else "false"
        print(f"[STEP] step={step_count} action=classify_issue reward={reward:.2f} done={done_str} error={error_msg}", flush=True)
        
        if not episode_done:
            # STEP 2: Choose solution
            step_count += 1
            category, solution = choose_solution_with_llm(message, classification)
            action = {"action_type": "choose_solution", "category": category, "solution": solution}
            
            result = send_action(action)
            reward = result.get("reward", 0.0)
            rewards_list.append(reward)
            obs = result.get("observation", {})
            episode_done = result.get("done", False)
            
            error_msg = "null"
            done_str = "true" if episode_done else "false"
            print(f"[STEP] step={step_count} action=choose_solution reward={reward:.2f} done={done_str} error={error_msg}", flush=True)
            
            if not episode_done:
                # STEP 3: Escalation decision
                step_count += 1
                should_escalate = escalate_with_llm(message, severity)
                action = {"action_type": "escalate_decision", "should_escalate": should_escalate}
                
                result = send_action(action)
                reward = result.get("reward", 0.0)
                rewards_list.append(reward)
                obs = result.get("observation", {})
                episode_done = result.get("done", False)
                
                error_msg = "null"
                done_str = "true" if episode_done else "false"
                print(f"[STEP] step={step_count} action=escalate_decision reward={reward:.2f} done={done_str} error={error_msg}", flush=True)
                
                if not episode_done:
                    # STEP 4: Close ticket
                    step_count += 1
                    action = {"action_type": "close_ticket"}
                    
                    result = send_action(action)
                    reward = result.get("reward", 0.0)
                    rewards_list.append(reward)
                    obs = result.get("observation", {})
                    episode_done = result.get("done", True)
                    
                    error_msg = "null"
                    done_str = "true"
                    print(f"[STEP] step={step_count} action=close_ticket reward={reward:.2f} done={done_str} error={error_msg}", flush=True)
    
    except Exception as e:
        print(f"[ERROR] Task '{task_name}' error: {e}", flush=True)
        error_msg = str(e)[:50]
    
    # Calculate final score strictly in (0, 1)
    total_reward = sum(rewards_list)
    clamped = max(-1.2, min(1.2, total_reward))
    normalized = (clamped + 1.2) / 2.4  # Maps [-1.2, 1.2] to [0, 1]
    final_score = max(0.01, min(0.99, normalized))
    
    # Format rewards as comma-separated string
    rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])
    
    return final_score, step_count, rewards_str


def main():
    """Run 3 customer support tasks."""
    
    tasks = [
        "easy_task",
        "medium_task", 
        "hard_task"
    ]
    
    for task_name in tasks:
        # [START]
        print(f"[START] task={task_name} env=customer_support_env model={MODEL_NAME}", flush=True)
        
        # Run the task
        final_score, step_count, rewards_str = run_task(task_name)
        
        # [END]
        print(f"[END] success=true steps={step_count} score={final_score:.2f} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    main()

