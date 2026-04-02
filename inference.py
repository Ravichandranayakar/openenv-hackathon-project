#!/usr/bin/env python3
"""
Customer Support OpenEnv - Inference Script for Hackathon

Uses OpenAI client to make decisions for customer support ticket handling.

Environment Variables (REQUIRED):
  - API_BASE_URL   The API endpoint for the LLM
  - MODEL_NAME     The model identifier to use for inference
  - HF_TOKEN       Your Hugging Face / API key

Usage:
  python inference.py
"""

import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "http://localhost:8000"),
    api_key=os.environ.get("HF_TOKEN", "")
)

# Constants
MAX_STEPS = 10
SERVER_URL = "http://localhost:8000"


def reset_environment():
    """Reset the environment and get initial state."""
    response = requests.post(f"{SERVER_URL}/reset")
    response.raise_for_status()
    return response.json()


def get_state():
    """Get current environment state."""
    response = requests.get(f"{SERVER_URL}/state")
    response.raise_for_status()
    return response.json()


def send_action(action):
    """Send action to environment and get observation."""
    payload = {"action": action}
    response = requests.post(
        f"{SERVER_URL}/step",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()


def get_llm_action(state):
    """Use LLM to decide next action based on current state."""
    state_str = json.dumps(state, indent=2)
    
    prompt = f"""
You are an AI agent handling customer support tickets.

Current ticket state:
{state_str}

Based on the ticket information, decide what action to take. You MUST respond with a JSON object in one of these exact formats:

For classifying issue:
{{"action_type": "classify_issue", "classification": "billing|account|bug|feature"}}

For choosing solution:
{{"action_type": "choose_solution", "category": "category_name", "solution": "solution_id"}}

For escalation decision:
{{"action_type": "escalate_decision", "should_escalate": true|false}}

For closing ticket:
{{"action_type": "close_ticket"}}

Respond with ONLY the JSON object, no other text.
"""
    
    response = client.chat.completions.create(
        model=os.environ.get("MODEL_NAME", "customer-support-agent"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256
    )
    
    try:
        action_text = response.choices[0].message.content.strip()
        action = json.loads(action_text)
        return action
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"[DEBUG] Failed to parse LLM response: {e}", flush=True)
        return {"action_type": "close_ticket"}


def main():
    """Main inference loop using OpenAI client."""
    
    print("[START] Episode initialized", flush=True)
    
    total_reward = 0.0
    step_count = 0
    
    try:
        # Reset environment
        initial_state = reset_environment()
        
        # Main loop
        for step in range(1, MAX_STEPS + 1):
            # Get current state
            state = get_state()
            
            # Check if done
            done = state.get("done", False)
            if done:
                break
            
            # Get action from LLM
            action = get_llm_action(state)
            
            # Send action to environment
            observation = send_action(action)
            
            # Extract reward
            reward = observation.get("reward", 0.0)
            done = observation.get("done", False)
            
            # Track
            total_reward += reward
            step_count = step
            
            # Log step - EXACT FORMAT REQUIRED
            action_str = json.dumps(action)
            print(f"[STEP] Action taken: {action_str} | Reward: {reward:.2f}", flush=True)
            
            if done:
                break
        
    except Exception as e:
        print(f"[DEBUG] Error during inference: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    finally:
        # Log end - EXACT FORMAT REQUIRED
        score = min(total_reward, 1.0)  # Normalize to 0.0-1.0
        print(f"[END] Episode finished | Grader Score: {score:.2f}", flush=True)


if __name__ == "__main__":
    main()
