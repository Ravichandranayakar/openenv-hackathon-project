#!/usr/bin/env python3
"""
Customer Support OpenEnv - Inference Script for Hackathon

Inference script for the OpenEnv hackathon submission.
Demonstrates an agent learning to handle customer support tickets.

Environment Variables (REQUIRED):
  - API_BASE_URL   The API endpoint for the LLM
  - MODEL_NAME     The model identifier to use for inference
  - HF_TOKEN       Your Hugging Face / API key

Usage:
  python inference.py

Output Format (MANDATORY):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys
import textwrap
from typing import List, Optional

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "customer-support-agent")
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_KEY = HF_TOKEN or os.getenv("API_KEY", "")

# Constants
TASK_NAME = "customer_support"
BENCHMARK = "customer_support_env"
MAX_STEPS = 10
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.6
TEMPERATURE = 0.7
MAX_TOKENS = 256

# Ensure my_env is importable
sys.path.insert(0, os.path.dirname(__file__))

try:
    from my_env import CustomerSupportEnvironment, SupportAction
    from my_env.agents import CurriculumLearningAgent
except ImportError:
    from my_env.server.customer_support_environment import CustomerSupportEnvironment
    from my_env.models import SupportAction
    from my_env.agents import CurriculumLearningAgent


# ============================================================================
# HACKATHON-COMPLIANT LOGGING (EXACT FORMAT REQUIRED)
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """
    Log episode start.
    Format: [START] task=<task> env=<env> model=<model>
    """
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """
    Log a single step.
    Format: [STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    """
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    # Escape quotes in action for proper formatting
    action_escaped = action.replace('"', '\\"')
    print(
        f'[STEP] step={step} action="{action_escaped}" reward={reward:.2f} done={done_str} error={error_str}',
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    rewards: List[float],
) -> None:
    """
    Log episode end.
    Format: [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
    """
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

def main() -> None:
    """
    Main inference loop demonstrating agent interaction with environment.
    
    Flow:
    1. Initialize environment and agent
    2. Reset environment → get initial observation
    3. For each step:
       - Agent selects action based on observation
       - Send action to environment via step()
       - Receive observation, reward, done
       - Log step
    4. Calculate final score and log end
    """
    
    # Initialize
    env = CustomerSupportEnvironment()
    agent = CurriculumLearningAgent()
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    # Log start
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset environment - returns SupportObservation directly
        observation = env.reset()
        
        # Convert Pydantic model to dict if needed
        if not isinstance(observation, dict):
            if hasattr(observation, 'model_dump'):
                observation = observation.model_dump()
            elif hasattr(observation, '__dict__'):
                observation = observation.__dict__
        
        last_message = observation.get("message", "")
        last_reward = 0.0
        
        # Main loop
        for step in range(1, MAX_STEPS + 1):
            # Check if done
            done = observation.get("done", False)
            if done:
                break
            
            # Agent selects action based on message
            message = observation.get("message", "")
            
            # Agent uses learned patterns to decide action
            action_type = agent._select_action(
                message,
                ["classify_issue", "choose_solution", "escalate_decision", "close_ticket"]
            )
            
            # Build action description for logging
            severity = observation.get("severity", "low")
            action_str = f"{action_type}: {message[:40]}... ({severity})"
            
            # Create structured action for environment
            try:
                if "classify" in action_type:
                    action = SupportAction(
                        action_type="classify_issue",
                        classification="billing",
                    )
                elif "escalate" in action_type:
                    action = SupportAction(
                        action_type="escalate_decision",
                        should_escalate=True,
                    )
                elif "close" in action_type:
                    action = SupportAction(
                        action_type="close_ticket",
                    )
                else:
                    action = SupportAction(
                        action_type="choose_solution",
                        category="general",
                        solution="contact_support",
                    )
            except Exception as e:
                print(f"[DEBUG] Action construction failed: {e}", flush=True)
                action = SupportAction(action_type="close_ticket")
            
            # Step environment
            error_msg = None
            try:
                result_obs = env.step(action)
                
                # Convert result to dict
                if not isinstance(result_obs, dict):
                    if hasattr(result_obs, 'model_dump'):
                        result_obs = result_obs.model_dump()
                    elif hasattr(result_obs, '__dict__'):
                        result_obs = result_obs.__dict__
                
                reward = result_obs.get("reward", 0.0)
                done = result_obs.get("done", False)
            except Exception as e:
                print(f"[DEBUG] Environment step failed: {e}", flush=True)
                reward = 0.0
                done = True
                error_msg = str(e)
                result_obs = observation
            
            # Track results
            rewards.append(reward)
            steps_taken = step
            last_message = result_obs.get("message", last_message)
            last_reward = reward
            observation = result_obs
            
            # Log step (HACKATHON FORMAT)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
            
            # Add to history
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")
            
            if done:
                break
        
        # Calculate final score
        total_reward = sum(rewards)
        success = total_reward >= SUCCESS_SCORE_THRESHOLD
    
    except Exception as e:
        print(f"[DEBUG] Inference failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        success = False
    
    finally:
        # Cleanup
        try:
            if hasattr(env, 'close'):
                env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        
        # Log end (HACKATHON FORMAT)
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
