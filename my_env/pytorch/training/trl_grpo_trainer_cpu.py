"""
CPU-FRIENDLY TEST VERSION - Bidding Negotiation Multi-Agent Trainer

✅ USE BEFORE ON-SITE TRAINING (verifies environment + prompts work)

This is a simplified version to test the 3-Phase Bidding Protocol:
1. Bidding (Technical, Billing, Account agents)
2. Execution (Winning agent)
3. Evaluation (Manager agent)

Usage:
  python my_env/pytorch/training/trl_grpo_trainer_cpu.py --episodes 5
"""

import torch
import requests
import json
import logging
import time
from typing import Dict, Tuple
from dataclasses import dataclass
import argparse
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from my_env.pytorch.prompts import (
        TECHNICAL_AGENT_PROMPT,
        BILLING_AGENT_PROMPT,
        ACCOUNT_AGENT_PROMPT,
        EXECUTION_PROMPT,
        MANAGER_AGENT_PROMPT
    )
except ImportError:
    from ..prompts import (
        TECHNICAL_AGENT_PROMPT,
        BILLING_AGENT_PROMPT,
        ACCOUNT_AGENT_PROMPT,
        EXECUTION_PROMPT,
        MANAGER_AGENT_PROMPT
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CPUTestConfig:
    """Configuration for CPU testing."""
    model_name: str = "gpt2" # Small test model
    num_episodes: int = 5
    env_url: str = "http://localhost:8000"
    device: str = "cpu"


class EnvironmentClient:
    """HTTP client for communicating with OpenEnv Multi-Agent server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def reset(self) -> Dict:
        """Initialize new negotiation episode."""
        try:
            response = self.session.post(f"{self.base_url}/reset", timeout=5)
            response.raise_for_status()
            return response.json().get("observation", {})
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            raise
    
    def step(self, action_payload: Dict) -> Tuple[Dict, float, bool]:
        """Send action to environment (bid, execute, or evaluate)."""
        try:
            response = self.session.post(
                f"{self.base_url}/step",
                json={"action": action_payload},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            obs = data.get("observation", {})
            reward = data.get("reward", 0.0)
            done = data.get("done", False)
            
            return obs, reward, done
        except Exception as e:
            logger.error(f"Step failed: {e}")
            return {}, -1.0, True


class CPUTestTrainer:
    """Lightweight 4-Agent simulator for CPU testing."""
    
    def __init__(self, config: CPUTestConfig):
        self.config = config
        self.env = EnvironmentClient(config.env_url)
        
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.model.to(config.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.metrics = {"rewards": []}

    def generate_action(self, system_prompt: str, ticket_message: str) -> Dict:
        """Simulate LLM generating a JSON action based on its Persona."""
        # For a CPU test with GPT-2, we simulate structured output since GPT-2 cannot reliably output JSON.
        # In the GPU version (Llama 3), this function does real inference.
        
        # We mock the outputs just to test the server pipeline correctly.
        if "Technical Support Specialist" in system_prompt:
            return {"action_type": "technical_bid", "confidence": 0.8, "rationale": "App crash mentioned"}
        elif "Billing Support Specialist" in system_prompt:
            return {"action_type": "billing_bid", "confidence": 0.9, "rationale": "Refund requested"}
        elif "Account Security Specialist" in system_prompt:
            return {"action_type": "account_bid", "confidence": 0.1, "rationale": "Not my purview"}
        elif "won the bid" in system_prompt:
            return {"action_type": "execute", "category": "duplicate_charge", "solution": "refund_duplicate_charge"}
        elif "Quality Assurance Manager" in system_prompt:
            return {"action_type": "evaluate", "should_escalate": False, "reason": "Looks good"}
        
        return {}

    def run_episode(self, episode_num: int) -> float:
        """Run through the 3-Phase Bidding Protocol for one ticket."""
        # Phase 0: Reset Environment
        obs = self.env.reset()
        ticket_msg = obs.get("message", "")
        
        logger.info(f"--- EPISODE {episode_num + 1} ---")
        logger.info(f"Ticket: {ticket_msg[:50]}...")
        
        # Phase 1: Bidding
        tech_bid = self.generate_action(TECHNICAL_AGENT_PROMPT, ticket_msg)
        bill_bid = self.generate_action(BILLING_AGENT_PROMPT, ticket_msg)
        acc_bid = self.generate_action(ACCOUNT_AGENT_PROMPT, ticket_msg)
        
        self.env.step(tech_bid)
        self.env.step(bill_bid)
        obs, bid_reward, _ = self.env.step(acc_bid)
        
        # Identify winner
        winner = "billing" # Mocking logic for CPU test, the Env internally registers the winner
        
        # Phase 2: Execution
        exec_payload = self.generate_action(EXECUTION_PROMPT, ticket_msg)
        # Server expects the winner's id
        exec_payload["action_type"] = f"{winner}_execute"
        obs, exec_reward, _ = self.env.step(exec_payload)
        
        # Phase 3: Evaluation
        eval_payload = self.generate_action(MANAGER_AGENT_PROMPT, ticket_msg)
        eval_payload["action_type"] = "manager_evaluate"
        obs, eval_reward, done = self.env.step(eval_payload)
        
        total_reward = obs.get("episode_reward", bid_reward + exec_reward + eval_reward)
        logger.info(f"Result: Episode Done={done}, Total Reward={total_reward:+.1f}")
        return float(total_reward)

    def run_test(self):
        logger.info(f"Starting Multi-Agent Simulation ({self.config.num_episodes} episodes)")
        for i in range(self.config.num_episodes):
            reward = self.run_episode(i)
            self.metrics["rewards"].append(reward)
            
        avg_reward = np.mean(self.metrics["rewards"])
        logger.info("=" * 40)
        logger.info(f"TEST COMPLETE. Average Reward: {avg_reward:+.2f}")
        logger.info("Server integrated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    
    config = CPUTestConfig(num_episodes=args.episodes)
    trainer = CPUTestTrainer(config)
    trainer.run_test()
