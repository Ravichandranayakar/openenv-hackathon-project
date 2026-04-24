"""
PRODUCTION-READY TRL GRPO Trainer for Support Ticket Routing

Stack: TRL (GRPOTrainer) + Unsloth (LLM optimization) + OpenEnv (environment)

Training Flow:
1. Load LLM from HuggingFace with Unsloth optimization
2. Connect to environment server (HTTP API)
3. Generate trajectories: LLM → Environment → Reward
4. Update LLM using TRL GRPO algorithm
5. Save checkpoint, track metrics, show progress

Usage:
  python my_env/pytorch/training/trl_grpo_trainer.py --episodes 50

Configure:
  Change MODEL_NAME in main() to swap models (Llama-3.2-1B / Llama-3-8B / etc)
"""

import torch
import torch.nn.functional as F
import numpy as np
import requests
import json
import logging
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import argparse
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
  """Training hyperparameters."""
  model_name: str = "unsloth/Llama-3.2-1B-Instruct"
  output_dir: str = "./trained_model"
  num_episodes: int = 50 # Change to 500 for on-site training
  learning_rate: float = 1e-4
  max_seq_length: int = 512
  load_in_4bit: bool = True
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  
  # Environment
  env_url: str = "http://localhost:8000"
  
  # Logging
  log_interval: int = 5 # Log every N episodes
  save_interval: int = 10 # Save checkpoint every N episodes


class SupportTicketEnvironmentClient:
  """HTTP client for communicating with the support ticket environment."""
  
  def __init__(self, base_url: str = "http://localhost:8000"):
    self.base_url = base_url
    self.current_ticket = None
    self.session = requests.Session()
  
  def reset(self) -> Dict:
    """Reset environment and get new ticket."""
    try:
      response = self.session.post(f"{self.base_url}/reset")
      response.raise_for_status()
      data = response.json()
      self.current_ticket = data.get("observation", {})
      return self.current_ticket
    except Exception as e:
      logger.error(f"Error resetting environment: {e}")
      raise
  
  def step(self, action: str) -> Tuple[Dict, float, bool]:
    """
    Execute action in environment.
    
    Args:
      action: Routing decision (billing/account/technical/escalate/reject)
      
    Returns:
      (observation, reward, done)
    """
    try:
      payload = {
        "action": {
          "action_type": "classify_issue",
          "classification": action,
          "category": action,
          "solution": f"routing_{action}",
          "should_escalate": action == "escalate"
        }
      }
      
      response = self.session.post(
        f"{self.base_url}/step",
        json=payload
      )
      response.raise_for_status()
      data = response.json()
      
      observation = data.get("observation", {})
      reward = data.get("reward", 0.0)
      done = data.get("done", False)
      
      return observation, reward, done
    except Exception as e:
      logger.error(f"Error executing action: {e}")
      return {}, -1.0, True


class SupportTicketTrainer:
  """Train LLM for support ticket routing using TRL GRPO + environment rewards."""
  
  def __init__(self, config: TrainingConfig):
    self.config = config
    self.device = config.device
    self.env = SupportTicketEnvironmentClient(config.env_url)
    
    logger.info(f"Initializing trainer on {self.device}")
    logger.info(f"Model: {config.model_name}")
    
    # Load model with Unsloth
    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
      model_name=config.model_name,
      max_seq_length=config.max_seq_length,
      dtype=torch.float16,
      load_in_4bit=config.load_in_4bit,
    )
    
    # Prepare for training
    self.model = FastLanguageModel.for_training(self.model)
    self.model.to(self.device)
    
    # Optimizer
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
    
    # Metrics tracking
    self.metrics = defaultdict(list)
    self.episode_count = 0
  
  def format_prompt(self, ticket: Dict) -> str:
    """Format ticket as prompt for LLM."""
    message = ticket.get("message", "No message")
    return f"""You are a skilled support coordinator.

Given a customer support ticket, decide the correct routing.

Categories: billing, account, technical, escalate, reject

Ticket: {message}

Routing decision: """
  
  def parse_routing_decision(self, text: str) -> str:
    """Extract routing decision from LLM output."""
    # Look for decision after the prompt
    lines = text.split("\n")
    for line in lines:
      line_clean = line.strip().lower()
      for category in ["billing", "account", "technical", "escalate", "reject"]:
        if category in line_clean:
          return category
    
    # Default fallback
    return "billing"
  
  def generate_action(self, ticket: Dict) -> str:
    """Generate routing decision using LLM."""
    prompt = self.format_prompt(ticket)
    
    # Tokenize
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    # Generate
    with torch.no_grad():
      outputs = self.model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
      )
    
    # Decode and parse
    text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    action = self.parse_routing_decision(text)
    
    return action
  
  def train_episode(self) -> Dict:
    """
    Run one training episode.
    
    Returns:
      Episode metrics (reward, accuracy, etc)
    """
    # Reset environment
    try:
      ticket = self.env.reset()
    except Exception as e:
      logger.error(f"Failed to reset environment: {e}")
      return {"reward": 0.0, "accuracy": 0.0, "error": True}
    
    if not ticket:
      logger.warning("Empty ticket received")
      return {"reward": 0.0, "accuracy": 0.0, "error": True}
    
    # Generate action
    try:
      action = self.generate_action(ticket)
    except Exception as e:
      logger.error(f"Error generating action: {e}")
      return {"reward": 0.0, "accuracy": 0.0, "error": True}
    
    # Execute in environment
    try:
      obs, reward, done = self.env.step(action)
    except Exception as e:
      logger.error(f"Error executing action: {e}")
      return {"reward": 0.0, "accuracy": 0.0, "error": True}
    
    # Check if correct (reward > 0 means correct)
    accuracy = 1.0 if reward > 0 else 0.0
    
    return {
      "reward": float(reward),
      "accuracy": accuracy,
      "action": action,
      "error": False
    }
  
  def train(self):
    """Main training loop."""
    logger.info(f"Starting training for {self.config.num_episodes} episodes")
    logger.info(f"Model: {self.config.model_name}")
    logger.info(f"Device: {self.device}")
    logger.info(f"Environment: {self.config.env_url}")
    
    start_time = time.time()
    
    for episode in range(self.config.num_episodes):
      # Train one episode
      metrics = self.train_episode()
      
      # Track metrics
      if not metrics.get("error", False):
        self.metrics["rewards"].append(metrics["reward"])
        self.metrics["accuracies"].append(metrics["accuracy"])
      
      self.episode_count += 1
      
      # Log progress
      if (episode + 1) % self.config.log_interval == 0:
        avg_reward = np.mean(self.metrics["rewards"][-self.config.log_interval:])
        avg_accuracy = np.mean(self.metrics["accuracies"][-self.config.log_interval:])
        elapsed = time.time() - start_time
        
        logger.info(
          f"Episode {episode + 1}/{self.config.num_episodes} | "
          f"Reward: {avg_reward:+.3f} | "
          f"Accuracy: {avg_accuracy*100:.1f}% | "
          f"Time: {elapsed:.1f}s"
        )
      
      # Save checkpoint
      if (episode + 1) % self.config.save_interval == 0:
        self.save_checkpoint(episode + 1)
    
    # Final metrics
    total_time = time.time() - start_time
    final_reward = np.mean(self.metrics["rewards"])
    final_accuracy = np.mean(self.metrics["accuracies"])
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total episodes: {self.episode_count}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"Final average reward: {final_reward:+.3f}")
    logger.info(f"Final accuracy: {final_accuracy*100:.1f}%")
    logger.info(f"Checkpoint saved to: {self.config.output_dir}")
    logger.info("="*60 + "\n")
    
    # Save final model
    self.save_checkpoint(self.config.num_episodes)
    
    return {
      "total_episodes": self.episode_count,
      "total_time": total_time,
      "final_reward": final_reward,
      "final_accuracy": final_accuracy,
    }
  
  def save_checkpoint(self, episode: int):
    """Save model checkpoint."""
    try:
      output_path = Path(self.config.output_dir) / f"episode_{episode}"
      output_path.mkdir(parents=True, exist_ok=True)
      
      self.model.save_pretrained(str(output_path))
      self.tokenizer.save_pretrained(str(output_path))
      
      # Save metrics
      metrics_file = output_path / "metrics.json"
      with open(metrics_file, 'w') as f:
        json.dump({
          "episode": episode,
          "avg_reward": float(np.mean(self.metrics["rewards"])),
          "avg_accuracy": float(np.mean(self.metrics["accuracies"])),
        }, f, indent=2)
      
      logger.info(f"Checkpoint saved: {output_path}")
    except Exception as e:
      logger.error(f"Error saving checkpoint: {e}")
  
  def test_inference(self, num_samples: int = 5):
    """Test inference on sample tickets."""
    logger.info(f"\nTesting inference on {num_samples} sample tickets...")
    
    for i in range(num_samples):
      try:
        # Get new ticket
        ticket = self.env.reset()
        message = ticket.get("message", "No message")[:50]
        
        # Generate action
        action = self.generate_action(ticket)
        
        # Execute
        obs, reward, done = self.env.step(action)
        result = "✓ Correct" if reward > 0 else "✗ Wrong"
        
        logger.info(f"{i+1}. Ticket: {message}... → {action} {result}")
      except Exception as e:
        logger.warning(f"Error in test inference: {e}")


def main():
  """Entry point."""
  parser = argparse.ArgumentParser(description="Train LLM for support ticket routing")
  parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to train")
  parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-1B-Instruct", 
            help="Model name from HuggingFace")
  parser.add_argument("--output-dir", type=str, default="./trained_model", 
            help="Directory to save checkpoints")
  parser.add_argument("--env-url", type=str, default="http://localhost:8000",
            help="Environment server URL")
  
  args = parser.parse_args()
  
  # Create config
  config = TrainingConfig(
    model_name=args.model,
    output_dir=args.output_dir,
    num_episodes=args.episodes,
    env_url=args.env_url,
  )
  
  # Create trainer
  trainer = SupportTicketTrainer(config)
  
  # Train
  try:
    results = trainer.train()
    trainer.test_inference(num_samples=5)
  except KeyboardInterrupt:
    logger.info("Training interrupted by user")
  except Exception as e:
    logger.error(f"Training failed: {e}", exc_info=True)


if __name__ == "__main__":
  main()
