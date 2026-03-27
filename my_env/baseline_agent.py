#!/usr/bin/env python3
"""
Baseline Agent - Customer Support Environment

Demo agent showing how to interact with the customer support environment.
Workflow: classify_issue -> choose_solution -> escalate_decision -> close_ticket
"""

import random
import argparse

try:
    from client import CustomerSupportEnv
    from models import SupportAction
except ImportError:
    from my_env.client import CustomerSupportEnv
    from my_env.models import SupportAction


class SimpleAgent:
    """Simple support agent that follows a basic strategy."""
    
    ISSUE_TYPES = ["billing", "account", "bug", "feature"]
    
    CATEGORIES = {
        "billing": ["duplicate_charge", "wrong_amount", "subscription_issue", "fraud"],
        "account": ["password", "email", "2fa", "security"],
        "bug": ["app_crash", "ui_glitch", "missing_data", "critical"],
        "feature": ["how_to", "capability", "api", "custom"],
    }
    
    SOLUTIONS = {
        "billing": {
            "duplicate_charge": ["refund_duplicate_charge", "investigate_fraud"],
            "wrong_amount": ["correct_invoice", "refund_difference"],
            "subscription_issue": ["cancel_subscription", "update_subscription"],
            "fraud": ["escalate_security", "freeze_account"],
        },
        "account": {
            "password": ["reset_password_link", "send_recovery_email"],
            "email": ["update_email_settings", "verify_new_email"],
            "2fa": ["reset_2fa", "send_recovery_codes"],
            "security": ["escalate_security", "freeze_account"],
        },
        "bug": {
            "app_crash": ["update_app_version", "clear_cache_restart"],
            "ui_glitch": ["clear_cache_restart", "escalate_engineering"],
            "missing_data": ["sync_data", "escalate_engineering"],
            "critical": ["escalate_engineering", "create_hotfix"],
        },
        "feature": {
            "how_to": ["explain_feature", "send_tutorial"],
            "capability": ["escalate_sales", "enable_feature_trial"],
            "api": ["escalate_sales", "schedule_consultation"],
            "custom": ["escalate_sales", "create_feature_request"],
        },
    }
    
    def __init__(self):
        pass
    
    def choose_classification(self) -> str:
        """Choose an issue type."""
        return random.choice(self.ISSUE_TYPES)
    
    def choose_category(self, classification: str) -> str:
        """Choose a category for the issue type."""
        categories = self.CATEGORIES.get(classification, [])
        return random.choice(categories) if categories else "how_to"
    
    def choose_solution(self, classification: str, category: str) -> str:
        """Choose a solution."""
        solutions = self.SOLUTIONS.get(classification, {}).get(category, [])
        return random.choice(solutions) if solutions else "explain_feature"
    
    def decide_escalation(self) -> bool:
        """Decide whether to escalate."""
        return random.choice([True, False])


def run_episode(env: CustomerSupportEnv, agent: SimpleAgent) -> dict:
    """Run one complete 4-step support ticket episode."""
    try:
        # Reset to get new ticket
        obs = env.reset()
        steps = 0
        total_reward = 0.0
        
        # Step 1: Classify Issue
        actions_taken = []
        classification = agent.choose_classification()
        action = SupportAction(
            action_type="classify_issue",
            classification=classification
        )
        obs = env.step(action)
        total_reward += obs.classification_reward or 0.0
        steps += 1
        actions_taken.append(f"classify:{classification}")
        
        # Step 2: Choose Solution (with category)
        category = agent.choose_category(classification)
        solution = agent.choose_solution(classification, category)
        action = SupportAction(
            action_type="choose_solution",
            category=category,
            solution=solution
        )
        obs = env.step(action)
        total_reward += obs.solution_reward or 0.0
        steps += 1
        actions_taken.append(f"category:{category},solution:{solution}")
        
        # Step 3: Make Escalation Decision
        should_escalate = agent.decide_escalation()
        action = SupportAction(
            action_type="escalate_decision",
            should_escalate=should_escalate
        )
        obs = env.step(action)
        total_reward += obs.escalation_reward or 0.0
        steps += 1
        actions_taken.append(f"escalate:{should_escalate}")
        
        # Step 4: Close Ticket
        action = SupportAction(action_type="close_ticket")
        obs = env.step(action)
        total_reward += obs.closure_reward or 0.0
        steps += 1
        actions_taken.append("closed")
        
        return {
            "ticket_id": obs.ticket_id,
            "total_reward": total_reward,
            "episode_score": obs.episode_score or 0.0,
            "steps": steps,
            "actions": actions_taken,
            "final_status": obs.status
        }
    
    except Exception as e:
        print(f"Error in episode: {e}")
        return {
            "ticket_id": "error",
            "total_reward": 0.0,
            "episode_score": 0.0,
            "steps": 0,
            "actions": [],
            "final_status": "error",
            "error": str(e)
        }


def run_baseline(env_url: str, num_episodes: int, seed: int, task_id: int = 1):
    """Run baseline agent on support environment."""
    random.seed(seed)
    agent = SimpleAgent()
    
    print(f"\nCustomer Support Baseline Agent")
    print(f"URL: {env_url}")
    print(f"Task: {task_id} (1=Easy, 2=Medium, 3=Hard)")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    episodes_data = []
    
    try:
        env = CustomerSupportEnv(base_url=env_url)
        # Set task
        env.set_task(task_id)
        
        for episode_num in range(num_episodes):
            ep_data = run_episode(env, agent)
            episodes_data.append(ep_data)
            
            score = ep_data.get("episode_score", 0.0)
            reward = ep_data.get("total_reward", 0.0)
            status = "[OK]" if ep_data["final_status"] == "resolved" else "[--]"
            print(f"Ep {episode_num+1:2d}: {status} score={score:.2f} reward={reward:.2f} ticket={ep_data['ticket_id']}")
        
        # Summary
        avg_reward = sum(e.get("total_reward", 0) for e in episodes_data) / (len(episodes_data) or 1)
        avg_score = sum(e.get("episode_score", 0) for e in episodes_data) / len(episodes_data)
        success_count = sum(1 for e in episodes_data if e["final_status"] == "resolved")
        success_rate = success_count / len(episodes_data)
        
        print(f"\n{'='*60}")
        print(f"Summary ({num_episodes} episodes):")
        print(f"  Avg Reward: {avg_reward:.3f}")
        print(f"  Avg Score: {avg_score:.3f} (0.0-1.0)")
        print(f"  Success Rate: {success_rate*100:.1f}%")
        print(f"  Completed Episodes: {success_count}/{num_episodes}")
    
    except Exception as e:
        print(f"Failed to connect to environment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline agent for customer support environment")
    parser.add_argument("--url", default="http://localhost:8000", help="Environment URL")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--task", type=int, default=1, help="Task ID (1/2/3)")
    
    args = parser.parse_args()
    
    try:
        run_baseline(args.url, args.episodes, args.seed, args.task)
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
    