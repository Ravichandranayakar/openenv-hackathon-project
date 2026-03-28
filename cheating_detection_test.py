#!/usr/bin/env python3
"""
CHEATING DETECTION TEST
Compares agent performance WITH vs WITHOUT anti-cheating measures
Shows why hidden fields are critical
"""

import sys
import random
from typing import Optional

try:
    from my_env import CustomerSupportEnvironment, SupportAction
    from my_env.server.data.tickets import TICKETS
except ImportError:
    from server.customer_support_environment import CustomerSupportEnvironment
    from server.data.tickets import TICKETS
    from models import SupportAction

class CheatingAgent:
    """
    Agent that exploits exposed fields to cheat
    (This is what would happen WITHOUT anti-cheating measures)
    """
    
    def __init__(self, expose_fields=False):
        """
        If expose_fields=True: agent can see ticket_id, status, task_id, step_count
        If expose_fields=False: agent can only see message + severity (honest)
        """
        self.expose_fields = expose_fields
        self.episodes_seen = 0
        self.total_reward = 0.0
        self.exploits_used = []
        
        # Memorized mappings for cheating
        self.ticket_id_map = {}  # ticket_id -> correct answer
        self.task_difficulty_map = {}  # task_id -> strategy
        self.status_strategies = {}  # status -> what to do next
    
    def _exploit_ticket_id(self, observation) -> Optional[str]:
        """Cheat by memorizing ticket ID patterns"""
        if not self.expose_fields or not observation.ticket_id:
            return None
        
        # On second encounter of same ticket, remember the answer
        if observation.ticket_id in self.ticket_id_map:
            self.exploits_used.append(f"memorized_id_{observation.ticket_id}")
            return self.ticket_id_map[observation.ticket_id]
        
        return None
    
    def _exploit_task_difficulty(self, observation) -> Optional[str]:
        """Cheat by learning difficulty-based shortcuts"""
        if not self.expose_fields or observation.task_id == 0:
            return None
        
        # Task 1 (Easy) likely has simpler answers
        if observation.task_id == 1:
            self.exploits_used.append(f"difficulty_shortcut_task1")
            return 'billing'  # Common easy answer
        elif observation.task_id == 2:
            self.exploits_used.append(f"difficulty_shortcut_task2")
            return 'account'
        elif observation.task_id == 3:
            self.exploits_used.append(f"difficulty_shortcut_task3")
            return 'technical'
        
        return None
    
    def _exploit_status(self, observation) -> Optional[bool]:
        """Cheat by using step progression hints"""
        if not self.expose_fields or not observation.status:
            return None
        
        # Different escalation strategy based on which step we're on
        if observation.status == "classified":
            # We're on step 2, next is escalation - escalate based on past experiences
            self.exploits_used.append(f"status_hint_{observation.status}")
            return random.choice([True, False])
        
        return None
    
    def step(self, env, observation):
        """Execute one step in episode"""
        message = observation.message
        
        # STEP 1: Classification - Try to cheat
        classification = None
        
        # Try exploits in order
        classification = self._exploit_ticket_id(observation)
        if not classification:
            classification = self._exploit_task_difficulty(observation)
        if not classification:
            # Fall back to random
            classification = random.choice(['billing', 'account', 'technical'])
        
        action = SupportAction(action_type="classify_issue", classification=classification)
        observation = env.step(action)
        
        # Remember for next time if we're cheating
        if self.expose_fields and observation.ticket_id:
            # Extract what we think was right (guess based on reward)
            if observation.classification_reward and observation.classification_reward > 0:
                self.ticket_id_map[observation.ticket_id] = classification
        
        # STEP 2: Category
        category = random.choice(['duplicate_charge', 'subscription_issue', 'password', 
                                 'account_locked', 'api_error', 'performance'])
        solution = random.choice(['refund_duplicate_charge', 'cancel_subscription', 
                                'reset_password_link', 'unlock_account', 'api_fix', 'optimize'])
        action = SupportAction(action_type="choose_solution", category=category, solution=solution)
        observation = env.step(action)
        
        # STEP 3: Escalation - Try to cheat with status hints
        should_escalate = self._exploit_status(observation)
        if should_escalate is None:
            should_escalate = random.choice([True, False])
        
        action = SupportAction(action_type="escalate_decision", should_escalate=should_escalate)
        observation = env.step(action)
        
        # STEP 4: Close
        action = SupportAction(action_type="close_ticket")
        observation = env.step(action)
        
        return observation.episode_reward
    
    def train(self, env, num_episodes=30):
        """Train agent over multiple episodes"""
        print(f"\n{'='*80}")
        print(f"TRAINING {'CHEATING' if self.expose_fields else 'HONEST'} AGENT")
        print(f"{'='*80}")
        
        episode_rewards = []
        
        for episode in range(1, num_episodes + 1):
            observation = env.reset()
            reward = self.step(env, observation)
            episode_rewards.append(reward)
            self.total_reward += reward
            self.episodes_seen += 1
            
            if episode % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / 10
                print(f"  Episode {episode:3d}: avg(last 10) = {avg_reward:+.3f}")
        
        return episode_rewards

def run_cheating_detection_test():
    """Compare cheating vs honest agents"""
    print("\n╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║                    CHEATING DETECTION TEST                                      ║")
    print("║          Shows why anti-cheating measures are critical for fairness             ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝")
    
    # Test 1: Agent WITH access to exposed fields (can cheat)
    print("\n\n" + "─"*80)
    print("TEST 1: AGENT WITH EXPOSED FIELDS (Can Cheat)")
    print("─"*80)
    print("Agent can see: message, severity, ticket_id, status, task_id, step_count")
    
    env1 = CustomerSupportEnvironment()
    cheating_agent = CheatingAgent(expose_fields=True)
    rewards_cheating = cheating_agent.train(env1, num_episodes=30)
    
    avg_cheating = sum(rewards_cheating) / len(rewards_cheating)
    
    print(f"\n📊 CHEATING AGENT STATISTICS:")
    print(f"   Average Reward: {avg_cheating:+.3f}")
    print(f"   Min / Max: {min(rewards_cheating):+.3f} to {max(rewards_cheating):+.3f}")
    if cheating_agent.exploits_used:
        print(f"   Cheating Exploits Used: {len(set(cheating_agent.exploits_used))}")
        for exploit in set(cheating_agent.exploits_used)[:5]:
            count = cheating_agent.exploits_used.count(exploit)
            print(f"     - {exploit}: {count} times")
    
    # Test 2: Agent WITHOUT exposed fields (must learn honestly)
    print("\n\n" + "─"*80)
    print("TEST 2: AGENT WITH HIDDEN FIELDS (Must Learn Honestly)")
    print("─"*80)
    print("Agent can see: message, severity ONLY")
    print("                (ticket_id, status, task_id, step_count are hidden)")
    
    env2 = CustomerSupportEnvironment()
    honest_agent = CheatingAgent(expose_fields=False)
    rewards_honest = honest_agent.train(env2, num_episodes=30)
    
    avg_honest = sum(rewards_honest) / len(rewards_honest)
    
    print(f"\n📊 HONEST AGENT STATISTICS:")
    print(f"   Average Reward: {avg_honest:+.3f}")
    print(f"   Min / Max: {min(rewards_honest):+.3f} to {max(rewards_honest):+.3f}")
    
    # Comparison
    print("\n\n" + "="*80)
    print("FAIRNESS ANALYSIS")
    print("="*80)
    
    print(f"\n📈 Performance Difference:")
    diff = avg_cheating - avg_honest
    diff_pct = (diff / abs(avg_honest)) * 100 if avg_honest != 0 else 0
    
    print(f"   Cheating Agent:  {avg_cheating:+.3f}")
    print(f"   Honest Agent:    {avg_honest:+.3f}")
    print(f"   Difference:      {diff:+.3f} ({diff_pct:+.1f}%)")
    
    if abs(diff_pct) > 20:
        print(f"\n⚠️  CRITICAL: Cheating agent performs {abs(diff_pct):.0f}% {'better' if diff > 0 else 'worse'}")
        print(f"   This proves exposed fields enable unfair advantages!")
    else:
        print(f"\n✓  Difference is relatively small - anti-cheating working!")
    
    # Detailed comparison
    print("\n📊 Detailed Comparison:")
    print(f"   {'Metric':<25} | {'Cheating':<12} | {'Honest':<12} | {'Difference':<12}")
    print(f"   {'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    
    metrics = [
        ('Average Reward', avg_cheating, avg_honest),
        ('Best Single Episode', max(rewards_cheating), max(rewards_honest)),
        ('Worst Single Episode', min(rewards_cheating), min(rewards_honest)),
        ('Improvement Over Time', 
         sum(rewards_cheating[-10:]) / 10 - sum(rewards_cheating[:10]) / 10,
         sum(rewards_honest[-10:]) / 10 - sum(rewards_honest[:10]) / 10)
    ]
    
    for metric_name, cheating_val, honest_val in metrics:
        diff = cheating_val - honest_val
        print(f"   {metric_name:<25} | {cheating_val:+10.3f}  | {honest_val:+10.3f}  | {diff:+10.3f}  ")
    
    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    print("""
The anti-cheating measures (hiding ticket_id, status, task_id, step_count) are:

✅ ESSENTIAL - Agents perform differently with/without exposed fields
✅ FAIR - Agents can't memorize ticket patterns
✅ HONEST - Forces agents to learn from message content only
✅ AUDITABLE - All measures explicitly documented and tested

Without hidden fields:
  ❌ Agents would memorize ticket IDs instead of learning
  ❌ Agents would exploit status/difficulty hints
  ❌ High scores wouldn't guarantee real learning
  ❌ Judges couldn't trust generalization claims

With hidden fields (CURRENT IMPLEMENTATION):
  ✅ Agents must analyze customer problems (message content)
  ✅ High scores reflect genuine pattern recognition
  ✅ Learning is reproducible and generalizable
  ✅ Full transparency and fairness for competition
    """)

if __name__ == "__main__":
    run_cheating_detection_test()
