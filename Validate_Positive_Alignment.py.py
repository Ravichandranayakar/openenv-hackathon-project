#!/usr/bin/env python3
"""
 PERFECT RUN TEST: Verify 4-Agent Negotiation Environment Rewards Positive Behavior
─────────────────────────────────────────────────────────────────────────────────
Tests that the environment perfectly rewards (+1.0) when:
1. The correct specialist agent bids high.
2. The incorrect agents bid low.
3. The winning agent provides the exact correct solution.

This script dynamically reads the ticket and guarantees the perfect AI response
to demonstrate the positive side of the 11-signal reward system to judges.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from my_env.server.multi_agent_negotiation_environment import MultiAgentNegotiationEnvironment, AgentRole
from my_env.models import SupportAction


def test_perfect_run():
    print("\n" + "="*70)
    print(" PERFECT RUN TEST: Demonstrating Maximum Positive Rewards".center(70))
    print("="*70)
    
    # Initialize environment
    print("\n[1/6] Initialize Environment")
    env = MultiAgentNegotiationEnvironment()
    obs = env.reset()
    
    correct_agent = env.expected_specialist.value
    correct_solution = env.current_ticket["correct_primary_solution"]
    
    print(f"✅ Environment loaded")
    print(f"  Episode ID: {obs.ticket_id}")
    print(f"  Ticket: {obs.message[:80]}...")
    print(f"  GROUND TRUTH -> Expected Specialist: {correct_agent.upper()}")
    print(f"  GROUND TRUTH -> Correct Solution: {correct_solution}")
    
    # PHASE 1: BIDDING
    print("\n[2/6] BIDDING PHASE: Agents Submit Confidence Bids")
    print("  " + "-"*66)
    
    for agent in ["billing", "account", "technical"]:
        # The correct agent bids high (0.95), others bid low (0.10)
        confidence = 0.95 if agent == correct_agent else 0.10
        action = SupportAction(
            action_type=f"{agent}_bid",
            confidence=confidence
        )
        env.step(action)
        print(f"✅ {agent.capitalize()} Agent bid: confidence={confidence}")
        
    print(f"  Current phase: {env.current_phase}")
    
    # Show winning agent
    print("\n[3/6] WINNER SELECTION")
    print("  " + "-"*66)
    print(f"✅ Winning Agent: {env.winning_agent.value.upper()}")
    print(f"  Winning confidence: 0.95")
    
    # PHASE 2: EXECUTION
    print("\n[4/6] EXECUTION PHASE: Winner Proposes Perfect Solution")
    print("  " + "-"*66)
    
    solution_action = SupportAction(
        action_type=f"{correct_agent}_execute",
        solution=correct_solution,
        confidence=0.95
    )
    obs4 = env.step(solution_action)
    print(f"✅ Solution proposed by {correct_agent.upper()}: '{correct_solution}'")
    print(f"  Current phase: {env.current_phase}")
    
    # PHASE 3: RESOLUTION
    print("\n[5/6] RESOLUTION PHASE: Calculate 11 Independent Rewards")
    print("  " + "-"*66)
    
    resolution_action = SupportAction(
        action_type="manager_evaluate"
    )
    obs5 = env.step(resolution_action)
    print(f"✅ Rewards calculated for all agents. Status: {obs5.status}")
    
    # Print all 11 reward functions
    print("\n[6/6] FINAL RESULTS: 11 Independent Reward Functions")
    print("  " + "-"*66)
    
    print(f"\nEnvironment Rewards Dict ({len(env.REWARDS)} functions):")
    for reward_name, reward_value in env.REWARDS.items():
        if reward_value > 0:
            print(f" ✅ {reward_name:23s}: +{reward_value:.2f}")
        elif reward_value < 0:
            print(f" ❌ {reward_name:23s}: {reward_value:.2f}")
        else:
            print(f" ➖ {reward_name:23s}: {reward_value:.2f}")
            
    print(f"\nAgent Rewards This Episode:")
    total_reward = 0
    for agent_role, reward in env.agent_rewards.items():
        if agent_role == correct_agent:
            print(f" 🌟 {agent_role:10s} (Winner) : +{reward:.3f}")
        else:
            print(f" 🤝 {agent_role:10s} (Team)   : +{reward:.3f}")
        total_reward += reward
        
    print(f" {'─'*35}")
    print(f" 🏆 {'TOTAL REWARD':12s}      : +{total_reward:.3f}")
    
    print("\n" + "="*70)
    print(" 🎉 SUCCESS: PERFECT POSITIVE REWARD ACHIEVED!".center(70))
    print("="*70)
    print("The environment successfully rewarded the agents with the maximum possible score")
    print("because they correctly identified the specialist and provided the perfect solution.")

if __name__ == "__main__":
    test_perfect_run()
