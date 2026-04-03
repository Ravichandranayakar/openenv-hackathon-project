#!/usr/bin/env python3
"""
COMPLETE ENVIRONMENT TESTING WALKTHROUGH
Shows EXACTLY what's happening at each step
"""

# Use hybrid imports
try:
    from my_env import (
        TICKETS, RewardCalculator, CustomerSupportEnvironment, 
        SupportAction, TicketResolver, get_random_ticket
    )
    from my_env.server.data.tickets import get_ticket_by_id
except ImportError:
    from server.data.tickets import TICKETS, get_ticket_by_id, RewardCalculator
    from server.customer_support_environment import CustomerSupportEnvironment
    from models import SupportAction

print("=" * 80)
print("STEP 1: LOADING AND CHECKING DATA")
print("=" * 80)

print(f"\n--> Loaded {len(TICKETS)} tickets from dataset")
print("\nSample tickets:")
for i, ticket in enumerate(TICKETS[:3]):
    print(f"\n  Ticket {i+1}: {ticket['id']}")
    print(f"    Message: {ticket['message'][:60]}...")
    print(f"    Severity: {ticket['severity']}")
    print(f"    - GROUND TRUTH (what agent needs to find):")
    print(f"       - Correct Type: {ticket['correct_type']}")
    print(f"       - Correct Category: {ticket['correct_category']}")
    print(f"       - Correct Solution: {ticket['correct_primary_solution']}")
    print(f"       - Needs Escalation: {ticket['needs_escalation']}")

print("\n" + "=" * 80)
print("STEP 2: TESTING REWARD SYSTEM")
print("=" * 80)

print("\n--> Testing reward calculations...")

# Pick first ticket
test_ticket = get_ticket_by_id("T001")
print(f"\nUsing Ticket: {test_ticket['id']}")
print(f"Message: {test_ticket['message'][:60]}...")

print("\n--- SCENARIO A: CORRECT ANSWERS ---")
# Correct answers
classification_reward = RewardCalculator.classify_step("T001", test_ticket['correct_type'])
print(f"Step 1 - Classify as '{test_ticket['correct_type']}': Reward = {classification_reward}")

category = test_ticket['correct_category']
solution = test_ticket['correct_primary_solution']
solution_reward = RewardCalculator.solution_step("T001", test_ticket['correct_type'], category, solution)
print(f"Step 2 - Solve with '{solution}': Reward = {solution_reward}")

escalation_reward = RewardCalculator.escalation_step("T001", test_ticket['needs_escalation'])
print(f"Step 3 - Escalate: {test_ticket['needs_escalation']}: Reward = {escalation_reward}")

closure_reward = 0.2  # Always 0.2 for closure
print(f"Step 4 - Close ticket: Reward = {closure_reward}")

total_correct = classification_reward + solution_reward + escalation_reward + closure_reward
print(f"\n--> TOTAL REWARD (all correct): {total_correct:.2f} / 1.0")

print("\n--- SCENARIO B: WRONG ANSWERS ---")
# Wrong answers
wrong_classification_reward = RewardCalculator.classify_step("T001", "billing" if test_ticket['correct_type'] != "billing" else "account")
print(f"Step 1 - Classify WRONG: Reward = {wrong_classification_reward}")

wrong_solution_reward = RewardCalculator.solution_step("T001", test_ticket['correct_type'], "wrong_category", "wrong_solution")
print(f"Step 2 - Solve WRONG: Reward = {wrong_solution_reward}")

wrong_escalation_reward = RewardCalculator.escalation_step("T001", not test_ticket['needs_escalation'])
print(f"Step 3 - Escalate WRONG: Reward = {wrong_escalation_reward}")

total_wrong = wrong_classification_reward + wrong_solution_reward + wrong_escalation_reward + closure_reward
print(f"\n--> TOTAL REWARD (mostly wrong): {total_wrong:.2f} / 1.0")

print("\n" + "=" * 80)
print("STEP 3: ENVIRONMENT INITIALIZATION")
print("=" * 80)

# Create environment (already imported at top)
env = CustomerSupportEnvironment()
print("\n--> Environment created")

# Reset to get first ticket
observation = env.reset()
print(f"\n--> Environment reset")
print(f"  - Message: {observation.message[:60]}...")
print(f"  - Severity: {observation.severity}")
print(f"  - Total reward so far: {observation.episode_reward}")

# Get ground truth from environment's current ticket
ground_truth_ticket = env.current_ticket
print(f"\n   - GROUND TRUTH FOR THIS TICKET (ticket ID hidden from agent):")
print(f"     - Actual Ticket ID: {ground_truth_ticket['id']} (NOT visible to agent)")
print(f"     - Should classify as: {ground_truth_ticket['correct_type']}")
print(f"     - Should choose category: {ground_truth_ticket['correct_category']}")
print(f"     - Should choose solution: {ground_truth_ticket['correct_primary_solution']}")
print(f"     - Should escalate: {ground_truth_ticket['needs_escalation']}")

print("\n" + "=" * 80)
print("STEP 4: EXECUTING FIRST ACTION (CLASSIFY)")
print("=" * 80)

# Try correct classification
correct_type = ground_truth_ticket['correct_type']
print(f"\n→ Taking action: Classify as '{correct_type}' (CORRECT)")

action = SupportAction(
    action_type="classify_issue",
    classification=correct_type
)
observation = env.step(action)

print(f"\n--> Action executed")
print(f"  - Classification reward: {observation.classification_reward}")
print(f"  - Total reward so far: {observation.episode_reward}")
print(f"  - Episode complete: {observation.done}")

print("\n" + "=" * 80)
print("STEP 5: EXECUTING SECOND ACTION (CHOOSE SOLUTION)")
print("=" * 80)

correct_category = ground_truth_ticket['correct_category']
correct_solution = ground_truth_ticket['correct_primary_solution']

print(f"\n→ Taking action: Category='{correct_category}', Solution='{correct_solution}' (CORRECT)")

action = SupportAction(
    action_type="choose_solution",
    category=correct_category,
    solution=correct_solution
)
observation = env.step(action)

print(f"\n--> Action executed")
print(f"  - Solution reward: {observation.solution_reward}")
print(f"  - Total reward so far: {observation.episode_reward}")
print(f"  - Episode complete: {observation.done}")

print("\n" + "=" * 80)
print("STEP 6: EXECUTING THIRD ACTION (ESCALATION)")
print("=" * 80)

should_escalate = ground_truth_ticket['needs_escalation']
print(f"\n→ Taking action: Escalate={should_escalate} (CORRECT)")

action = SupportAction(
    action_type="escalate_decision",
    should_escalate=should_escalate
)
observation = env.step(action)

print(f"\n--> Action executed")
print(f"  - Escalation reward: {observation.escalation_reward}")
print(f"  - Total reward so far: {observation.episode_reward}")
print(f"  - Episode complete: {observation.done}")

print("\n" + "=" * 80)
print("STEP 7: EXECUTING FOURTH ACTION (CLOSE TICKET)")
print("=" * 80)

print(f"\n→ Taking action: Close ticket")

action = SupportAction(
    action_type="close_ticket"
)
observation = env.step(action)

print(f"\n--> Action executed")
print(f"  - Closure reward: {observation.closure_reward}")
print(f"  - Total reward so far: {observation.episode_reward}")
print(f"  - Episode complete: {observation.done}")

print("\n" + "=" * 80)
print("STEP 8: COMPLETE EPISODE SUMMARY")
print("=" * 80)

print(f"\n--> EPISODE COMPLETE")
print(f"  - Final Episode Score: {observation.episode_score} / 1.0")
print(f"\n  Score breakdown:")
print(f"    - Classification: +{0.2} (CORRECT)")
print(f"    - Solution: +{0.3} (CORRECT)")
print(f"    - Escalation: +{0.3 if should_escalate else 0.3} (CORRECT)")
print(f"    - Closure: +{0.2} (ALWAYS)")
print(f"    ────────────────")
print(f"    - TOTAL: {observation.episode_score:.2f}")

print("\n" + "=" * 80)
print("STEP 9: TESTING WRONG ANSWERS")
print("=" * 80)

# Reset to new ticket
observation = env.reset()
wrong_ticket = env.current_ticket

print(f"\n--> New ticket:")
print(f"  Message: {observation.message[:60]}...")
print(f"  Ground truth type: {wrong_ticket['correct_type']}")

# Deliberately give WRONG answer
wrong_type = "billing" if wrong_ticket['correct_type'] != "billing" else "account"
print(f"\n--> Taking WRONG action: Classify as '{wrong_type}' (should be '{wrong_ticket['correct_type']}')")

action = SupportAction(
    action_type="classify_issue",
    classification=wrong_type
)
observation = env.step(action)

print(f"\n--> Wrong action executed")
print(f"  - Classification reward: {observation.classification_reward}")
print(f"  - Status: {observation.status}")
print(f"  - Episode reward: {observation.episode_reward}")
print(f"  -> NOTICE: Reward is NEGATIVE for wrong answer! {observation.classification_reward}")

print("\n" + "=" * 80)
print("STEP 10: UNDERSTANDING WHAT YOU'RE TESTING")
print("=" * 80)

print("""
--> DATA LAYER:
  - 14 tickets with complete ground truth
  - Each ticket has: id, message, severity, correct_type, correct_category, 
    correct_solution, needs_escalation
  - Tickets are deterministic (same every time)

--> GRADING LAYER:
  - Classification: ±0.2 (correct/wrong)
  - Solution: ±0.3 (correct/wrong)
  - Escalation: ±0.3 (correct/wrong)
  - Closure: +0.2 (always)
  - Total: 0.0-1.0 per episode

--> ENVIRONMENT LAYER:
  - Resets to get new ticket
  - Steps through 4 actions in sequence
  - Validates each action against ground truth
  - Returns reward immediately if wrong
  - Episode ends after 4 steps

--> AGENT LEARNING:
  - Agent sees: ticket message + severity (NOT ground truth)
  - Agent makes: 4 sequential decisions
  - Environment returns: reward + feedback
  - Agent learns: pattern matching through trial-and-error
  - Over episodes: scores should increase (learning!)

--> WHAT MAKES THIS VALID:
  -  Deterministic grading (no randomness)
  -  Complete ground truth (all answers known)
  -  Fair rewards (normalized 0-1)
  -  No external APIs (fully offline)
  -  Multiple difficulty levels (3 task types)
  -  Fast feedback loop (immediate rewards)
""")

print("\n" + "=" * 80)
print("TESTING COMPLETE!")
print("=" * 80)
