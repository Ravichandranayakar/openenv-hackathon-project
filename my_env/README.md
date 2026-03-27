---
title: Customer Support OpenEnv Environment
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - customer-support
  - agent-training
---

# Customer Support OpenEnv Environment

A production-grade **Customer Support ticket resolution environment** built with the **OpenEnv framework** (Meta PyTorch + Hugging Face). Agents learn to classify issues, select solutions, and decide on escalation with realistic, deterministic grading.

## Quick Start

```bash
# Install dependencies
pip install -e my_env

# Start server (Terminal 1)
python -m uvicorn my_env.server.app:app --reload --port 8000

# Run baseline agent (Terminal 2)
python my_env/baseline_agent.py --url http://localhost:8000 --episodes 5 --task 1
```

**Expected output:**
```
Episode 1/5 | Step 3/4 completed
  Classification: ✓ | Solution: ✓ | Escalation: ✓
  Score: 1.0 (100%) | Agent learning...
Episode 2/5 ...
```

## Key Features

- ✅ **Full OpenEnv spec** - typed models, reset/step/state, openenv.yaml
- ✅ **Gymnasium API** - explicit reward, done, truncated fields
- ✅ **3 difficulty levels** - Easy, Medium, Hard (task_id 1-3)
- ✅ **Policy-based grading** - issue type → category → solution → escalation
- ✅ **14 realistic tickets** - pre-generated with ground truth answers
- ✅ **Step-by-step feedback** - agents receive ground truth when wrong
- ✅ **Docker ready** - runs in HF Spaces with `openenv push`
- ✅ **Baseline agent included** - example learning strategy

## Environment Design

### Gymnasium-Style API

Agents interact using the **Gymnasium standard pattern**:

```python
obs = env.reset()  # Start episode

while not obs.done:
    action = SupportAction(action_type="...", ...)
    obs = env.step(action)
    
    # Read Gymnasium returns:
    reward = obs.reward          # Points for THIS action
    done = obs.done              # Episode complete?
    episode_reward = obs.episode_reward  # Total accumulated
```

**Key fields in SupportObservation:**
- `reward: float` - Reward for the current step
- `done: bool` - Episode is complete (terminal state)
- `truncated: bool` - Episode was cut short (max steps)
- `episode_reward: float` - Total reward accumulated (0.0-1.0)
- `resolution_message: str` - Feedback with ground truth if wrong

### Action Space

Agents interact through 4 sequential actions:

**Action 1: Classify Issue**
```json
{
  "action_type": "classify_issue",
  "classification": "billing|account|bug|feature"
}
```

**Action 2: Choose Solution**
```json
{
  "action_type": "choose_solution",
  "category": "duplicate_charge|password|app_crash|...",
  "solution": "refund_duplicate_charge|reset_password_link|..."
}
```

**Action 3: Escalation Decision**
```json
{
  "action_type": "escalate_decision",
  "should_escalate": true|false
}
```

**Action 4: Close Ticket**
```json
{
  "action_type": "close_ticket"
}
```

### Observation Space

**What agents see and how to read it:**

```python
# STATE - What the agent needs to know
observation.ticket_id          # "T001"
observation.message            # "Database connection timing out..."
observation.severity           # "low" | "medium" | "high"
observation.status             # "open" | "classified" | "resolved" | "error"

# FEEDBACK - How the agent did on this step
observation.classification     # What the agent classified as
observation.correct_classification  # True/False
observation.classification_reward    #  +0.2 if correct, ±0.0 if wrong

observation.category           # Agent's chosen category
observation.correct_category   # True/False  
observation.solution           # Agent's chosen solution
observation.correct_solution   # True/False
observation.solution_reward    # +0.3 if correct, ±0.0 if wrong

observation.escalation_decision # Agent's true/false decision
observation.correct_escalation  # True/False
observation.escalation_reward   # +0.3 if correct, ±0.0 if wrong

# GYMNASIUM RETURNS - Standard RL API
observation.reward             # Reward for THIS step (0.0-0.3)
observation.done               # Episode complete? True/False
observation.truncated          # Cut short by max steps? True/False

# EPISODE SUMMARY
observation.episode_reward     # Total accumulated (0.0-1.0)
observation.episode_score      # Normalized score (0.0-1.0)
observation.resolution_message # Feedback text + ground truth if wrong
```

**When agent is wrong, agent sees ground truth:**
```
✗ INCORRECT. Correct answer: 'bug' (Learn: 'bug' issues are technical problems)
Correct decision: ESCALATE
Category - Correct: 'app_crash' | Solution - Correct: 'restart_service'
```

### Reward Function

| Phase | Action | Reward | Scoring |
|-------|--------|--------|---------|
| 1 | classify_issue (correct) | +0.2 | 20% |
| 2 | choose_solution (correct) | +0.3 | 30% |
| 3 | escalate_decision (correct) | +0.3 | 30% |
| 4 | close_ticket (if phase 3 correct) | +0.2 | 20% |

**Max Episode Reward: 1.0**

## Tasks

**Task 1 - Easy:** 
- Simple unambiguous tickets (billing refunds, password resets)
- Minimal escalation required
- Example: Duplicate charge, account lockout

**Task 2 - Medium:**
- Mixed ticket types with some escalation cases
- Requires category reasoning
- Example: Fraud investigation, email verification issues

**Task 3 - Hard:**
- Complex security/critical issues
- Frequent escalation scenarios
- Example: Account hacked, critical feature broken, data loss

## Tickets & Resolution Policies

### Supported Categories

**Billing Issues:**
- duplicate_charge → refund_duplicate_charge, investigate_fraud
- wrong_amount → correct_invoice, refund_difference
- subscription_issue → cancel_subscription, update_subscription
- fraud → escalate_security, freeze_account

**Account Issues:**
- password → reset_password_link, send_recovery_email
- email → update_email_settings, verify_new_email
- 2fa → reset_2fa, send_recovery_codes
- security → escalate_security, freeze_account

**Bug Issues:**
- app_crash → update_app_version, clear_cache_restart
- ui_glitch → clear_cache_restart, escalate_engineering
- missing_data → sync_data, escalate_engineering
- critical → escalate_engineering, create_hotfix

**Feature Issues:**
- how_to → explain_feature, send_tutorial
- capability → escalate_sales, enable_feature_trial
- api → escalate_sales, schedule_consultation
- custom → escalate_sales, create_feature_request

## API Endpoints

- `POST /reset` - Start new episode, load random ticket
- `POST /step` - Process agent action (classify/choose_solution/escalate_decision/close_ticket)
- `GET /state` - Get current episode state
- `GET /health` - Health check
- `POST /tasks` - List available tasks
- `POST /grader` - Grade episode (returns score 0.0-1.0)

## Running Locally

**Prerequisites:** Python 3.10+

```bash
# 1. Install environment
pip install -e my_env

# 2. Start FastAPI server (Terminal 1)
python -m uvicorn my_env.server.app:app --reload --port 8000

# 3. Run agents (Terminal 2)
python my_env/baseline_agent.py --url http://localhost:8000 --episodes 5 --task 1
```

**Verify success:**
- Server logs: `INFO: Application startup complete`
- Agent output: `Episode 1/5 | Score: 0.95 (95%)`

### Testing Framework

All files compile without syntax errors:
```bash
python -m py_compile my_env/*.py my_env/server/*.py my_env/server/data/*.py
```

Environment is fully tested with baseline agent on all difficulty levels (Task 1-3).

## How Agents Learn

**The Learning Loop:**

1. **Episode starts**: Agent receives ticket (e.g., "Database timing out")
2. **Phase 1**: Agent classifies type (guess: "billing" → WRONG)
   - Feedback: `✗ INCORRECT. Correct answer: 'bug'` 
   - Reward: 0.0, episode_reward: 0.0
3. **Phase 2**: Agent sees feedback, chooses solution category
   - On next similar ticket, remembers "database → bug"
   - Guesses "bug" (now CORRECT)
   - Reward: +0.2, episode_reward: 0.2
4. **Phase 3**: Agent proposes escalation decision
5. **Phase 4**: Ticket closed, episode complete
   - Final score: 0.6-1.0 depending on accuracy

**Why ground truth feedback matters:**
- Without it: Agent only knows right/wrong
- With it: Agent learns **what correct looks like**
- 5-10 episodes: Agent shows improvement
- 20+ episodes: Agent masters task

This is how real RL training works!

## Deployment

### To HuggingFace Spaces

```bash
openenv push --name RavichandraNayakar/my_env --token <hf_token>
```

Then visit: https://huggingface.co/spaces/RavichandraNayakar/my_env

### Docker

```bash
docker build -t my-env .
docker run -p 8000:8000 my-env
```

## Architecture

