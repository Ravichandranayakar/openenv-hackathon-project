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

## Overview

**Key Features:**
- Full OpenEnv spec compliance (typed models, reset/step/state, openenv.yaml)
- 3 task difficulty levels (Easy → Medium → Hard)
- Policy-based resolution validation (issue type → category → solution)
- 14 realistic pre-generated support tickets
- Step-by-step grading (4-phase workflow)
- Automated graders (scores 0.0-1.0 per episode)
- Baseline agent with example strategy
- Containerized deployment (Docker + HF Spaces)

## Environment Design

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

```python
ticket_id: str                          # Unique ticket ID (T001-T014)
message: str                            # Customer's issue description
severity: str                           # "low" | "medium" | "high"

# Per-step feedback
classification: str                     # Agent's classification (if provided)
correct_classification: bool            # Correctness flag
classification_reward: float            # Points awarded (±0.2)

category: str                           # Agent's category choice
correct_category: bool                  # Correctness flag

solution: str                           # Agent's solution choice
correct_solution: bool                  # Correctness flag
solution_reward: float                  # Points awarded (±0.3)

escalation_decision: bool               # Agent's escalation decision
correct_escalation: bool                # Correctness flag
escalation_reward: float                # Points awarded (±0.3)

closure_reward: float                   # Points for proper closure (±0.2)

# Episode summary
episode_reward: float                   # Total accumulated reward
episode_score: float                    # Normalized score (0.0-1.0)
task_id: int                            # Current difficulty (1-3)
status: str                             # Current episode status
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

```bash
# Activate environment
source my_env/.venv/bin/activate  # Linux/Mac
# or
my_env\.venv\Scripts\Activate.ps1  # Windows

# Start server
python -m uvicorn my_env.server.app:app --reload --port 8000

# In another terminal, run baseline agent
python -m my_env.baseline_agent --url http://localhost:8000 --episodes 5 --task 1
```

## Deployment

Visit: https://huggingface.co/spaces/RavichandraNayakar/my_env

