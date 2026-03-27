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

A production-grade customer support ticket resolution environment. Agents learn to classify issues, select solutions, and decide when to escalate - with realistic, deterministic grading.

## Overview

**Key Features:**
- Full OpenEnv spec compliance (typed models, reset/step/state, openenv.yaml)
- 3 task difficulty levels (Easy → Medium → Hard)
- 14 realistic pre-generated support tickets
- Policy-based resolution validation (issue type → category → solution)
- 4-phase workflow with step-by-step grading
- Automated scoring (0.0-1.0 per episode)
- Baseline agent with example strategy
- Docker + HF Spaces ready

## What Agents Can Do

Agents learn to **process customer support tickets** by:

1. **Identify Issue Type** - Is it billing, account, bug, or feature?
2. **Assess Severity** - Low, medium, or high urgency?
3. **Choose Right Solution** - Pick category + specific resolution
   - Billing: refunds, subscriptions, fraud investigation
   - Account: password resets, 2FA, security lockdowns
   - Bugs: app updates, cache clears, engineering escalation
   - Features: tutorials, trials, consultations
4. **Decide Escalation** - Should a human take over?
5. **Close Properly** - End the ticket when resolved

**Real Example:**
Agent receives: "I was charged twice this month!"
- Classifies: "billing"
- Chooses: category="duplicate_charge", solution="refund_duplicate_charge"  
- Decides: Should escalate? No (basic refund)
- Closes: Ticket resolved
- Score: +0.2 +0.3 +0.3 +0.2 = 1.0 (100%)

**Next ticket is different?** Agent learns and adapts - ground truth feedback shows correct answers.

## Quick Start

```bash
pip install -e .
python -m uvicorn my_env.server.app:app --port 8000
python demo.py --episodes 5 --task 1
```

## Project Structure

```
my_env/
├── models.py              Types & models
├── client.py              HTTP client  
├── baseline_agent.py      Example agent
└── server/
    ├── app.py             FastAPI server
    ├── customer_support_environment.py  Core logic
    ├── data/tickets.py    14 test tickets
    └── logic/ticket_resolver.py  Grading
```

## How It Works

Agent handles a ticket in 4 steps:

**Action 1: Classify Issue**
```json
{
  "action_type": "classify_issue",
  "classification": "billing|account|bug|feature"
}
```
Reward: +0.2 if correct

**Action 2: Choose Solution**
```json
{
  "action_type": "choose_solution",
  "category": "duplicate_charge|password|app_crash|...",
  "solution": "refund_duplicate_charge|reset_password_link|..."
}
```
Reward: +0.3 if correct

**Action 3: Escalation Decision**
```json
{
  "action_type": "escalate_decision",
  "should_escalate": true|false
}
```
Reward: +0.3 if correct

**Action 4: Close Ticket**
```json
{
  "action_type": "close_ticket"
}
```
Reward: +0.2 if correct

**Max Score Per Episode: 1.0**

## Observation

What agent sees each step:

```
ticket_id: str                    # "T001" - "T014"
message: str                      # Customer's issue
severity: str                     # "low" | "medium" | "high"

classification: str               # What agent classified
correct_classification: bool      # Right or wrong?
classification_reward: float      # Points for this step

category: str                     # Agent's category choice
correct_category: bool            # Right or wrong?

solution: str                     # Agent's solution choice  
correct_solution: bool            # Right or wrong?
solution_reward: float            # Points for this step

escalation_decision: bool         # Agent's escalation decision
correct_escalation: bool          # Right or wrong?
escalation_reward: float          # Points for this step

episode_reward: float             # Total points so far
episode_score: float              # Normalized 0.0-1.0
task_id: int                      # Which difficulty? (1-3)
status: str                       # Episode state
```

## 3 Tasks

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

Run any: `python demo.py --task 1` (or 2, or 3)

## Tickets & Categories

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
- `POST /step` - Process agent action
- `GET /state` - Get current episode state
- `GET /health` - Health check
- `POST /tasks` - List available tasks
- `POST /grader` - Grade episode (returns score 0.0-1.0)

## Running Locally

**Activate environment:**
```bash
source my_env/.venv/bin/activate  # Linux/Mac
# or
my_env\.venv\Scripts\Activate.ps1  # Windows
```

**Start server:**
```bash
python -m uvicorn my_env.server.app:app --reload --port 8000
```

**Run baseline agent (another terminal):**
```bash
python -m my_env.baseline_agent --url http://localhost:8000 --episodes 5 --task 1
```

## Deployment

**HuggingFace Spaces:**
```bash
openenv push --name YourUsername/support-env --token <hf_token>
```

Visit: https://huggingface.co/spaces/RavichandraNayakar/my_env

**If HF Space shows blank screen use:**
https://ravichandranayakar-customer-support-env.hf.space/web/

**Docker:**
```bash
docker build -t my-env .
docker run -p 8000:8000 my-env
```

## Code Guide

- [models.py](my_env/models.py) - Action and observation types
- [customer_support_environment.py](my_env/server/customer_support_environment.py) - Core 4-step logic
- [baseline_agent.py](my_env/baseline_agent.py) - Example agent
- [tickets.py](my_env/server/data/tickets.py) - All 14 tickets + solutions
- [ticket_resolver.py](my_env/server/logic/ticket_resolver.py) - Grading logic
