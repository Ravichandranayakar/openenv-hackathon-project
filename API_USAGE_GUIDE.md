# FastAPI Server Usage Guide

## Overview
This guide explains how to interact with the Customer Support OpenEnv FastAPI server. The server provides REST endpoints for training an AI agent to handle customer support tickets.

## Table of Contents
1. [Starting the Server](#starting-the-server)
2. [API Endpoints](#api-endpoints)
3. [Complete Episode Workflow](#complete-episode-workflow)
4. [Detailed Examples](#detailed-examples)

---

## Starting the Server

### Local Development
```bash
cd c:\path\to\openenv-hackathon-project
python -m uvicorn my_env.server.app:app --port 8000 --log-level warning
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Docker Deployment
```bash
docker build -t customer-support-env .
docker run -p 8000:8000 customer-support-env
```

---

## API Endpoints

### 1. **GET /health** - Health Check
Check if the server is running.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Response (200 OK):**
```json
{
  "status": "healthy"
}
```

---

### 2. **POST /reset** - Start New Episode
Load a random customer support ticket and initialize environment.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/reset" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response (200 OK):**
```json
{
  "observation": {
    "message": "I was charged twice for my subscription this month. Please help!",
    "severity": "high",
    "task_id": 1,
    "task_name": "Easy - Simple ticket classification",
    "status": "open",
    "reward": 0.0,
    "done": false,
    "resolution_message": "Ticket loaded. Please classify the issue type."
  },
  "reward": 0.0,
  "done": false
}
```

---

### 3. **POST /step** - Execute Action
Send an action to the environment and get observation + reward.

**Generic Format:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "ACTION_TYPE",
      "... action parameters ..."
    }
  }'
```

---

## Complete Episode Workflow

An episode has **4 mandatory steps** to complete successfully:

```
┌─────────┐
│  RESET  │  Load random ticket
└────┬────┘
     │
     ▼
┌─────────────────────────────┐
│ STEP 1: CLASSIFY ISSUE      │  Identify issue type
└────┬────────────────────────┘
     │ Reward: 0.2
     ▼
┌─────────────────────────────┐
│ STEP 2: CHOOSE SOLUTION     │  Pick category & solution
└────┬────────────────────────┘
     │ Reward: 0.3
     ▼
┌─────────────────────────────┐
│ STEP 3: ESCALATION DECISION │  Decide: escalate or close?
└────┬────────────────────────┘
     │ Reward: 0.3
     ▼
┌─────────────────────────────┐
│ STEP 4: CLOSE TICKET        │  Finalize episode
└────┬────────────────────────┘
     │ Reward: 0.2
     ▼
  DONE=true
  TOTAL SCORE: 1.0
```

---

## Detailed Examples

### Complete Example: Full Episode (4 Steps)

#### **Step 0: Reset Environment**
```bash
curl -X POST "http://127.0.0.1:8000/reset" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response:**
```json
{
  "observation": {
    "message": "I was charged twice for my subscription this month. Please help!",
    "severity": "high",
    "status": "open",
    "done": false,
    "resolution_message": "Ticket loaded. Please classify the issue type."
  },
  "reward": 0.0,
  "done": false
}
```

---

#### **Step 1: Classify Issue**
Identify the issue type: `billing`, `account`, `bug`, or `feature`

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "classify_issue",
      "classification": "billing"
    }
  }'
```

**Response (200 OK):**
```json
{
  "observation": {
    "message": "I was charged twice for my subscription this month. Please help!",
    "severity": "high",
    "classification": "billing",
    "correct_classification": true,
    "classification_reward": 0.2,
    "reward": 0.2,
    "done": false,
    "resolution_message": "Classified as 'billing'. [OK] CORRECT! (+0.2) -> Next: Select solution category."
  },
  "reward": 0.2,
  "done": false
}
```

**Valid Classifications:**
- `billing` - Payment, subscription, charges
- `account` - Login, password, profile
- `bug` - Crashes, errors, glitches
- `feature` - How-to, capabilities, requests

---

#### **Step 2: Choose Solution**
Select category and solution for the classified issue.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "choose_solution",
      "category": "duplicate_charge",
      "solution": "refund_duplicate_charge"
    }
  }'
```

**Response (200 OK):**
```json
{
  "observation": {
    "category": "duplicate_charge",
    "correct_category": true,
    "solution": "refund_duplicate_charge",
    "correct_solution": true,
    "solution_reward": 0.3,
    "reward": 0.3,
    "done": false,
    "resolution_message": "Selected category 'duplicate_charge', solution 'refund_duplicate_charge'. [OK] CORRECT! (+0.3)"
  },
  "reward": 0.3,
  "done": false
}
```

**Valid Categories & Solutions:**

| Classification | Categories | Solutions |
|---|---|---|
| **billing** | `duplicate_charge`, `wrong_amount`, `subscription_issue`, `fraud` | `refund_duplicate_charge`, `correct_invoice`, `cancel_subscription`, `escalate_security` |
| **account** | `password`, `email`, `2fa`, `security` | `reset_password_link`, `update_email_settings`, `reset_2fa`, `freeze_account` |
| **bug** | `app_crash`, `ui_glitch`, `missing_data`, `critical` | `update_app_version`, `sync_data`, `clear_cache_restart`, `escalate_engineering` |
| **feature** | `how_to`, `capability`, `api`, `custom` | `explain_feature`, `enable_feature_trial`, `escalate_sales`, `create_feature_request` |

---

#### **Step 3: Escalation Decision**
Decide whether to escalate to a human or close the ticket.

**Request (Close the ticket):**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "escalate_decision",
      "should_escalate": false
    }
  }'
```

**Response (200 OK):**
```json
{
  "observation": {
    "escalation_decision": false,
    "correct_escalation": true,
    "escalation_reward": 0.3,
    "reward": 0.3,
    "done": false,
    "resolution_message": "Decision: CLOSE. [OK] CORRECT! (+0.3) -> Next: Close ticket."
  },
  "reward": 0.3,
  "done": false
}
```

**Valid Escalation Decisions:**
- `true` - Escalate to human specialist
- `false` - Close the ticket

---

#### **Step 4: Close Ticket (Final Step)**
Close the ticket and complete the episode.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "close_ticket"
    }
  }'
```

**Response (200 OK):**
```json
{
  "observation": {
    "status": "resolved",
    "closure_reward": 0.2,
    "reward": 0.2,
    "episode_reward": 1.0,
    "episode_score": 1.0,
    "done": true,
    "resolution_message": "Ticket closed. Episode complete. Total reward: 1.0/1.0 = 100%"
  },
  "reward": 0.2,
  "done": true
}
```

---

### 4. **GET /state** - Current Episode State
Get the current state of the ongoing episode.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/state"
```

**Response (200 OK):**
```json
{
  "episode_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "step_count": 4
}
```

---

### 5. **GET /schema** - JSON Schemas
Get the schemas for actions, observations, and state.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/schema"
```

**Response (200 OK):**
```json
{
  "action": {
    "type": "object",
    "properties": {
      "action_type": {"type": "string"},
      "classification": {"type": "string"},
      "category": {"type": "string"},
      "solution": {"type": "string"},
      "should_escalate": {"type": "boolean"}
    },
    "required": ["action_type"]
  },
  "observation": {
    "type": "object",
    "properties": {
      "message": {"type": "string"},
      "severity": {"type": "string"},
      "reward": {"type": "number"},
      "done": {"type": "boolean"},
      "episode_score": {"type": "number"}
    },
    "required": ["message", "severity"]
  },
  "state": {
    "type": "object",
    "properties": {
      "episode_id": {"type": "string"},
      "step_count": {"type": "integer"}
    }
  }
}
```

---

### 6. **POST /tasks** - List Available Tasks
Get list of task difficulties.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/tasks"
```

**Response (200 OK):**
```json
{
  "tasks": [
    {
      "id": 1,
      "name": "Easy",
      "description": "Simple ticket classification"
    },
    {
      "id": 2,
      "name": "Medium",
      "description": "Mixed ticket types with escalation"
    },
    {
      "id": 3,
      "name": "Hard",
      "description": "Complex cases requiring expertise"
    }
  ]
}
```

---

## Common Errors & Solutions

### Error: 422 Validation Error
**Cause:** Invalid request format or missing required fields.
**Fix:** Verify action structure matches the API spec.

Example of **wrong** format:
```json
{
  "observation": {  
    "data": "..."
  }
}
```

Example of **correct** format:
```json
{
  "action": {
    "action_type": "classify_issue",
    "classification": "billing"
  }
}
```

---

### Error: "No episode in progress"
**Cause:** Called `/step` without calling `/reset` first.
**Fix:** Always call `/reset` before `/step`.

---

### Error: "Classification already done"
**Cause:** Called `/step` with `classify_issue` twice.
**Fix:** Proceed to next step (choose_solution).

---

## Python Example: Complete Agent

```python
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Step 0: Reset
reset_resp = requests.post(f"{BASE_URL}/reset", json={})
observation = reset_resp.json()["observation"]
print(f"Ticket: {observation['message']}")

# Step 1: Classify
action1 = {"action": {"action_type": "classify_issue", "classification": "billing"}}
resp1 = requests.post(f"{BASE_URL}/step", json=action1)
print(f"Step 1 Reward: {resp1.json()['observation']['classification_reward']}")

# Step 2: Choose Solution
action2 = {
    "action": {
        "action_type": "choose_solution",
        "category": "duplicate_charge",
        "solution": "refund_duplicate_charge"
    }
}
resp2 = requests.post(f"{BASE_URL}/step", json=action2)
print(f"Step 2 Reward: {resp2.json()['observation']['solution_reward']}")

# Step 3: Escalation Decision
action3 = {"action": {"action_type": "escalate_decision", "should_escalate": False}}
resp3 = requests.post(f"{BASE_URL}/step", json=action3)
print(f"Step 3 Reward: {resp3.json()['observation']['escalation_reward']}")

# Step 4: Close
action4 = {"action": {"action_type": "close_ticket"}}
resp4 = requests.post(f"{BASE_URL}/step", json=action4)
final_score = resp4.json()['observation']['episode_score']
print(f"Final Score: {final_score} (Perfect: 1.0)")
```

---

## Reward System

| Step | Max Reward | Condition |
|---|---|---|
| Classify Issue | 0.2 | Correct classification |
| Choose Solution | 0.3 | Correct category & solution |
| Escalation Decision | 0.3 | Correct decision (escalate vs close) |
| Close Ticket | 0.2 | Ticket closed properly |
| **Total** | **1.0** | All steps correct = 100% |

---

## Integration with Inference Script

The `inference.py` script automatically:
1. Calls `/reset` to load a ticket
2. Calls `/step` 4 times with optimal actions
3. Emits structured logs in `[START]`, `[STEP`, `[END]` format
4. Calculates final score

Run it with:
```bash
export API_BASE_URL="http://127.0.0.1:8000"
export MODEL_NAME="gpt-4-turbo"
export HF_TOKEN="your-api-key"
python inference.py
```

---

## OpenAPI Documentation
Once the server is running, access the interactive Swagger UI at:
```
http://127.0.0.1:8000/docs
```

Or the ReDoc documentation at:
```
http://127.0.0.1:8000/redoc
```
