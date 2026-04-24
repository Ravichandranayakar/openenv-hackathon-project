# FastAPI Server & Multi-Agent API Usage Guide

## Overview
This guide explains how to interact with the **Customer Support Multi-Agent OpenEnv** FastAPI server. The server provides REST endpoints for:
- **OpenEnv Core**: Environment reset/step/state for RL training
- **Multi-Agent System**: Agent bidding, negotiation, and resolution
- **Monitoring & Metrics**: Agent status, performance tracking, and configuration inspection
- **Specialization Management**: View agent roles and training details

## Table of Contents
1. [Starting the Server](#starting-the-server)
2. [Core OpenEnv Endpoints](#core-openenv-endpoints)
3. [Multi-Agent Negotiation Endpoints](#multi-agent-negotiation-endpoints)
4. [Agent Monitoring & Metrics](#agent-monitoring--metrics)
5. [Complete Episode Workflow](#complete-episode-workflow)
6. [Detailed Examples](#detailed-examples)
7. [Python Client Examples](#python-client-examples)

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

## Core OpenEnv Endpoints

These endpoints implement the standard OpenEnv interface for RL training loops.

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

### 2. **POST /reset** - Initialize New Episode
Load a random customer support ticket and initialize a new negotiation episode.

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
    "ticket_id": "TICKET-12345",
    "message": "I was charged twice for my subscription this month. Please help!",
    "severity": "high",
    "category": "billing",
    "status": "open",
    "phase": "bidding",
    "reward": 0.0,
    "done": false,
    "resolution_message": "New episode initialized. Waiting for agent bids..."
  },
  "reward": 0.0,
  "done": false
}
```

---

### 3. **POST /step** - Submit Agent Action
Execute an action in the current negotiation phase.

**Generic Format:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "phase": "CURRENT_PHASE",
      "agent": "AGENT_NAME",
      "... phase-specific params ..."
    }
  }'
```

---

### 4. **GET /state** - Get Current Episode State
Retrieve the current state of the ongoing negotiation episode.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/state"
```

**Response (200 OK):**
```json
{
  "episode_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "step_count": 2,
  "current_phase": "bidding",
  "ticket_id": "TICKET-12345",
  "agent_bids": [
    {"agent": "technical", "confidence": 0.85, "timestamp": 1713960000},
    {"agent": "billing", "confidence": 0.92, "timestamp": 1713960001}
  ],
  "team_score": 0.0,
  "done": false
}
```

---

### 5. **GET /schema** - Get Action/Observation Schemas
Get JSON schemas for valid actions and observations.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/schema"
```

**Response (200 OK):**
```json
{
  "action_schema": {
    "type": "object",
    "properties": {
      "phase": {"type": "string"},
      "agent": {"type": "string"},
      "confidence": {"type": "number"},
      "solution": {"type": "string"}
    }
  },
  "observation_schema": {
    "type": "object",
    "properties": {
      "ticket_id": {"type": "string"},
      "message": {"type": "string"},
      "phase": {"type": "string"},
      "reward": {"type": "number"},
      "done": {"type": "boolean"}
    }
  }
}
```

---

## Multi-Agent Negotiation Endpoints

These endpoints provide visibility into the multi-agent negotiation system.

### 6. **GET /api/agents/status** - Agent Status (ROUND 2)
Get status of all 4 specialized agents.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/api/agents/status"
```

**Response (200 OK):**
```json
{
  "agents": [
    {
      "name": "technical",
      "status": "ready",
      "model": "unsloth/Llama-3.2-1B-Instruct",
      "mode": "inference",
      "role": "Technical Support Specialist"
    },
    {
      "name": "billing",
      "status": "ready",
      "model": "unsloth/Llama-3.2-1B-Instruct",
      "mode": "inference",
      "role": "Billing Specialist"
    },
    {
      "name": "account",
      "status": "ready",
      "model": "unsloth/Llama-3.2-1B-Instruct",
      "mode": "inference",
      "role": "Account Manager"
    },
    {
      "name": "manager",
      "status": "ready",
      "model": "unsloth/Llama-3.2-1B-Instruct",
      "mode": "inference",
      "role": "Quality Manager"
    }
  ]
}
```

---

### 7. **GET /api/agents/metrics** - Agent Performance Metrics (ROUND 2)
Get training and performance metrics for each agent.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/api/agents/metrics"
```

**Response (200 OK):**
```json
{
  "metrics": [
    {
      "agent": "technical",
      "episodes": 156,
      "avg_reward": 0.78,
      "success_rate": 0.82,
      "avg_confidence": 0.73,
      "training_status": "completed"
    },
    {
      "agent": "billing",
      "episodes": 142,
      "avg_reward": 0.81,
      "success_rate": 0.87,
      "avg_confidence": 0.75,
      "training_status": "completed"
    },
    {
      "agent": "account",
      "episodes": 138,
      "avg_reward": 0.76,
      "success_rate": 0.79,
      "avg_confidence": 0.71,
      "training_status": "completed"
    },
    {
      "agent": "manager",
      "episodes": 145,
      "avg_reward": 0.79,
      "success_rate": 0.85,
      "avg_confidence": 0.74,
      "training_status": "completed"
    }
  ]
}
```

---

### 8. **GET /api/agents/{agent_name}/specialization** - Agent Specialization (ROUND 2)
Get detailed specialization info for a specific agent.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/api/agents/billing/specialization"
```

**Response (200 OK):**
```json
{
  "agent_name": "billing",
  "specializes_in": [
    "Payment issues",
    "Subscription management",
    "Refund processing",
    "Fraud detection"
  ],
  "model": "unsloth/Llama-3.2-1B-Instruct",
  "training_examples": 100,
  "training_method": "TRL GRPO"
}
```

---

### 9. **POST /api/agents/bid** - Manual Bid Submission (ROUND 2)
Manually submit a bid for testing purposes (bypasses agent inference).

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/agents/bid" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "billing",
    "confidence": 0.92,
    "reasoning": "High confidence in duplicate charge resolution"
  }'
```

**Response (200 OK):**
```json
{
  "agent": "billing",
  "confidence": 0.92,
  "reasoning": "High confidence in duplicate charge resolution",
  "timestamp": "2026-04-23T10:30:45.123456",
  "status": "bid_recorded"
}
```

---

### 10. **GET /api/episodes/{episode_id}/agent-decisions** - Episode Details (ROUND 2)
Get detailed per-agent decisions and rewards for a specific episode.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/api/episodes/ep-001/agent-decisions"
```

**Response (200 OK):**
```json
{
  "episode_id": "ep-001",
  "ticket": {
    "id": "TICKET-12345",
    "message": "I was charged twice",
    "severity": "high",
    "category": "billing"
  },
  "phase_1_bidding": {
    "technical": {"confidence": 0.65, "reward": -0.1},
    "billing": {"confidence": 0.92, "reward": 0.2},
    "account": {"confidence": 0.40, "reward": -0.2},
    "manager": {"confidence": 0.78, "reward": 0.1}
  },
  "phase_2_execution": {
    "winner": "billing",
    "solution": "refund_duplicate_charge",
    "reward": 0.2
  },
  "phase_3_resolution": {
    "manager_decision": "close",
    "quality_reward": 0.2,
    "team_success_bonus": 0.2
  },
  "total_episode_reward": 1.0,
  "anti_hacking_safeguards_triggered": []
}
```

---

### 11. **GET /api/environment/config** - Environment Configuration (ROUND 2)
Get the 11-signal reward structure and anti-hacking safeguards.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/api/environment/config"
```

**Response (200 OK):**
```json
{
  "reward_functions": {
    "positive_signals": [
      {"name": "correct_specialist_bid", "value": 0.2, "description": "Agent bid for correct specialist"},
      {"name": "correct_solution", "value": 0.2, "description": "Solution matches ground truth"},
      {"name": "appropriate_confidence", "value": 0.1, "description": "Confidence score calibrated well"},
      {"name": "solution_format", "value": 0.05, "description": "Solution in correct format"},
      {"name": "team_success_bonus", "value": 0.2, "description": "Team successfully resolved ticket"}
    ],
    "negative_signals": [
      {"name": "wrong_specialist", "value": -0.2, "description": "Non-specialist agent won bid"},
      {"name": "wrong_solution", "value": -0.2, "description": "Solution does not match ground truth"},
      {"name": "overconfident", "value": -0.1, "description": "Agent bid > actual accuracy"},
      {"name": "team_failure", "value": -0.1, "description": "Team failed to resolve"},
      {"name": "invalid_bid", "value": -0.05, "description": "Bid outside valid range [0,1]"},
      {"name": "timeout_penalty", "value": -0.15, "description": "Episode exceeded MAX_STEPS"}
    ]
  },
  "anti_hacking_safeguards": {
    "bid_range_validation": "[0.0, 1.0] with penalty for violations",
    "bid_history_logging": "All bids logged with timestamp, agent, confidence, ticket_id",
    "max_steps_per_episode": 10,
    "invalid_bid_penalty": -0.05
  }
}
```

---

## Complete Episode Workflow

An episode has **3 mandatory phases**. Multiple agents act within each phase:

```
┌──────────────────────────┐
│  PHASE 0: RESET          │  Load random ticket
└────┬─────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│  PHASE 1: BIDDING                                │
│  All 4 agents submit confidence (0.0 - 1.0)     │
│  Rewards computed for calibration accuracy       │
│  Winner = agent with highest valid bid           │
└────┬─────────────────────────────────────────────┘
     │ Reward: +0.2 (correct specialist)
     │         -0.2 (wrong specialist)
     │         +0.1 (appropriate confidence)
     ▼
┌──────────────────────────────────────────────────┐
│  PHASE 2: EXECUTION                              │
│  Winning agent proposes solution                 │
│  Evaluated against ground truth policy matrix    │
└────┬─────────────────────────────────────────────┘
     │ Reward: +0.2 (correct solution)
     │         -0.2 (wrong solution)
     │         +0.05 (format correct)
     ▼
┌──────────────────────────────────────────────────┐
│  PHASE 3: RESOLUTION                             │
│  Manager agent evaluates & makes final decision  │
│  Team success bonus applied if ticket resolved   │
└────┬─────────────────────────────────────────────┘
     │ Reward: +0.2 (team success bonus)
     │         -0.1 (team failure penalty)
     │         -0.15 (timeout)
     ▼
  DONE=true
  TOTAL EPISODE REWARD: 0.5 - 1.0 (varies by agent decisions)
```

### 11-Signal Reward Design

The negotiation episode uses **11 independent reward signals** to prevent reward hacking:
- **5 positive signals**: Incentivize correct decisions
- **6 negative signals**: Penalize poor decisions or violations
- **Anti-gaming**: Bid validation, history logging, timeout limits

---

## Detailed Examples

### Complete Example: Full Multi-Agent Negotiation Episode

#### **Phase 0: Reset Environment**
```bash
curl -X POST "http://127.0.0.1:8000/reset" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response:**
```json
{
  "observation": {
    "ticket_id": "TICKET-67890",
    "message": "I was charged twice for my subscription this month. Please help!",
    "severity": "high",
    "category": "billing",
    "status": "open",
    "phase": "bidding",
    "done": false,
    "resolution_message": "Episode initialized. Waiting for bids from all 4 agents..."
  },
  "reward": 0.0,
  "done": false
}
```

---

#### **Phase 1.1: Technical Agent Submits Bid**
Technical agent assesses confidence in resolving this billing ticket.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "phase": "bidding",
      "agent": "technical",
      "confidence": 0.45,
      "reasoning": "Low confidence - this is not a technical issue"
    }
  }'
```

**Response:**
```json
{
  "observation": {
    "phase": "bidding",
    "bids_received": 1,
    "latest_bid": {
      "agent": "technical",
      "confidence": 0.45,
      "reward_for_calibration": -0.05
    },
    "resolution_message": "Technical agent bid: 0.45 (-0.05 penalty: inappropriate confidence)"
  },
  "reward": -0.05,
  "done": false
}
```

---

#### **Phase 1.2: Billing Agent Submits Bid (WINNING)**
Billing agent is confident in handling duplicate charges.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "phase": "bidding",
      "agent": "billing",
      "confidence": 0.92,
      "reasoning": "High confidence - duplicate charge is a billing issue"
    }
  }'
```

**Response:**
```json
{
  "observation": {
    "phase": "bidding",
    "bids_received": 2,
    "latest_bid": {
      "agent": "billing",
      "confidence": 0.92,
      "reward_for_calibration": +0.2
    },
    "resolution_message": "Billing agent bid: 0.92 (+0.2 reward: correct specialist)"
  },
  "reward": 0.2,
  "done": false
}
```

---

#### **Phase 1.3: Account Agent Submits Bid**
Account agent is less confident.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "phase": "bidding",
      "agent": "account",
      "confidence": 0.35,
      "reasoning": "Low confidence - not account-related"
    }
  }'
```

**Response:**
```json
{
  "observation": {
    "phase": "bidding",
    "bids_received": 3,
    "latest_bid": {
      "agent": "account",
      "confidence": 0.35,
      "reward_for_calibration": 0.0
    },
    "resolution_message": "Account agent bid: 0.35 (appropriate for non-account issue)"
  },
  "reward": 0.0,
  "done": false
}
```

---

#### **Phase 1.4: Manager Agent Submits Bid**
Manager evaluates overall team performance.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "phase": "bidding",
      "agent": "manager",
      "confidence": 0.78,
      "reasoning": "Billing agent is strong candidate for this ticket"
    }
  }'
```

**Response:**
```json
{
  "observation": {
    "phase": "execution",
    "winner": "billing",
    "highest_bid": 0.92,
    "resolution_message": "Bidding complete. Winner: BILLING (0.92). Moving to execution phase...",
    "bids_summary": {
      "technical": 0.45,
      "billing": 0.92,
      "account": 0.35,
      "manager": 0.78
    }
  },
  "reward": 0.0,
  "done": false
}
```

---

#### **Phase 2: Execution (Billing Agent Proposes Solution)**
Winning agent (billing) now proposes a solution.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "phase": "execution",
      "agent": "billing",
      "solution": "refund_duplicate_charge",
      "category": "duplicate_charge"
    }
  }'
```

**Response:**
```json
{
  "observation": {
    "phase": "resolution",
    "proposed_solution": "refund_duplicate_charge",
    "solution_valid": true,
    "solution_reward": 0.2,
    "resolution_message": "Billing solution accepted (+0.2). Moving to manager resolution..."
  },
  "reward": 0.2,
  "done": false
}
```

---

#### **Phase 3: Resolution (Manager Evaluates & Closes)**
Manager makes final decision on ticket resolution.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "phase": "resolution",
      "agent": "manager",
      "decision": "close",
      "quality_score": 0.95
    }
  }'
```

**Response:**
```json
{
  "observation": {
    "phase": "resolution",
    "manager_decision": "close",
    "quality_reward": 0.2,
    "team_success_bonus": 0.2,
    "episode_reward": 0.8,
    "done": true,
    "resolution_message": "Ticket resolved successfully! Total reward: 0.8/1.0 = 80%"
  },
  "reward": 0.2,
  "done": true
}
```

---

## Python Client Examples

### Example 1: Multi-Agent Negotiation Client
```python
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Phase 0: Reset
print("=== PHASE 0: RESET ===")
reset_resp = requests.post(f"{BASE_URL}/reset", json={})
observation = reset_resp.json()["observation"]
print(f"Ticket: {observation['message']}")
print(f"Category: {observation['category']}")

# Phase 1: Bidding (all agents submit bids)
print("\n=== PHASE 1: BIDDING ===")
agents = [
    ("technical", 0.45),
    ("billing", 0.92),
    ("account", 0.35),
    ("manager", 0.78)
]

cumulative_reward = 0
for agent_name, confidence in agents:
    action = {
        "action": {
            "phase": "bidding",
            "agent": agent_name,
            "confidence": confidence
        }
    }
    resp = requests.post(f"{BASE_URL}/step", json=action)
    reward = resp.json()["reward"]
    cumulative_reward += reward
    print(f"{agent_name:12} | Bid: {confidence} | Reward: {reward:+.2f}")

print(f"\nBidding phase total reward: {cumulative_reward}")

# Phase 2: Execution (winner proposes solution)
print("\n=== PHASE 2: EXECUTION ===")
action = {
    "action": {
        "phase": "execution",
        "agent": "billing",
        "solution": "refund_duplicate_charge",
        "category": "duplicate_charge"
    }
}
resp = requests.post(f"{BASE_URL}/step", json=action)
exec_reward = resp.json()["reward"]
cumulative_reward += exec_reward
print(f"Execution reward: {exec_reward:+.2f}")

# Phase 3: Resolution (manager closes)
print("\n=== PHASE 3: RESOLUTION ===")
action = {
    "action": {
        "phase": "resolution",
        "agent": "manager",
        "decision": "close",
        "quality_score": 0.95
    }
}
resp = requests.post(f"{BASE_URL}/step", json=action)
result = resp.json()
res_reward = result["reward"]
cumulative_reward += res_reward
print(f"Resolution reward: {res_reward:+.2f}")
print(f"\n✅ Episode Complete!")
print(f"Final Episode Reward: {cumulative_reward:.2f}")
print(f"Done: {result['done']}")
```

### Example 2: Monitor Agent Metrics
```python
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Get all agent metrics
resp = requests.get(f"{BASE_URL}/api/agents/metrics")
metrics = resp.json()["metrics"]

print("=== AGENT PERFORMANCE METRICS ===\n")
print(f"{'Agent':<15} {'Episodes':<12} {'Avg Reward':<12} {'Success %':<12}")
print("-" * 51)
for agent_metrics in metrics:
    agent = agent_metrics["agent"]
    episodes = agent_metrics["episodes"]
    avg_reward = agent_metrics["avg_reward"]
    success_rate = agent_metrics["success_rate"] * 100
    print(f"{agent:<15} {episodes:<12} {avg_reward:<12.2f} {success_rate:<12.1f}%")
```

### Example 3: Get Specific Agent Specialization
```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# Get billing agent specialization
resp = requests.get(f"{BASE_URL}/api/agents/billing/specialization")
spec = resp.json()

print("=== BILLING AGENT SPECIALIZATION ===\n")
print(f"Agent: {spec['agent_name']}")
print(f"Model: {spec['model']}")
print(f"Training Examples: {spec['training_examples']}")
print(f"Training Method: {spec['training_method']}")
print(f"\nSpecializes in:")
for specialty in spec['specializes_in']:
    print(f"  • {specialty}")
```

---
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
