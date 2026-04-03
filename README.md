---
title: Customer Support OpenEnv Environment
emoji:  🤖
colorFrom: green
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

## Customer Support OpenEnv Environment

This project is a simple, realistic environment for training AI agents to handle customer support tickets. Agents learn to classify issues, pick solutions, and decide when to escalate.

---

### How to Run

1. **Activate virtual environment:**
   ```bash
   # Windows PowerShell
   .\openenv\Scripts\Activate.ps1
   
   # macOS/Linux
   source openenv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server:**
   ```bash
   python -m uvicorn my_env.server.app:app --reload --port 8000
   ```

4. **Run tests:**
   ```bash
   python tests/final_comprehensive_test.py
   ```

5. **Test a single scenario:**
   ```bash
   python improved_agent_training.py
   # or run demo.py for a quick test
   ```

---

### How It Works (Agent Workflow)

1. Agent receives a support ticket (text + severity)
2. Agent classifies the issue (billing, account, bug, feature)
3. Agent chooses a solution (category + action)
4. Agent decides if escalation is needed
5. Agent closes the ticket

At each step, the agent gets feedback and a reward. The goal is to maximize the score (max 1.0 per ticket).

---

### Project Structure

```
openenv-hackathon-project/
│
├── Configuration & Documentation
│   ├── README.md                          # Main project documentation
│   ├── API_USAGE_GUIDE.md                 # Complete API endpoint guide with examples
│   ├── pyproject.toml                     # Python project metadata
│   ├── requirements.txt                   # Python dependencies
│   ├── Dockerfile                         # Docker container configuration
│   ├── .dockerignore                      # Docker build exclusions
│   ├── .gitignore                         # Git exclusions
│   ├── .hfignore                          # HuggingFace exclusions
│   ├── openenv.yaml                       # Main OpenEnv spec (root level)
│   └── uv.lock                            # Dependency lock file
│
├── Root-Level Scripts & Entry Points
│   ├── inference.py                       # Hackathon inference script (logging format: [START]/[STEP]/[END])
│   ├── client.py                          # HTTP client for testing endpoints
│   ├── models.py                          # Pydantic models (Action, Observation)
│   ├── demo.py                            # Quick demo of environment
│   ├── improved_agent_training.py         # Advanced agent training example
│   └── __init__.py                        # Python package marker
│
├── my_env/                                # Main OpenEnv environment package
│   ├── __init__.py
│   ├── agents.py                          # Agent baseline implementations
│   ├── openenv.yaml                       # OpenEnv spec (my_env level, optional override)
│   │
│   └── server/                            # FastAPI server & core environment logic
│       ├── __init__.py
│       ├── app.py                         # FastAPI server with 6 endpoints (/reset, /step, /state, /schema, /health, /tasks)
│       ├── customer_support_environment.py # Core RL environment class
│       │
│       ├── data/                          # Ticket data and utilities
│       │   ├── __init__.py
│       │   └── tickets.py                 # Ticket dataset, categories, solutions
│       │
│       └── logic/                         # Reward and resolution logic
│           ├── __init__.py
│           └── ticket_resolver.py         # Reward calculation, solution validation
│
├── tests/                                 # Comprehensive test suite
│   ├── final_comprehensive_test.py        # Main anti-cheating test suite
│   ├── test_complete_walkthrough.py       # Full episode workflow test
│

```

**Key Files at a Glance:**
- **API Server**: `my_env/server/app.py` (FastAPI with persistent environment)
- **Core Logic**: `my_env/server/customer_support_environment.py` + `my_env/server/logic/ticket_resolver.py`
- **Inference**: `inference.py` (for hackathon submission with proper logging)
- **Testing**: `tests/final_comprehensive_test.py` (anti-cheating measures)
- **API Docs**: `API_USAGE_GUIDE.md` (complete examples for using endpoints)
---

### Key Features

- Realistic customer support workflow (classify, solve, escalate, close)
- 3 difficulty levels (Easy, Medium, Hard)
- Step-by-step feedback for learning
- OpenEnv and Gymnasium compatible
- Ready for HuggingFace Spaces and Docker

---

### Example Agent Loop

```python
obs = env.reset()
while not obs.done:
    action = agent.act(obs)
    obs = env.step(action)
    print(obs.reward, obs.resolution_message)
```

---

### API Endpoints

Endpoint | Method | Status | Description
---------|--------|--------|-------
/reset | **POST** | 200 | Initialize new episode and load a random ticket
/step | **POST** | 200 | Send action (classify, choose solution, escalate, close)
/state | **GET** | 200 | Get current episode state
/health | **GET** | 200 | Health check endpoint
/schema | **GET** | 200 | Get action and observation schemas
/tasks | **POST** | 200 | List available task difficulties (Easy/Medium/Hard)

---

###  API Usage Guide

**For detailed examples, request/response formats, and complete workflows, see:**

 **[API_USAGE_GUIDE.md](./API_USAGE_GUIDE.md)** ← **START HERE**

This guide includes:
- ✅ Step-by-step curl examples for each endpoint
- ✅ Complete episode workflow with actual response bodies
- ✅ Valid action types and formats
- ✅ Reward structure explanation
- ✅ Python code examples
- ✅ Error handling & troubleshooting
- ✅ OpenAPI Swagger UI access

---

### Quick Start: Test the API Locally

```bash
# Terminal 1: Start server
python -m uvicorn my_env.server.app:app --port 8000

# Terminal 2: Run full episode test
python -c "
import requests
import json

# Reset
r = requests.post('http://127.0.0.1:8000/reset', json={})
print('1️  Reset:', r.status_code, '200')

# Step 1: Classify
r = requests.post('http://127.0.0.1:8000/step', json={
    'action': {'action_type': 'classify_issue', 'classification': 'billing'}
})
print('2️  Classify:', r.status_code, '200')

# Step 2: Solution
r = requests.post('http://127.0.0.1:8000/step', json={
    'action': {'action_type': 'choose_solution', 'category': 'duplicate_charge', 'solution': 'refund_duplicate_charge'}
})
print('3️  Solution:', r.status_code, '200')

# Step 3: Escalation
r = requests.post('http://127.0.0.1:8000/step', json={
    'action': {'action_type': 'escalate_decision', 'should_escalate': False}
})
print('4️  Escalation:', r.status_code, '200')

# Step 4: Close
r = requests.post('http://127.0.0.1:8000/step', json={
    'action': {'action_type': 'close_ticket'}
})
data = r.json()
print('5️  Close:', r.status_code, f'Score: {data[\"observation\"][\"episode_score\"]}', '200')
"
```

---

For more details, see [API_USAGE_GUIDE.md](./API_USAGE_GUIDE.md) for complete endpoint documentation.
- 2fa -> reset_2fa, send_recovery_codes
- security -> escalate_security, freeze_account

**Bug Issues:**
- app_crash -> update_app_version, clear_cache_restart
- ui_glitch -> clear_cache_restart, escalate_engineering
- missing_data -> sync_data, escalate_engineering
- critical -> escalate_engineering, create_hotfix

**Feature Issues:**
- how_to -> explain_feature, send_tutorial
- capability -> escalate_sales, enable_feature_trial
- api -> escalate_sales, schedule_consultation
- custom -> escalate_sales, create_feature_request

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
2. **Phase 1**: Agent classifies type (guess: "billing" -> WRONG)
   - Feedback: `[FAIL] INCORRECT. Correct answer: 'bug'` 
   - Reward: 0.0, episode_reward: 0.0
3. **Phase 2**: Agent sees feedback, chooses solution category
   - On next similar ticket, remembers "database -> bug"
   - Guesses "bug" (now [OK])
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
**Hugging Face Space URL**
```
visit: https://huggingface.co/spaces/RavichandraNayakar/customer_support_env
```
**If HF Space shows blank screen:** Use this direct link instead:
```
visit: https://ravichandranayakar-customer-support-env.hf.space/web
```

### Docker

```bash
docker build -t my-env .
docker run -p 8000:8000 my-env
```