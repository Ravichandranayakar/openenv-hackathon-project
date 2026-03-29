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

## Customer Support OpenEnv Environment

This project is a simple, realistic environment for training AI agents to handle customer support tickets. Agents learn to classify issues, pick solutions, and decide when to escalate.

---

### How to Run

1. **Install dependencies:**
   ```bash
   pip install -e my_env
   ```
2. **Start the server:**
   ```bash
   python -m uvicorn my_env.server.app:app --reload --port 8000
   ```
3. **Train or test an agent:**
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
├── README.md
├── Dockerfile
├── pyproject.toml
├── openenv.yaml
├── demo.py
├── improved_agent_training.py
├── my_env/
│   ├── __init__.py
│   ├── agents.py
│   ├── client.py
│   ├── models.py
│   └── server/
│       ├── app.py
│       ├── customer_support_environment.py
│       └── ...
└── tests/
    └── test_cheating_comparison.py
```

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

- `POST /reset` – Start new episode
- `POST /step` – Take an action
- `GET /state` – Get current state
- `GET /health` – Health check

---

For more details, see the code and comments in `my_env/`.
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
observation.classification_reward    #  +0.2 if correct, +0.0 if wrong

observation.category           # Agent's chosen category
observation.correct_category   # True/False  
observation.solution           # Agent's chosen solution
observation.correct_solution   # True/False
observation.solution_reward    # +0.3 if correct, +0.0 if wrong

observation.escalation_decision # Agent's true/false decision
observation.correct_escalation  # True/False
observation.escalation_reward   # +0.3 if correct, +0.0 if wrong

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
[FAIL] INCORRECT. Correct answer: 'bug' (Learn: 'bug' issues are technical problems)
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
- duplicate_charge -> refund_duplicate_charge, investigate_fraud
- wrong_amount -> correct_invoice, refund_difference
- subscription_issue -> cancel_subscription, update_subscription
- fraud -> escalate_security, freeze_account

**Account Issues:**
- password -> reset_password_link, send_recovery_email
- email -> update_email_settings, verify_new_email
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

**If HF Space shows blank screen:** Use this direct link instead:
```
https://ravichandranayakar-customer-support-env.hf.space/web/
```

Then visit: https://huggingface.co/spaces/RavichandraNayakar/my_env

### Docker

```bash
docker build -t my-env .
docker run -p 8000:8000 my-env
```


## Project Structure

The repository is organized for clarity and ease of review:

```
openenv-hackathon-project/
├── .gitignore
├── .dockerignore
├── README.md
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── openenv.yaml
├── demo.py
├── improved_agent_training.py
├── my_env/
│   ├── __init__.py
│   ├── agents.py
│   ├── client.py
│   ├── models.py
│   └── server/
│       ├── __init__.py
│       ├── app.py
│       ├── customer_support_environment.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── tickets.py
│       └── logic/
│           ├── __init__.py
│           └── ticket_resolver.py
└── tests/
  ├── test_cheating_comparison.py
  ├── test_agent_learning.py
  ├── test_complete_walkthrough.py
  ├── test_minimal_agent.py
  └── final_comprehensive_test.py
```

- **Root**: Submission files, configuration, and documentation.
- **my_env/**: Single clean package with all environment logic, agent, client, and models.
- **tests/**: All test files, tracked in git (not ignored).

No duplicate files, no build artifacts, and no unnecessary outputs are committed. This structure is optimized for OpenEnv, HuggingFace Spaces, and easy review by judges.

