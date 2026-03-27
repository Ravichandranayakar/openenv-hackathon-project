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

Train agents to handle customer support tickets. They learn to classify issues, find solutions, and decide when to escalate.

**What you get:** 14 realistic tickets, ground truth feedback, deterministic grading, Docker-ready.

## Quick Start

```bash
pip install -e .
python -m uvicorn my_env.server.app:app --port 8000
python demo.py --episodes 5 --task 1
```

That's it.

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

---

## Code Guide

Want to dig into the code? Here's where to look:

- [models.py](my_env/models.py) - Action and observation types
- [customer_support_environment.py](my_env/server/customer_support_environment.py) - Core 4-step logic
- [baseline_agent.py](my_env/baseline_agent.py) - Example agent
- [tickets.py](my_env/server/data/tickets.py) - All 14 tickets + solutions
- [ticket_resolver.py](my_env/server/logic/ticket_resolver.py) - Grading logic

---

## Running Locally

**Start the server:**
```bash
pip install -e .
python -m uvicorn my_env.server.app:app --port 8000
```

**Run an agent:**
```bash
python demo.py --episodes 5 --task 1
```

---

## Deploy

**HuggingFace Spaces:**
```bash
openenv push --name YourUsername/support-env --token <hf_token>
```

**Docker:**
```bash
docker build -t my-env .
docker run -p 8000:8000 my-env
```

## How It Works

Agent handles a ticket in 4 steps:

1. **Classify** - What type? (billing/account/bug/feature) → +0.2 if right
2. **Solve** - What's the fix? (category + solution) → +0.3 if right
3. **Escalate?** - Does it need a human? (true/false) → +0.3 if right
4. **Close** - Ticket done → +0.2 if right

**Max score per ticket: 1.0**

When agent guesses wrong, it sees the right answer. Next similar ticket, agent knows better. That's how learning works.

## 3 Tasks

- **Task 1 (Easy):** Simple tickets - refunds, password resets
- **Task 2 (Medium):** Mixed types with some escalation 
- **Task 3 (Hard):** Complex security issues, frequent escalation

Run any task: `python demo.py --task 1` (or 2, or 3)

## What Agents Can Solve

14 pre-made tickets covering:

- **Billing:** Charges, refunds, subscriptions, fraud
- **Account:** Passwords, emails, 2FA, security
- **Bugs:** App crashes, UI issues, missing data
- **Features:** How-tos, capabilities, API questions

Full list in [my_env/server/data/tickets.py](my_env/server/data/tickets.py)


