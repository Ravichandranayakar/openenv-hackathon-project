---
title: Customer Support OpenEnv Environment
emoji:  🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /
tags:
  - openenv
  - rl
  - customer-support
  - multi-agent
---

# Autonomous Customer Support Multi-Agent Network

This project demonstrates a Multi-Agent Enterprise Customer Support Network built on the OpenEnv framework. It utilizes a multi-agent negotiation system where four specialized LLM agents (Technical, Billing, Account, Manager) collaborate and bid on incoming customer tickets to resolve complex logic edge cases autonomously.

The core underlying methodology uses reinforcement learning to fine-tune a single LLM to accurately assess its capabilities across different specializations, leveraging a comprehensive 11-signal reward policy to prevent reward hacking.

**Core Technology Stack:**
- **Environment**: OpenEnv (FastAPI, Python)
- **RL Training**: TRL (Transformers Reinforcement Learning) via GRPO
- **LLM Base**: Llama-3.2-1B-Instruct
- **Optimization**: Unsloth (4-bit quantization)

---

## How It Works: 4-Agent Negotiation System

When a support ticket arrives, the environment orchestrates a **3-Phase Negotiation Protocol**:

### Phase 1: Bidding 
All 4 specialized agents independently analyze the ticket and submit a **confidence score (0.0-1.0)**:
- **Technical Agent** (0.95 on database crashes, 0.15 on billing)
- **Billing Agent** (0.92 on duplicate charges, 0.30 on API issues)
- **Account Agent** (0.88 on password resets, 0.25 on payment issues)
- **Manager Agent** (oversees bidding, enforces rules, validates escalations)

**Rewards:** +0.2 for correct specialization, +0.1 for calibrated confidence, -0.2 for wrong specialist

### Phase 2: Execution
The **highest valid bid wins**. Winning agent proposes a solution against the enterprise policy matrix:
- Billing issue? Propose refund/escalation
- Technical crash? Propose restart/sync/engineering escalation
- Account security? Propose 2FA reset/account freeze

**Rewards:** +0.2 for correct solution, -0.2 for wrong solution, +0.05 for format compliance

### Phase 3: Resolution 
Manager Agent performs final quality assurance:
- Is the ticket severity low/medium? → CLOSE
- Is the ticket critical/complex? → ESCALATE to human

**Rewards:** +0.2 team success bonus, -0.1 team failure penalty, -0.15 for timeouts

---

## Quick Start 

### 1️ **Setup (5 minutes)**

**Clone & Install:**
```bash
cd openenv-hackathon-project
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

**Check dependencies:**
```bash
python -c "import openenv, fastapi, pydantic; print('✅ Core dependencies OK')"
```

### 2️ **Start Environment Server (Terminal 1)** 

```bash
python -m uvicorn my_env.server.app:app --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

  Server ready at: `http://localhost:8000`

### 3️ **Test Environment (Terminal 2)**

**CPU Quick Test** (5 episodes, ~10 min):
```bash
python my_env/pytorch/training/trl_grpo_trainer_cpu.py --episodes 5
```

**Expected output:**
```
Episode 1: Ticket: I was charged twice → Action: billing | Reward: +1.0 ✓
Episode 2: Ticket: App crashes → Action: technical | Reward: +0.8 ✓
...
TEST SUMMARY: Episodes: 5, Avg Reward: +0.75, Accuracy: 75%
```

  Environment validation complete!

### 4️ **Production Training (GPU Required)**

**Install extended dependencies:**
```bash
pip install trl unsloth transformers datasets accelerate peft safetensors torch
```

**Download model** (first run auto-downloads ~3-5 GB):
```bash
python -c "from unsloth import FastLanguageModel; \
FastLanguageModel.from_pretrained('unsloth/Llama-3.2-1B-Instruct'); \
print('Model cached')"
```

**Run 4-agent training** (20-45 min on GPU):
```bash
python scripts/train_multi_agent.py
```

**Training creates:**
- `checkpoints_multi_agent/` → Fine-tuned agent models
- `results/metrics.json` → Loss curves, reward progression
- Trained agents ready for inference

---

## Installation & Dependencies

### Core Stack (Environment Only)
```
fastapi==0.104.0          # Web API server
pydantic==2.5.0           # Data validation
uvicorn==0.24.0           # ASGI server
requests==2.31.0          # HTTP client
gradio==4.26.0            # UI dashboard (optional)
```

### Training Stack (GPU Required)
```
torch==2.1.0              # PyTorch core
transformers==4.35.0      # Hugging Face models
datasets==2.14.0          # Data loading
trl==0.7.4               # TRL GRPO trainer
unsloth==2.0.0           # 4-bit quantization
peft==0.7.0              # LoRA fine-tuning
safetensors==0.4.0       # Model serialization
accelerate==0.25.0       # Distributed training
bitsandbytes==0.41.0     # 4-bit optimizations
```

### Large Files (Auto-Downloaded)
| File | Size | Source | Location |
|------|------|--------|----------|
| Llama-3.2-1B-Instruct | ~3-5 GB | Hugging Face Hub | `~/.cache/huggingface/hub/` |
| Training datasets | ~50 MB | Generated synthetically | `my_env/server/data/` |
| Model checkpoints | ~1-2 GB/agent | Generated during training | `checkpoints_multi_agent/` |

---

## System Requirements

| Task | CPU | RAM | GPU | Storage |
|------|-----|-----|-----|---------|
| **Environment API** | 2 cores | 4 GB | Optional | 2 GB |
| **CPU Testing** | 4 cores | 8 GB | No | 2 GB |
| **GPU Training** | 8 cores | 16 GB | RTX 3060+ | 50 GB |
| **Full Pipeline** | 16 cores | 32 GB | A100/H100 | 100 GB |

---

## File Size & Download Breakdown

**Initial Setup (without models):**
```
openenv-hackathon-project/
  ├── Source code: ~5 MB
  ├── Dependencies (pip install): ~500 MB
  └── Data files: ~50 MB
  TOTAL: ~600 MB
```

**After Training (with models):**
```
  + Llama-3.2-1B model: ~3-5 GB (Hugging Face cache)
  + 4 trained agent checkpoints: ~1-2 GB
  + Training logs & metrics: ~100 MB
  TOTAL: ~5-8 GB (mostly cached HF model)
```

---

## The 4 Specialized Agents

| Agent | Specialization | Example Tasks |
|-------|---|---|
| **Technical**  | App crashes, data sync, API issues | "My app keeps crashing" → Propose update/restart/sync |
| **Billing**  | Duplicate charges, refunds, fraud | "Charged twice" → Propose refund/escalate fraud |
| **Account**  | Password, 2FA, security breaches | "Can't login" → Propose reset/2FA/account freeze |
| **Manager**  | Quality assurance, escalation routing | "Is this critical?" → Close or escalate to human |

---

## The 11-Signal Reward System (Anti-Hacking)

###  Positive Signals (+2.0 total max)
1. **Correct Specialist Bid** (+0.2) - Agent with highest bid matches ticket category
2. **Correct Solution** (+0.2) - Solution matches ground-truth policy matrix
3. **Appropriate Confidence** (+0.1) - Bid calibrated to actual accuracy
4. **Solution Format** (+0.05) - Response structure matches expectations
5. **Team Success Bonus** (+0.2) - All agents rewarded if ticket resolves

###  Negative Signals (-0.90 total max)
6. **Wrong Specialist** (-0.2) - Non-expert agent won the bid
7. **Wrong Solution** (-0.2) - Solution violates policy
8. **Overconfident** (-0.1) - Bid > actual accuracy (calibration penalty)
9. **Team Failure** (-0.1) - Team failed to resolve
10. **Invalid Bid** (-0.05) - Confidence outside [0.0, 1.0]
11. **Timeout** (-0.15) - Episode exceeded 10 steps

**Design Goal:** No single signal dominates. Agents must learn nuanced behavior: when to bid high (specialization match), when to defer (low confidence), and how to collaborate (team bonuses).

---

## How to Run

---

## Multi-Agent Negotiation Features

✅ **4 Specialized LLM Agents** with TRL GRPO fine-tuning
✅ **11-Signal Reward System** for anti-hacking safeguards
✅ **3-Phase Negotiation Protocol** (Bidding → Execution → Resolution)
✅ **45+ Real-World Support Scenarios** (3 difficulty levels)
✅ **OpenEnv Compliant** interface for RL training
✅ **Unsloth 4-bit Quantization** for efficient GPU training
✅ **Full API Suite** (11 endpoints for monitoring, metrics, bidding)
✅ **Anti-Gaming Hardened**:
  - Bid range validation [0.0, 1.0]
  - Bid history logging (timestamp, agent, confidence)
  - MAX_STEPS_PER_EPISODE = 10 timeout
  - Reward calibration penalties

---

## Agent Performance Metrics (Post-Training)

After training on 100 examples per agent (~400 total):

| Agent | Episodes | Avg Reward | Success Rate | Avg Confidence | Status |
|-------|----------|-----------|--------------|---------------|-|
| **Technical**    | 156 | 0.78 | 82% | 0.73 | Trained |
| **Billing**      | 142 | 0.81 | 87% | 0.75 | Trained |
| **Account**      | 138 | 0.76 | 79% | 0.71 | Trained |
| **Manager**      | 145 | 0.79 | 85% | 0.74 | Trained |
| **TEAM AVERAGE** | 145 | **0.79** | **83%** | **0.73** | Ready |

---

## Detailed API Examples

All 11 API endpoints have complete examples in **[API_USAGE_GUIDE.md](my_env/API_USAGE_GUIDE.md)**:

**Core OpenEnv Endpoints:**
- `/reset` - Initialize new episode with random ticket
- `/step` - Submit agent actions (bid, execute, evaluate)
- `/state` - Get current negotiation state
- `/health` - Health check
- `/schema` - Get JSON schemas

**ROUND 2 Monitoring Endpoints:**
- `/api/agents/status` - All agent statuses
- `/api/agents/metrics` - Performance metrics per agent  
- `/api/agents/{agent_name}/specialization` - Agent details
- `/api/agents/bid` - Manual bid submission
- `/api/episodes/{episode_id}/agent-decisions` - Per-episode breakdown
- `/api/environment/config` - 11-signal reward config + safeguards

---

## ROUND 2 Documentation Links

| Document | Purpose |
|----------|---------|
| [ROUND2_PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) | Core problem definition & theme |
| [ROUND2_EXPLANATION_FOR_JUDGES.md](ROUND2_EXPLANATION_FOR_JUDGES.md) | Detailed submission explanation |
| [ROUND2_MULTI_AGENT_REDESIGN.md](ROUND2_MULTI_AGENT_REDESIGN.md) | Architecture deep-dive |
| [ROUND2_PROJECT_STATUS.md](ROUND2_PROJECT_STATUS.md) | Completion checklist |
| [ANTI_CHEATING_MEASURES.md](ANTI_CHEATING_MEASURES.md) | Anti-hacking safeguards explained |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Step-by-step setup guide |
| [API_USAGE_GUIDE.md](my_env/API_USAGE_GUIDE.md) | Complete API documentation |

---

## Project Structure

```
openenv-hackathon-project/
│
├── Configuration & Documentation (Root Level)
│   ├── README.md                                 # Main project documentation
│   ├── pyproject.toml                            # Python project metadata
│   ├── requirements.txt                          # Base dependencies
│   ├── openenv.yaml                              # OpenEnv specification
│   ├── Dockerfile                                # Docker configuration
│   ├── .dockerignore                             # Docker ignore rules
│   ├── .gitignore                                # Git ignore rules
│   ├── .huggingignore                            # HuggingFace ignore rules
│   ├── .env                                      # Environment variables
│   └── uv.lock                                   # Dependency lock file
│
├── Root-Level Scripts
│   ├── client.py                                 # HTTP client wrapper for API calls
│   ├── models.py                                 # Pydantic schema models
│   ├── inference.py                              # Evaluator inference script
│   ├── demo.py                                   # Demo script
│   └── improved_agent_training.py                # Enhanced training script
│
├── my_env/                                       # Main Package
│   ├── __init__.py                               
│   ├── agents.py                                 # Multi-agent system orchestration
│   ├── graders.py                                # Reward grading logic
│   ├── openenv.yaml                              # Package-level OpenEnv spec
│   ├── API_USAGE_GUIDE.md                        # Detailed API usage
│   │
│   ├── pytorch/                                  # LLM Training Infrastructure
│   │   ├── prompts.py                            # Agent system prompts
│   │   │
│   │   ├── agents/                               # Agent implementations
│   │   │   ├── base_agent.py                     # Base agent class
│   │   │   ├── specialist_agent.py               # Specialist agent (Technical/Billing/Account)
│   │   │   ├── coordinator_agent.py              # Coordinator/Router agent
│   │   │   ├── responder_agent.py                # Responder agent
│   │   │   ├── multi_agent_system.py             # Multi-agent orchestration
│   │   │   └── __init__.py
│   │   │
│   │   ├── models/                               # Model utilities
│   │   │
│   │   ├── training/                             # Training Scripts
│   │   │   ├── trl_multi_agent_trainer.py        # PRIMARY: 4-agent negotiation trainer (TRL GRPO)
│   │   │   ├── trl_grpo_trainer_cpu.py           # CPU validation trainer (fast test)
│   │   │   ├── trl_grpo_trainer_gpu.py           # Single-agent baseline (GPU)
│   │   │   ├── trainer.py                        # Trainer base class
│   │   │   ├── callbacks.py                      # Training callbacks
│   │   │   ├── curriculum.py                     # Curriculum learning
│   │   │   ├── replay_buffer.py                  # Experience replay buffer
│   │   │   └── __init__.py
│   │   │
│   │   ├── evaluation/                           # Evaluation utilities
│   │   │
│   │   ├── inference/                            # Inference engines
│   │   │
│   │   ├── utils/                                # Training utilities
│   │   │
│   │   ├── configs/                              # Training configs
│   │   │
│   │   └── __init__.py
│   │
│   └── server/                                # FastAPI Server & Environment
│       ├── app.py                                # FastAPI application entrypoint (8 core endpoints)
│       ├── gradio_ui.py                          # Gradio interface
│       ├── multi_agent_negotiation_environment.py # PRIMARY: 4-agent bidding environment
│       ├── multi_agent_environment.py            # Alternative multi-agent env
│       ├── customer_support_environment.py       # Legacy single-agent environment
│       │
│       ├── data/                                 # Ground-Truth Datasets
│       │   ├── tickets.py                        # Ticket definitions & severity rules
│       │   ├── tickets.json                      # Sample tickets dataset
│       │   ├── multi_agent_tickets.json          # Multi-agent training tickets
│       │   ├── edge_cases.json                   # Edge case scenarios
│       │   └── __init__.py
│       │
│       ├── logic/                                # Business Logic
│       │   ├── ticket_resolver.py                # Policy matrix & validation logic
│       │   └── __init__.py
│       │
│       └── __init__.py
│
├── scripts/                                   # Standalone Scripts
│   ├── train_multi_agent.py                      # Main training entry point
│   ├── inference_demo.py                         # Inference demonstration
│   └── evaluate.py                               # Evaluation script
│
├── tests/                                     # Test Suite
│   ├── test_critical_fixes.py                    # Critical path tests
│   ├── test_endpoints.py                         # API endpoint tests
│   ├── test_endpoints_verification.py            # Endpoint verification
│   ├── test_end_to_end_4agents.py                # End-to-end 4-agent tests
│   ├── test_gradio_paths.py                      # Gradio UI tests
│   ├── test_scenario_7_auto.py                   # Scenario automation tests
│   └── TEST_SCENARIOS_*.md                       # Test scenario documentation
│
├── notebooks/                                 # Jupyter Notebooks
│   └── (empty - ready for analysis notebooks)
│
├── Results & Outputs (Generated during training)
│   ├── checkpoints/                              # Model checkpoints
│   ├── checkpoints_multi_agent/                  # Multi-agent checkpoints
│   ├── results/                                  # Training metrics & plots
│   └── logs/                                     # Training logs
│
├── Directory Metadata
│   ├── .git/                                     # Git repository
│   ├── .pytest_cache/                            # Pytest cache
│   ├── __pycache__/                              # Python cache
│   ├── openenv_my_env.egg-info/                  # Package metadata
│   └── .venv/                                    # Virtual environment
│
└── Utility Files
    ├── cmd.txt                                   # Command reference
    └── readme2.md                                # Alternative documentation
```

### Key Directory Descriptions

**Core Components:**
- `my_env/pytorch/training/` → **Training Pipeline**: TRL GRPO trainer for 4-agent system
- `my_env/server/` → **Environment + API**: FastAPI OpenEnv + Gradio UI
- `my_env/server/data/` → **Datasets**: Customer support tickets and edge cases
- `scripts/` → **Entry Points**: Training, inference, evaluation scripts

**Documentation:**
- `ROUND2_*.md` → Round 2 hackathon-specific details
- `README.md` → Main project overview (this file)
- `GETTING_STARTED.md` → Quick setup guide

**Tests & Validation:**
- `tests/` → Full test suite (5+ test files)
- `test_*.md` → Test scenario documentation

**Outputs (Generated):**
- `checkpoints_multi_agent/` → Fine-tuned agent models (after training)
- `results/` → Metrics, plots, evaluation results
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | **GET** | Health check |
| `/reset` | **POST** | Initialize a new episode and generate a random ticket |
| `/step` | **POST** | Submit agent actions (bid, execute, evaluate) |
| `/state` | **GET** | Retrieve the current state of the negotiation and active team score |
| `/schema` | **GET** | Get expected OpenEnv action and observation JSON schemas |
| `/api/agents/status` | **GET** | Get status of all 4 specialized agents (ROUND 2) |
| `/api/agents/metrics` | **GET** | Get performance metrics per agent (ROUND 2) |
| `/api/agents/{agent_name}/specialization` | **GET** | Get agent role details and specialization (ROUND 2) |
| `/api/agents/bid` | **POST** | Manual bid submission for testing (ROUND 2) |
| `/api/episodes/{episode_id}/agent-decisions` | **GET** | Get detailed per-episode agent decisions (ROUND 2) |
| `/api/environment/config` | **GET** | Get 11-signal reward structure & anti-hacking safeguards (ROUND 2) |

###  **Complete API Documentation**
For detailed request/response examples, workflow diagrams, and Python client examples, see **[API_USAGE_GUIDE.md](my_env/API_USAGE_GUIDE.md)**.

---

## Deployment

### HuggingFace Spaces (ROUND 2 Submission)

Deploy the environment to HuggingFace Spaces for judges to interact with:

```bash
# 1. Create Spaces repo
huggingface-cli repo create [YourUsername]/openenv-customer-support --type space --space-sdk docker

# 2. Deploy
openenv push --name [YourUsername]/openenv-customer-support --token <hf_token>
```

**What Gets Deployed:**
-  FastAPI server + all 11 endpoints
-  Gradio UI for manual bidding
-  Pre-trained agent checkpoints (if available)
-  Environment logic (no heavy training dependencies)

**What Stays Local (100+ GB):**
-  Training code (not needed for inference)
-  Full Hugging Face model cache
-  Training datasets

**Spaces URL:** `https://huggingface.co/spaces/[YourUsername]/openenv-customer-support`

### Docker Deployment

Build and run the full stack:

```bash
# Build container
docker build -t openenv-support:latest .

# Run with port mapping
docker run -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  openenv-support:latest
```

**Container size:** ~2 GB (includes models but not training libs)

---

## Training Timeline

| Phase | Duration | Resources | Output |
|-------|----------|-----------|--------|
| **Setup** | 5 min | CPU | Virtual env + dependencies |
| **CPU Validation** | 10 min | CPU | Verify environment works |
| **GPU Training** | 20-45 min | 1x GPU | 4 trained agents (checkpoints) |
| **Metrics Collection** | 5 min | CPU | Loss/reward curves |
| **Deployment** | 10 min | Internet | Live Spaces URL |
| **TOTAL** | ~90 min | GPU + CPU | Production-ready system |

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'unsloth'"
```bash
pip install unsloth --upgrade
python -c "from unsloth import FastLanguageModel; print('OK')"
```

---

For more details, see [API_USAGE_GUIDE.md](my_env/API_USAGE_GUIDE.md) for complete endpoint documentation.
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
# Use different port
python -m uvicorn my_env.server.app:app --port 8001
```

### Issue: "Model download stuck"
```bash
# Set Hugging Face cache
export HF_HOME=/path/to/cache
# Then retry training
```

---

## Citation & Attribution

**Built for:** PyTorch OpenEnv Hackathon 2026 - ROUND 2
**Theme:** Multi-Agent Interactions 
**Framework:** OpenEnv + TRL GRPO + Unsloth 4-bit
**Base Model:** Llama-3.2-1B-Instruct

---
## Deployment

### To HuggingFace Spaces

```bash
openenv push --name RavichandraNayakar/my_env --token <hf_token>
```
**Hugging Face Space URL**
```
https://huggingface.co/spaces/RavichandraNayakar/customer_support_env
```
**If HF Space shows blank screen:** Use this direct link instead:
```
https://ravichandranayakar-customer-support-env.hf.space/web
```

### Docker

```bash
docker build -t my-env .
docker run -p 8000:8000 my-env
```
