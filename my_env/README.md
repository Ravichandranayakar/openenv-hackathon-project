---
title: Tic-Tac-Toe OpenEnv Environment
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - tictactoe
  - agent-training
---

# Tic-Tac-Toe OpenEnv Environment

A production-grade **Tic-Tac-Toe game environment** built with the **OpenEnv framework** (Meta PyTorch + Hugging Face). This environment is designed for **RL agent training and evaluation** with clear task definitions, automated graders, and meaningful reward signals.

## Overview

**Key Features:**
- Full OpenEnv spec compliance (typed models, reset/step/state, openenv.yaml)
- 3 difficulty-based tasks (Easy -> Medium -> Hard)
- Automated graders (scores 0.0-1.0 per task)
- Opponent AI with 3 strategies (random, strategic, optimal)
- Meaningful reward shaping (partial progress signals)
- Baseline agent with reproducible scores
- Containerized deployment (Docker + HF Spaces)

## Environment Design

### Action Space

Agent places its mark (1) at any empty cell (0) on the 3x3 board.

```
row: int  # 0-2 (board row)
col: int  # 0-2 (board column)
```

### Observation Space

```
board: List[List[int]]           # 3x3 board state
done: bool                       # Episode terminal flag
reward: float                    # Step reward
message: str                     # Status/feedback
winner: int                      # 0=none, 1=agent, 2=opponent, 3=draw
task_id: int                     # Current task (1-3)
opponent_strength: str           # "random"/"strategic"/"optimal"
```

### Reward Function

| Event | Reward | Purpose |
|-------|--------|---------|
| Agent wins | +1.0 | Terminal success |
| Draw | +0.5 | Partial credit |
| Agent loses | -1.0 | Terminal failure |
| Valid move | +0.1 | Progress signal |
| Invalid move | -0.1 | Rule violation |

## Tasks

**Task 1 - Easy:** Random opponent - Win rate 80-90%
**Task 2 - Medium:** Strategic opponent - Win rate 40-60%
**Task 3 - Hard:** Optimal opponent - Win rate 0-20%

## API Endpoints

- `POST /reset` - Start new episode
- `POST /step` - Execute agent action
- `GET /state` - Get current game state
- `GET /health` - Health check
- `POST /tasks` - List available tasks
- `POST /grader` - Grade episode

## Running Locally

```bash
python -m openenv.client http://localhost:8000
```

Visit: https://huggingface.co/spaces/RavichandraNayakar/my_env
