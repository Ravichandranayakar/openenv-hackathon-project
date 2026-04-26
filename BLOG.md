# Teaching AI to Cooperate: A 4-Agent Negotiation Protocol in OpenEnv

In modern enterprise systems, a single monolithic AI model cannot handle every customer support edge-case effectively. Routing bugs to engineers, resolving billing disputes, and managing account security all require specialized domain knowledge. 

To solve this, we built a **Multi-Agent Negotiation Environment** on top of the OpenEnv framework. Instead of a single model attempting to do everything, four specialized agents (Technical, Billing, Account, and Manager) must negotiate in real-time to claim and resolve tickets. 

This environment serves as a rigorous testing ground for Reinforcement Learning (specifically GRPO), teaching LLMs how to accurately estimate their own confidence, cooperate in a partially observable environment, and defer to specialists.

## The Environment Architecture

The environment orchestrates a strict **3-Phase Negotiation Protocol**:

### 1. Bidding Phase (Theory of Mind & Confidence Calibration)
When a new customer ticket arrives (e.g., "I was charged twice!"), the environment does not tell the agents who should take it. Instead, all three specialist agents (Technical, Billing, Account) must independently analyze the ticket and submit a **Confidence Bid (0.0 - 1.0)**. 
Because the environment is partially observable, agents only see each other's numerical bids, not the reasoning behind them. They must learn to calibrate their confidence based on their own specialization.

### 2. Execution Phase (Specialized Problem Solving)
The environment acts as the auctioneer, selecting the agent with the highest valid bid. The winning agent is granted exclusive execution rights to propose a resolution based on enterprise policy matrices. If a non-winning agent attempts to hijack the execution, the environment strictly blocks them with an anti-hacking safeguard, simulating strict enterprise IAM permissions.

### 3. Resolution Phase (Independent Reward Calculation)
The environment evaluates the outcome and calculates **11 Independent Reward Functions**. Instead of a simple pass/fail, agents are rewarded and penalized based on micro-behaviors:
* `correct_specialist_bid`: +0.20 for bidding high on your true specialty.
* `overconfident`: -0.10 for winning a bid but failing to provide the correct solution.
* `team_success_bonus`: +0.20 given to ALL agents if the ticket is resolved correctly, encouraging team coordination rather than greedy bidding.

## Why This Matters for Reinforcement Learning
Training on this environment pushes an LLM beyond simple instruction-following. It forces the model to develop **strategic emergent behavior**. Through reinforcement learning (using `trl` + GRPO), the model learns that artificially inflating its confidence (reward hacking) results in an `overconfident` penalty, while correctly deferring a task to a more suited team member yields a `team_success_bonus`.

This Multi-Agent Command Center demonstrates that the future of enterprise AI isn't just about making models smarter—it's about teaching them how to work together.

**Links:**
* **GitHub Repository:** [RavichandraNayakar/openenv-hackathon-project](https://github.com/RavichandraNayakar/openenv-hackathon-project)
* **Trained Model (Merged):** [RavichandraNayakar/openenv-grpo-merged](https://huggingface.co/RavichandraNayakar/openenv-grpo-merged)
* **LoRA Adapters:** [RavichandraNayakar/openenv-multi-agent-grpo](https://huggingface.co/RavichandraNayakar/openenv-multi-agent-grpo)
* **Training Notebook:** [notebooks/Multi_Agent_GRPO_Training_output.ipynb](./notebooks/Multi_Agent_GRPO_Training_output.ipynb)
