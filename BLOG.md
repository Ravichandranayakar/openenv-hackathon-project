# Teaching AI Agents to Negotiate: How We Built a Multi-Agent Support Environment from Scratch

> Built for the **Meta × Hugging Face OpenEnv Hackathon 2026 — Grand Finale**

---

Most AI demos look clean. What you don't see is the merge conflict at midnight, the schema mismatch that kept returning `KeyError: 'ticket'`, and the moment when 9 out of 9 end-to-end tests finally passed and you genuinely felt like something real had just come alive.

This is the story of how we built a **4-agent negotiation environment** using the OpenEnv framework, trained it with GRPO reinforcement learning on Llama-3.2-1B, and why we think this is the right direction for enterprise AI — not just a hackathon project.

---

## The Problem We Were Actually Trying to Solve

Here's a real scenario: a customer sends a ticket — *"The app crashes when I upload a file, and I've also been charged twice this month."* 

What does a monolithic LLM do with that? It tries to handle both. It hallucinates a technical fix that doesn't exist in the codebase. It gives billing advice it's not authorized to give. It confidently gets both wrong.

This isn't a model capability problem. It's a **routing and coordination** problem. And the solution isn't a smarter single model. It's a team of specialized agents that know their limits, bid on what they understand, and defer when they don't.

That's what we built.

---

## The Architecture: A 3-Phase Negotiation Protocol

We didn't start with training. We started with the **environment** — the game the agents would eventually learn to play. The entire system runs on the OpenEnv API standard, exposed via FastAPI endpoints (`/reset`, `/state`, `/step`) and interacted with through a Gradio-based Command Center UI.

Every time a new customer ticket loads, the environment runs through three strict phases:

### Phase 1 — Bidding (Where Theory-of-Mind Begins)

No agent is told who should handle the ticket. All three specialist agents — **Technical**, **Billing**, and **Account** — independently read the ticket and submit a **confidence bid** between `0.0` and `1.0`.

This is where partial observability matters. Each agent can see the *numerical bids* of the others, but not their reasoning. They're flying partially blind. An agent seeing a rival bid `0.95` on a billing complaint has to infer: *"They're more confident than me. I should step back."* That's theory-of-mind — and it's exactly what GRPO will teach the model to do through experience.

### Phase 2 — Execution (The Winner Takes Responsibility)

The environment acts as a strict auctioneer. The agent with the highest valid confidence bid wins **exclusive execution rights**. They must now propose a real resolution based on enterprise policy.

Here's something we built that we're proud of: if a non-winning agent tries to submit a solution anyway, the environment blocks them instantly with an **IAM-style anti-hacking safeguard** and returns an error observation. It literally tells the agent: *"Only account agent can execute. You are execution."* Clean, strict, no bypassing.

### Phase 3 — Resolution (11 Independent Reward Signals)

This is where the environment does something most RL environments skip — it doesn't give a single reward. It calculates **11 independent reward functions** simultaneously, each targeting a specific behavior the agents need to learn:

| Signal | Value | What It Teaches |
|---|---|---|
| `correct_specialist_bid` | +0.20 | Bid high when it's genuinely your domain |
| `correct_solution` | +0.20 | Actually solve the ticket correctly |
| `appropriate_confidence` | +0.10 | Calibrate your bid to match your actual accuracy |
| `solution_format` | +0.05 | Output clean, structured JSON |
| `team_success_bonus` | +0.20 | Given to ALL agents if the ticket resolves correctly |
| `wrong_specialist` | -0.20 | Penalty for bidding high on the wrong domain |
| `wrong_solution` | -0.20 | Penalty for winning and then failing |
| `overconfident` | -0.10 | Penalty for bidding 0.9+ and being wrong |
| `team_failure_penalty` | -0.10 | The whole team suffers if the wrong agent wins |
| `invalid_bid` | -0.05 | Penalty for malformed bid output |
| `timeout` | -0.15 | Penalty for not responding in time |

The `team_success_bonus` and `team_failure_penalty` are intentional design choices. They force the agents to develop **cooperative intelligence** — a losing bidder is actually rewarded for correctly letting the right agent win. Greedy bidding is actively discouraged. This is what makes the negotiation real.

---

## Proving the Environment Actually Works

Before we touched any LLM, we wrote a full end-to-end test suite to validate that the environment's internal logic was correct. We ran `test_end_to_end_4agents.py` and watched it walk through all 6 stages:

```
[1/6] Initialize Environment      ✅ Episode loaded: T005
[2/6] Bidding Phase               ✅ 3 agents submitted bids
[3/6] Winner Selection            ✅ ACCOUNT won with bid 0.85
[4/6] Execution Phase             ✅ Solution proposed
[5/6] Resolution Phase            ✅ 11 rewards calculated
[6/6] Final Results               ✅ 9/9 validation checks passed
```

What this proved: the environment correctly applied all 11 reward functions, enforced phase transitions, blocked invalid actions, and calculated per-agent rewards that made logical sense. The technical agent got `+0.100` because that was their actual specialty — even though they lost the bid. The account agent who won but solved the wrong problem got penalized with `-0.550`.

The environment understood cause and consequence. That's when we knew it was ready for training.

---

## Why GRPO, Not PPO?

We chose **GRPO (Group Relative Policy Optimization)** via TRL over PPO for a specific reason: our environment *is* the reward function. We didn't need a learned value model to estimate returns — we had 11 deterministic reward signals firing every step.

PPO requires a separate value network running alongside the policy to estimate future rewards. That's extra VRAM, extra training complexity, and extra things to go wrong. GRPO eliminates the value model entirely by comparing groups of rollouts against each other to compute the policy gradient. In a verifiable environment like ours, this is a perfect fit.

It also let us run training on **Llama-3.2-1B-Instruct** with **Unsloth 4-bit quantization** without needing enterprise-grade hardware — which mattered a lot for a hackathon with credit constraints.

**The results were concrete:**

| Metric | Before Training | After Training |
|---|---|---|
| JSON output validity | Often malformed | Reliable structured output |
| Bidding behavior | Uniform ~0.5 on everything | Domain-specific (0.9+ on specialty) |
| Manager escalation accuracy | ~50% random | 85%+ accuracy |
| Average episode reward | -0.10 to +0.10 | +0.70 to +0.85 |

The agents genuinely learned to specialize. The technical agent stopped bidding on billing tickets. The billing agent stopped pretending to know how to fix bugs. That's emergent strategic behavior — not hard-coded rules.

---

## The Command Center UI

We built a Gradio-based **Multi-Agent Support Orchestrator** UI to make the environment's behavior observable and interactive. The design went through a few iterations (we'll be honest — we threw out the first two versions entirely).

The final UI runs on a glassmorphism dark theme (deep navy-to-midnight-purple gradient, semi-transparent glass cards with `backdrop-filter: blur`) and gives you full control over the negotiation:

- **Current Ticket Context** — the active customer problem
- **Protocol Execution** — choose which agent acts, set confidence bid or solution text
- **Live Scoreboard** — real-time per-agent reward tracking
- **Negotiation History** — every action logged with its outcome
- **11-Signal Reward Breakdown** — full episode reward breakdown after resolution
- **Debug JSON** — raw state dump for inspection

The key thing here is that the UI isn't decorative. You can manually act as the AI — clicking through the 3 phases yourself — and the environment's scoring logic responds exactly the same way it does during LLM training. It's a developer-facing proof that the game engine works before any model is plugged in.

---

## What This Environment Actually Tests

The hackathon theme was **Multi-Agent Interactions** — specifically environments that enable cooperation, competition, negotiation, and theory-of-mind reasoning. We mapped every design decision we made to those criteria:

- **Competition** — agents bid against each other for execution rights
- **Cooperation** — `team_success_bonus` means helping the right agent win is in your interest
- **Negotiation** — confidence bids are a real negotiation signal, not just random numbers
- **Partial observability** — agents see bids but not reasoning, forcing inference about others
- **Theory-of-mind** — "they bid high, so they probably know something I don't"
- **Emergent behavior** — agents learn to specialize without being explicitly told to

This isn't a demonstration environment with canned outputs. The 11 reward functions, the phase-enforcement, the anti-hacking safeguards — all of it exists to create a space where intelligent, strategic behavior is the *only* thing that gets rewarded.

---

## What's Next

We trained on Llama-3.2-1B as a proof of concept. The architecture is model-agnostic — the environment doesn't care what LLM is plugged into the `/step` endpoint. The next natural step is scaling to Llama-3-8B or a domain-tuned model and running longer training episodes to see whether inter-agent communication strategies emerge.

For anyone interested in building on this: the environment is fully open, the reward functions are modular, and the FastAPI interface makes it straightforward to swap in different agent architectures.

---

**Repository:** [RavichandraNayakar/openenv-hackathon-project](https://github.com/RavichandraNayakar/openenv-hackathon-project)  
**Trained Model (Merged):** [RavichandraNayakar/openenv-grpo-merged](https://huggingface.co/RavichandraNayakar/openenv-grpo-merged)  
**LoRA Adapters:** [RavichandraNayakar/openenv-multi-agent-grpo](https://huggingface.co/RavichandraNayakar/openenv-multi-agent-grpo)  
**Training Notebook:** [notebooks/Multi_Agent_GRPO_Training_output.ipynb](./notebooks/Multi_Agent_GRPO_Training_output.ipynb)