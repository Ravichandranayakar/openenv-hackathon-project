# Problem Statement: Multi-Agent Customer Support Negotiation Environment

**Theme:** Multi-Agent Interactions  
**Hackathon:** Meta × Hugging Face OpenEnv Grand Finale 2026

---

## 1. The Problem

Enterprise customer support at scale is broken — not because models aren't smart enough, but because they're being asked to do too many things at once.

When a customer sends a ticket like *"My database crashed and I also see a duplicate charge,"* a standard single-LLM chatbot faces an impossible situation. It has to be a rigorous technical engineer and an empathetic billing specialist simultaneously. It attempts both. It usually fails both — either hallucinating a fix that doesn't exist, or giving billing advice it isn't authorized to make.

The core issue isn't intelligence. It's **routing and accountability**. A model that handles everything is accountable for nothing.

Our solution is a **Multi-Agent Negotiation System** — a virtual enterprise where four specialized agents (Technical, Billing, Account, and Manager) must *compete and cooperate* to claim and resolve tickets through a structured bidding protocol. The model that wins the bid owns the outcome. That accountability is what GRPO will learn to optimize for.

---

## 2. The Environment

The environment (`MultiAgentNegotiationEnvironment`) is fully compliant with the **OpenEnv Framework** and simulates a real-time enterprise ticketing queue. It exposes three standard endpoints — `/reset`, `/state`, and `/step` — and enforces a strict 3-phase state machine that agents cannot bypass.

**State Space per Episode:**
- Customer ticket message (the raw problem description)
- Ticket severity — `low`, `medium`, `high`, or `critical`
- Ticket difficulty — `easy`, `medium`, or `hard`
- Ground-truth category — which agent *should* handle it (used for reward calculation)
- Current negotiation phase — `bidding`, `execution`, or `resolution`

**The 3-Phase Protocol:**

```
BIDDING PHASE → EXECUTION PHASE → RESOLUTION PHASE
```

Each phase is enforced at the environment level. If an agent tries to submit a solution during the bidding phase, it gets blocked. If a non-winning agent tries to execute during the execution phase, the environment returns an explicit IAM-style error: *"Only [winning agent] can execute. You are [other agent]."* There is no workaround.

**Action Space:**
- `bid` — submit a confidence score `[0.0–1.0]` representing self-assessed competence on the ticket
- `execute` — the winning bidder proposes a resolution based on internal enterprise policy
- `evaluate` — the Manager agent performs quality assurance and decides whether to close or escalate

---

## 3. Agent Capabilities

We run a single fine-tuned LLM backbone (**Llama-3.2-1B-Instruct** via Unsloth 4-bit quantization) and instantiate four distinct agents through targeted system prompts, each with hard domain boundaries:

| Agent | Domain | Key Capabilities |
|---|---|---|
| **Technical Agent** | Engineering | App crashes, API failures, data sync errors, database issues |
| **Billing Agent** | Finance | Duplicate charges, subscription management, payment fraud detection |
| **Account Agent** | Identity | 2FA resets, password recovery, unauthorized access, security breaches |
| **Manager Agent** | Oversight | Auctioneer in bidding phase, final QA check, escalation decision-maker |

The Manager agent has a dual role: it acts as the neutral auctioneer that selects the highest valid bidder, and it performs the final evaluation at the end of each episode — deciding whether the proposed resolution is adequate or whether the ticket needs to escalate to a human operator.

---

## 4. Tasks the System Must Perform

The environment loads from a dataset of **45 distinct customer tickets** spanning 3 difficulty levels and 3 domain categories. For every episode, the system must:

1. **Independently analyze** the incoming ticket — each specialist agent reads it in isolation, with no access to other agents' reasoning
2. **Negotiate via confidence bidding** — e.g., Technical Agent bids `0.95` on a database crash ticket; Billing Agent bids `0.10` on the same ticket
3. **Enforce winner accountability** — only the highest valid bidder can submit a solution; other agents are locked out
4. **Resolve accurately** — the winning agent's solution is checked against a hardcoded enterprise policy matrix to determine correctness
5. **Escalate intelligently** — the Manager must recognize when a ticket's difficulty (`critical` + `hard`) exceeds AI capability and trigger human escalation

The partial observability constraint applies throughout: agents can see each other's *numerical bids* but never their reasoning. This forces agents to develop inference — "the Billing Agent bid `0.90`, so they likely have high domain confidence on this ticket, I should defer."

---

## 5. The Reward Model — 11 Independent Signals

A single aggregate reward function is an invitation for reward hacking. An agent can bid `1.0` on every ticket, occasionally win, and extract reward without ever developing real domain specialization. We prevent this with **11 independent reward and penalty signals** that fire simultaneously at the end of each episode:

**Positive Rewards:**

| Signal | Value | Purpose |
|---|---|---|
| `correct_specialist_bid` | +0.20 | Bid high on a ticket that matches your true domain |
| `correct_solution` | +0.20 | The winning agent's solution matches the policy matrix |
| `appropriate_confidence` | +0.10 | Bid calibration matches actual solution accuracy |
| `solution_format` | +0.05 | Clean, valid structured JSON output |
| `team_success_bonus` | +0.20 | Given to **all agents** when the ticket resolves correctly |

**Penalty Signals:**

| Signal | Value | What It Prevents |
|---|---|---|
| `wrong_specialist` | -0.20 | Bidding high outside your domain |
| `wrong_solution` | -0.20 | Winning the bid but failing to resolve correctly |
| `overconfident` | -0.10 | Bidding high (`>0.8`) and then getting it wrong |
| `team_failure_penalty` | -0.10 | Applied to all agents when the ticket fails to resolve |
| `invalid_bid` | -0.05 | Malformed or out-of-range bid output |
| `timeout` | -0.15 | Failing to complete a phase within the time limit |

The `team_success_bonus` and `team_failure_penalty` pair is critical to the design. They make cooperation *mathematically incentivized* — even an agent that loses the bid profits from helping the right agent win. Greedy bidding actively pulls down the whole team's score. The agents must learn that honest confidence calibration is the optimal long-term strategy.

---

## 6. Post-Training / Self-Improvement Strategy

We train the shared LLM backbone using **TRL GRPO (Group Relative Policy Optimization)** rather than PPO, for a specific architectural reason: our environment is itself the verifier. We don't need a learned value model to estimate future rewards — the 11 reward functions calculate the outcome deterministically after each episode. GRPO directly compares groups of rollouts against each other to compute the policy gradient, which fits our setup exactly and eliminates the extra VRAM overhead of a separate value network.

**The Training Loop:**

```
1. Environment resets → loads a fresh customer ticket
2. All 4 agents receive the ticket via their specialized system personas
3. Agents generate structured JSON outputs: bids, solutions, evaluations
4. Environment verifies each action → calculates reward from 11 independent signals
5. TRL GRPO computes the policy gradient using group-relative reward comparison
6. Shared Llama-3.2-1B backbone weights are updated
7. Repeat → agents gradually learn when to bid high, when to defer, when to escalate
```

**Measured Improvement After Training:**

| Behavior | Before GRPO | After GRPO |
|---|---|---|
| JSON output validity | Frequently malformed | Consistently valid |
| Bidding behavior | Uniform ~0.5 on all tickets | Domain-specific (`0.9+` on specialty, `<0.1` on others) |
| Manager escalation accuracy | ~50% (near-random) | 85%+ |
| Average episode reward | `-0.10` to `+0.10` | `+0.70` to `+0.85` |

The agents learned genuine specialization — not because we told them to, but because the reward structure made it the only rational strategy. That is the outcome we designed this environment to produce.