# Round 2 Project - Complete Explanation for Judges

**Date**: April 23, 2026 (Final Submission Ready)
**Theme**: Multi-Agent Interactions
**Stack**: OpenEnv + TRL GRPO + Unsloth (Official Hackathon Stack)

---

## Table of Contents
1. The Problem: The "Generalist" AI Bottleneck
2. Our Solution: The 4-Agent Negotiation Protocol
3. Why This Matches the Prompt Perfectly
4. System Architecture: How It Actually Works
5. The 11-Signal Anti-Hacking Reward Model
6. The Training Stack (TRL + Unsloth)
7. What We Built (Codebase Overview)

---

## 1. The Problem: The "Generalist" AI Bottleneck

### The Status Quo
Most companies implement customer support AI as a single, massive "Generalist" LLM. They prompt one model to be an expert in everything.
**The result?** The model hallucinates on complex technical edge cases, or inappropriately handles sensitive billing disputes because it lacks deep, isolated context.

### The Real Enterprise Reality
In a real enterprise, support isn't handled by one omniscient person. A ticket is analyzed, routed, and handled by **specialized teams** (Billing, IT, Security).

We hypothesized: *What if we simulated a specialized corporate hierarchy using multiple AI personas?*

---

## 2. Our Solution: The 4-Agent Negotiation Protocol

Instead of a single bot, we built a **Multi-Agent Corporate Environment**. When a ticket enters the system, it triggers a 3-Phase State Machine:

1. **The Bidding Phase (Competition)**: Our 3 specialist agents (Technical, Billing, Account) read the ticket simultaneously. Based on their system prompts, they mathematically calculate their confidence (0.0 to 1.0) on whether the ticket belongs in their domain. They submit "bids".
2. **The Execution Phase (Resolution)**: The environment acts as an auctioneer. The agent with the highest valid bid "wins", takes ownership of the ticket, and proposes a solution against enterprise policy.
3. **The Evaluation Phase (Quality Assurance)**: Our 4th agent, the **Manager**, reviews the specialist's prescribed solution. If the ticket is deemed too critical or the solution invalid, the Manager escalates to humans. Otherwise, the ticket closes.

---

## 3. Why This Matches the Round 2 prompt Perfectly

The official hackathon rubric asks for: *"Environments for this theme involve cooperation, competition, negotiation, and coalition formation... driving theory-of-mind reasoning."*

**Our Bidding Protocol is textbook Multi-Agent Negotiation.**
- **Competition**: Agents compete for the highest bid to "win" the ticket.
- **Cooperation**: Our reward matrix contains a "Team Bonus" (awarded only if the ticket resolves successfully) and a "Team Penalty" (if the wrong agent wins and ruins the resolution, everyone loses points). 
- **Theory of Mind**: Agents must learn *not* to bid 1.0 on every ticket. The Technical agent must learn to recognize a Billing issue, realize it will fail the Execution phase, and strategically bid 0.1 to let the Billing agent handle it and secure the Team Bonus.

---

## 4. System Architecture: How It Actually Works

```text
Customer Support Ticket
"I was charged twice AND my app keeps crashing"
     ↓
  [ Open Environment Broker ]
     ↓
  (Bidding Phase Triggered)
  - Technical Agent Bids: 0.85
  - Billing Agent Bids:  0.95
  - Account Agent Bids:  0.10
     ↓
 [ Billing Agent Wins Auction ]
     ↓
 Billing Agent Proposes: "Refund duplicate charge"
     ↓
 [ Manager Agent Reviews ]
 Manager Decision: "Approved, Close Ticket"
     ↓
 [ Environment Computes Rewards ]
  Result: Team Success (+1.0)
```

---

## 5. The 11-Signal Anti-Hacking Reward Model

As per OpenEnv principles, a single reward function is susceptible to reward hacking (e.g., an agent always bidding 1.0 just to maximize exposure). We implemented **11 Independent Metrics** across the 3 phases to mathematically prevent cheating:

**Positive Rewards (The Carrots):**
1. Correct Specialist Won (+0.2)
2. Accurate Solution Generated (+0.2)
3. Confidence Calibration Bonus (+0.1)
4. Valid JSON Formatting (+0.05)
5. **Team Synergy Bonus (+0.2)** *(Only applied if the Manager approves the final close)*

**Anti-Hacking Penalties (The Sticks):**
6. Wrong Specialist Won (-0.2)
7. Invalid Policy Solution (-0.2)
8. **False Confidence Penalty (-0.1)** *(Agent bid > 0.8 but got the solution wrong)*
9. Team Failure Penalty (-0.1)
10. Out-of-bounds Bid (-0.05)
11. **Timeout Penalty (-0.15)** *(Agents looping without resolution)*

---

## 6. The Training Stack (TRL + Unsloth)

To actually train these agents to optimize this 11-signal matrix, we use **Transformers Reinforcement Learning (TRL) - GRPO.**

Rather than loading 4 separate neural networks into GPU memory, we load **a single Llama-3.2-1B-Instruct backbone**, optimized using **Unsloth 4-bit Quantization**.
We instantiate the 4 agents dynamically by altering the system prompt context window in the RL trajectory loop.

The GRPO algorithm measures the expected reward of the agents' bids and solutions against the Environment's Oracle ground-truth. Over 500 episodes, the underlying weights of the LLM are updated to maximize the total Team Synergy score, naturally teaching the underlying AI the boundaries of the 4 specializations.

---

## 7. What We Built (Codebase Overview)

Our repository represents a fully production-ready OpenEnv enterprise architecture:

- `multi_agent_negotiation_environment.py`: The core OpenEnv FastApi server implementing the strict 3-phase state machine and Oracle evaluation logic.
- `app.py`: The HTTP wrapper exposing 11 distinct endpoints (including manual Live Metrics routing).
- `gradio_ui.py`: A custom front-end command center allowing judges to manually "play" as any of the 4 agents and test the bidding protocol in real time.
- `trl_multi_agent_trainer.py`: The primary RL execution loop that wraps TRL and sequences the agent fine-tuning.
- `data/tickets.py`: 45 hand-crafted edge-case support scenarios designed specifically to confuse naive LLMs.

**Ready for Deployment.** We've packaged the entire inference architecture in a Docker container ready for HuggingFace Spaces.
