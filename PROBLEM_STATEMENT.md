# PROBLEM_STATEMENT

## Theme:  Multi-Agent Interactions 

## 1. The Problem Statement
In modern enterprise environments, scaling customer support is a monumental challenge. Current AI support chatbots suffer from a "generalist" problem: a single LLM prompted to answer everything inevitably hallucinates on complex technical edge cases or fumbles sensitive billing disputes. When an angry customer writes, *"My database crashed and I see a duplicate charge,"* a standard chatbot fails because it cannot simultaneously act as an empathetic billing specialist and a rigorous technical engineer.

**Our Solution:** 
We have designed a **Multi-Agent Negotiation System**. Instead of a single generalist bot, we simulate a virtual enterprise containing 4 hyper-specialized AI agents: a Router, a Technical Specialist, a Billing Specialist, and a Quality Assurance Manager. When a ticket arrives, these agents use a **Bidding Protocol** to negotiate who has the highest confidence to solve the issue, collaborating to ensure the most qualified agent handles the customer.

---

## 2. The Environment
The environment (`MultiAgentNegotiationEnvironment`) is fully compliant with the **OpenEnv Framework** and simulates a real-time ticketing queue. 

**Structure:**
- **State Space**: Customer ticket message, severity (low, medium, high, critical), difficulty (easy, medium, hard), and the current negotiation phase.
- **Phases**: The environment enforces a strict state machine: `Bidding Phase` → `Execution Phase` → `Resolution Phase`.
- **Action Space**:
 - `bid`: Agents submit a confidence score (0.0 to 1.0) and rationale.
 - `execute`: The winning agent proposes a solution based on internal enterprise policies.
 - `evaluate`: The Manager agent decides if the resolution is adequate or if human escalation is required.

---

## 3. The Capabilities of the Agents
Using a single optimized LLM (Llama 3.2 1B Instruct via Unsloth), we instantiate 4 distinct personas via targeted system prompts:

1. **Billing Agent**: Capable of identifying duplicate charges, managing subscriptions, and detecting payment fraud.
2. **Technical Agent**: Capable of diagnosing app crashes, data sync issues, and API outages.
3. **Account Agent**: Capable of handling 2FA resets, password recoveries, and security breaches.
4. **Manager Agent**: Operates as the "Auctioneer" during the Bidding Phase and serves as the final Quality Assurance check before a ticket is closed.

---

## 4. The Tasks to be Performed
The multi-agent system must process a high-velocity stream of incoming support tickets (45 distinct scenarios spanning 3 difficulty levels). 
For every ticket, the system must:
1. **Analyze** the ticket context independently.
2. **Negotiate** via confidence bidding (e.g., Technical Agent bids 0.95 for a database crash, while Billing Agent bids 0.10).
3. **Resolve** the ticket accurately based on a strict hardcoded enterprise policy matrix.
4. **Determine** if the ticket complexity exceeds AI capabilities (triggering an escalation).

---

## 5. The Reward Model & Evaluation Logic (Anti-Hacking Hardened)
As per OpenEnv principles, a single reward function is susceptible to reward hacking. We implemented **11 Independent Reward/Penalty Signals**:

**Positive Rewards:**
- **`CORRECT_SPECIALIST_REWARD` (+0.3)**: Awarded if the agent with the highest bid actually corresponds to the ticket's ground-truth category.
- **`TEAM_SYNERGY_REWARD` (+0.2)**: Awarded to all agents if the ticket is successfully resolved, encouraging honest bidding (agents are penalized if they bid high on a ticket they can't solve, bringing down the whole team).
- **`CORRECT_ESCALATION` (+0.25)**: Awarded to the Manager for successfully identifying critical/hard tickets that require human intervention.

**Anti-Hacking Penalties:**
- **`TIMEOUT_PENALTY` (-0.1)**: Applied if agents loop continuously without reaching a resolution phase.
- **`MALFORMED_BID_PENALTY` (-0.1)**: Applied if an agent bids outside the valid [0, 1] range.
- **`FALSE_CONFIDENCE_PENALTY` (-0.2)**: Applied when an agent bids >0.8 but proposes an invalid solution, preventing agents from always blindly bidding 1.0 to win the auction.

---

## 6. Post-Training Strategy
Our system improves via **Transformers Reinforcement Learning (TRL) GRPO**.

**The Training Loop:**
1. A batch of tickets is pushed to the Environment.
2. The LLM generates trajectories (bids and resolution actions) for all 4 agents.
3. The Environment evaluates the actions against the 11-signal reward matrix.
4. GRPO updates the underlying weights of the LLM to maximize expected team reward.

**Execution:**
By leveraging **Unsloth 4-bit quantization**, we can efficiently train this collaborative multi-agent dynamic on a single consumer GPU, allowing the shared LLM backbone to learn the nuances of when to speak up (bid high) and when to defer to a colleague (bid low).
