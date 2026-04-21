"""Metrics calculation for multi-agent cooperation."""

from typing import Any, Dict, List
from collections import defaultdict


class CooperationMetrics:
    """Computes cooperation and performance metrics for multi-agent system.
    
    Tracks:
    - Routing accuracy (% correct specialist)
    - Communication efficiency (messages per resolution)
    - Escalation appropriateness
    - Consensus score (agent agreement)
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.routing_history = defaultdict(list)  # classification -> [routing_decisions]
        self.escalation_history = []
        self.communication_log = []
        self.accuracy_per_specialist = defaultdict(lambda: {"correct": 0, "total": 0})

    def record_routing(
        self, classification: int, routing_decision: int, was_correct: bool
    ) -> None:
        """Record routing decision.
        
        Args:
            classification: Ticket classification
            routing_decision: Routing choice
            was_correct: Whether routing was correct
        """
        self.routing_history[classification].append(
            {"decision": routing_decision, "correct": was_correct}
        )

        specialist_name = self._action_to_specialist(routing_decision)
        self.accuracy_per_specialist[specialist_name]["total"] += 1
        if was_correct:
            self.accuracy_per_specialist[specialist_name]["correct"] += 1

    def record_escalation(self, was_necessary: bool) -> None:
        """Record escalation decision.
        
        Args:
            was_necessary: Whether escalation was actually needed
        """
        self.escalation_history.append({"necessary": was_necessary})

    def record_message(self, sender: str, recipient: str, message_type: str) -> None:
        """Record inter-agent message.
        
        Args:
            sender: Sender agent
            recipient: Recipient agent
            message_type: Type of message
        """
        self.communication_log.append(
            {"sender": sender, "recipient": recipient, "type": message_type}
        )

    def compute_routing_accuracy(self) -> Dict[int, float]:
        """Compute routing accuracy per classification.
        
        Returns:
            Dict mapping classification -> accuracy (0-1)
        """
        accuracy = {}
        for classification, decisions in self.routing_history.items():
            if len(decisions) == 0:
                accuracy[classification] = 0.0
            else:
                correct = sum(1 for d in decisions if d["correct"])
                accuracy[classification] = correct / len(decisions)
        return accuracy

    def compute_specialist_accuracy(self) -> Dict[str, float]:
        """Compute accuracy per specialist.
        
        Returns:
            Dict mapping specialist_name -> accuracy (0-1)
        """
        accuracy = {}
        for specialist, stats in self.accuracy_per_specialist.items():
            if stats["total"] == 0:
                accuracy[specialist] = 0.0
            else:
                accuracy[specialist] = stats["correct"] / stats["total"]
        return accuracy

    def compute_escalation_appropriateness(self) -> float:
        """Compute % of escalations that were necessary.
        
        Returns:
            Appropriateness score (0-1)
        """
        if len(self.escalation_history) == 0:
            return 0.0
        necessary = sum(1 for e in self.escalation_history if e["necessary"])
        return necessary / len(self.escalation_history)

    def compute_communication_efficiency(self) -> float:
        """Compute avg messages per resolution.
        
        Returns:
            Messages per resolution
        """
        if len(self.communication_log) == 0:
            return 0.0
        return len(self.communication_log) / max(1, len(self.routing_history))

    def compute_consensus_score(self) -> float:
        """Compute consensus score (agent agreement on routing).
        
        Returns:
            Consensus score (0-1)
        """
        # Placeholder: would require tracking agent disagreements
        return 0.5

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.
        
        Returns:
            Comprehensive metrics dictionary
        """
        return {
            "routing_accuracy_per_class": self.compute_routing_accuracy(),
            "specialist_accuracy": self.compute_specialist_accuracy(),
            "escalation_appropriateness": self.compute_escalation_appropriateness(),
            "communication_efficiency": self.compute_communication_efficiency(),
            "consensus_score": self.compute_consensus_score(),
            "total_routings": sum(
                len(decisions) for decisions in self.routing_history.values()
            ),
            "total_escalations": len(self.escalation_history),
            "total_messages": len(self.communication_log),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.routing_history.clear()
        self.escalation_history.clear()
        self.communication_log.clear()
        self.accuracy_per_specialist.clear()

    @staticmethod
    def _action_to_specialist(action: int) -> str:
        """Convert routing action to specialist name."""
        names = ["billing", "account", "technical", "specialist", "escalate"]
        return names[min(action, len(names) - 1)]
