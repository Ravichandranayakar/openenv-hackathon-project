"""Production inference engine for multi-agent system."""

from typing import Any, Dict, Optional
import torch
import time


class InferenceEngine:
    """Runs multi-agent inference in production mode.
    
    Optimized for latency and memory efficiency.
    """

    def __init__(self, multi_agent_system: Any, device: str = "cpu"):
        """Initialize inference engine.
        
        Args:
            multi_agent_system: MultiAgentSystem instance
            device: 'cpu' or 'cuda'
        """
        self.system = multi_agent_system
        self.device = device
        self.system.to_device(device)
        self.system.is_training = False

        # Enable inference optimizations
        for agent in self.system.get_all_agents().values():
            agent.eval()

    def infer(self, ticket_text: str, return_timing: bool = False) -> Dict[str, Any]:
        """Run inference on ticket.
        
        Args:
            ticket_text: Customer support ticket text
            return_timing: Whether to return execution time
            
        Returns:
            Dict with routing decision and metadata
        """
        with torch.no_grad():
            start_time = time.time()

            # Run multi-agent system
            observation = {"ticket_text": ticket_text}
            result = self.system.forward(observation)

            inference_time = time.time() - start_time

        result["inference_time_ms"] = inference_time * 1000

        if return_timing:
            return result
        else:
            return {k: v for k, v in result.items() if k != "inference_time_ms"}

    def batch_infer(
        self, tickets: list, return_timing: bool = False
    ) -> list[Dict[str, Any]]:
        """Run batch inference.
        
        Args:
            tickets: List of ticket texts
            return_timing: Whether to return execution time
            
        Returns:
            List of results
        """
        results = []
        for ticket in tickets:
            result = self.infer(ticket, return_timing=return_timing)
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "device": self.device,
            "total_parameters": self.system.total_parameters,
            "system_agents": list(self.system.get_all_agents().keys()),
        }
