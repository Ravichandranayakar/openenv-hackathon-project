"""Benchmark suite for multi-agent system."""

from typing import Any, Dict, List
import time


class BenchmarkSuite:
    """Runs comprehensive benchmarks on multi-agent system."""

    def __init__(self, multi_agent_system: Any):
        """Initialize benchmark suite.
        
        Args:
            multi_agent_system: MultiAgentSystem instance
        """
        self.system = multi_agent_system

    def benchmark_inference_latency(
        self, tickets: List[str], num_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark inference latency.
        
        Args:
            tickets: Sample tickets to test
            num_runs: Number of runs per ticket
            
        Returns:
            Latency statistics (ms)
        """
        latencies = []

        for _ in range(num_runs):
            for ticket in tickets:
                start = time.time()
                _ = self.system.forward({"ticket_text": ticket})
                latencies.append((time.time() - start) * 1000)

        return {
            "mean_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "median_ms": sorted(latencies)[len(latencies) // 2],
        }

    def benchmark_memory(self) -> Dict[str, float]:
        """Benchmark memory usage.
        
        Returns:
            Memory statistics
        """
        total_params = self.system.total_parameters
        # Approximate: 4 bytes per float32
        model_memory_mb = (total_params * 4) / (1024 * 1024)

        return {
            "total_parameters": total_params,
            "model_memory_mb": model_memory_mb,
        }

    def run_all_benchmarks(self, tickets: List[str]) -> Dict[str, Any]:
        """Run all benchmarks.
        
        Args:
            tickets: Sample tickets
            
        Returns:
            Complete benchmark results
        """
        return {
            "latency": self.benchmark_inference_latency(tickets),
            "memory": self.benchmark_memory(),
        }
