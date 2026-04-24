"""Structured logging utilities."""

import logging
import json
from typing import Any, Dict


class StructuredLogger:
  """Structured logging for multi-agent system."""

  def __init__(self, name: str, log_file: str = None):
    """Initialize logger.
    
    Args:
      name: Logger name
      log_file: Optional log file path
    """
    self.logger = logging.getLogger(name)
    self.logger.setLevel(logging.INFO)

    if log_file:
      handler = logging.FileHandler(log_file)
      formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
      )
      handler.setFormatter(formatter)
      self.logger.addHandler(handler)

  def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
    """Log structured event.
    
    Args:
      event_type: Type of event
      data: Event data dictionary
    """
    log_data = {"event_type": event_type, "data": data}
    self.logger.info(json.dumps(log_data))

  def log_metric(self, metric_name: str, value: float) -> None:
    """Log metric.
    
    Args:
      metric_name: Name of metric
      value: Metric value
    """
    self.log_event("metric", {"name": metric_name, "value": value})
