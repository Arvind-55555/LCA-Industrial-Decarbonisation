"""
Metrics Collection for LCA Optimizer
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and track metrics for LCA calculations.
    
    Tracks:
    - Calculation times
    - API call counts
    - Cache hit rates
    - Error rates
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics: List[Metric] = []
        self.start_times: Dict[str, float] = {}
        logger.info("Metrics collector initialized")
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def stop_timer(self, operation: str) -> float:
        """Stop timing and record duration"""
        if operation not in self.start_times:
            logger.warning(f"Timer for {operation} was not started")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        self.record_metric(f"{operation}_duration", duration)
        del self.start_times[operation]
        return duration
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        # Group by metric name
        by_name = {}
        for metric in self.metrics:
            if metric.name not in by_name:
                by_name[metric.name] = []
            by_name[metric.name].append(metric.value)
        
        # Calculate statistics
        for name, values in by_name.items():
            summary[name] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "sum": sum(values)
            }
        
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()
        logger.info("Metrics reset")

