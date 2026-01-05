"""
Isolation Forest - Anomaly Detection Algorithm

A Python implementation of the Isolation Forest algorithm for anomaly detection.
The algorithm isolates anomalies by randomly partitioning the data space.

The key insight: anomalies are easier to isolate (require fewer partitions)
because they are 'few and different' from normal instances.
"""

__version__ = "1.0.0"

from .forest import IsolationForest
from .tree import IsolationTree
from .nodes import ExternalNode, InternalNode

__all__ = [
    'IsolationForest',
    'IsolationTree',
    'ExternalNode',
    'InternalNode',
]

