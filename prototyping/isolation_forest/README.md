# Isolation Forest

A Python implementation of the Isolation Forest algorithm for anomaly detection from the research paper by Liu et al.

> **Note**: This is a reference implementation for discussion with SystemDS contributors. This code is not intended for direct integration into SystemDS but serves as a prototype for implementing Isolation Forest functionality.

## Overview

Isolation Forest is an unsupervised learning algorithm for anomaly detection. The key insight is that **anomalies are easier to isolate** because they are 'few and different' from normal instances.

The algorithm builds an ensemble of random trees that partition the data. Anomalies require fewer random partitions to be isolated, resulting in shorter path lengths in the trees.

## Installation

**Requirements**: Python 3.7+

```bash
git clone https://github.com/yourusername/isolation_forest.git
cd isolation_forest
pip install -r requirements.txt
```

**Note**: The core implementation only requires `numpy`. `scikit-learn` is used in the example for convenience (train/test split and metrics).

Run the included example:
```bash
python examples/basic_usage.py
```

## How It Works

### 1. Building Isolation Trees

Each tree recursively partitions data by:
- Randomly selecting a feature
- Randomly selecting a split value between min and max
- Splitting data into left and right branches

Trees stop growing when they reach:
- Height limit (⌈log₂(sample_size)⌉)
- Only one instance in a node
- All instances are identical

### 2. Path Length & Anomaly Scoring

The **path length** is the number of edges from root to leaf. Anomalies have shorter average path lengths across all trees because they're easier to isolate.

The anomaly score is calculated as:

```
score = 2^(-average_path_length / c(n))
```

Where `c(n)` normalizes the path length based on the sample size.

**Score interpretation:**
- **~1.0**: Strong anomaly
- **~0.5**: Ambiguous
- **~0.0**: Normal instance

## API Reference

### IsolationForest

Main class for anomaly detection.

**Parameters:**
- `n_estimators` (int, default=100): Number of isolation trees
- `max_samples` (int, default=256): Number of samples to draw for each tree
- `random_state` (int, optional): Random seed for reproducibility

**Methods:**
- `fit(X)`: Build the forest from training data
- `predict(X)`: Return predictions (-1 for anomaly, 1 for normal)
- `score_samples(X)`: Return anomaly scores (0 to 1, higher = more anomalous)
- `decision_function(X)`: Return shifted scores (negative = more anomalous)

## Parameters Guide

### n_estimators (default: 100)
Number of trees in the forest. The paper found that path lengths converge well before 100 trees.
- More trees = more stable results but slower
- 100 is a good default for most cases

### max_samples (default: 256)
Subsample size for each tree. The paper empirically found that 256 gives good results.
- Larger values don't significantly improve performance
- 256 = 2^8 provides sufficient tree depth

## Project Structure

```
isolation_forest/
├── isoforest/              # Main package
│   ├── __init__.py         # Package exports
│   ├── nodes.py            # Node structures (ExternalNode, InternalNode)
│   ├── tree.py             # IsolationTree class
│   ├── forest.py           # IsolationForest class
│   └── utils.py            # Utility functions (c-factor calculation)
├── tests/                  # Unit tests (99% coverage)
│   ├── test_nodes.py       # Node structure tests
│   ├── test_tree.py        # Isolation tree tests
│   ├── test_forest.py      # Isolation forest tests
│   └── test_utils.py       # Utility function tests
├── examples/               # Usage examples
│   └── basic_usage.py      # Basic usage demonstration
├── .gitignore              # Git ignore patterns
├── README.md               # This file
├── requirements.txt        # Dependencies
├── requirements-dev.txt    # Development dependencies
└── pytest.ini              # Pytest configuration
```

## Examples

Run the basic example:

```bash
python examples/basic_usage.py
```

## Development

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

Test coverage: 99%


## References

Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.

