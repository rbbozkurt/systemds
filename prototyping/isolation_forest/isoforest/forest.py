"""
Isolation Forest implementation.

This module contains the IsolationForest class which implements the main
anomaly detection algorithm using an ensemble of isolation trees.
"""
import numpy as np

from .tree import IsolationTree
from .utils import calculate_c_factor


class IsolationForest:
    """
    Isolation Forest anomaly detector.

    The forest builds multiple isolation trees on random subsamples.
    Anomalies will have shorter average path lengths across all trees.

    Attributes:
        n_estimators: Number of isolation trees
        max_samples: Subsampling size for each tree
        random_state: Random seed for reproducibility
        trees: List of fitted IsolationTree instances
    """

    def __init__(self, n_estimators=100, max_samples=256, random_state=None):
        """
        Initialize Isolation Forest.

        Args:
            n_estimators: Number of isolation trees (default optimum from paper: 100)
            max_samples: Subsampling size (default optimum from paper: 256)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.trees = []
        self._fitted = False

    def fit(self, X):
        """
        Build the Isolation Forest.

        This implements Algorithm 1 (iForest) from the paper.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]

        # Determine actual sample size (don't exceed available data)
        sample_size = min(self.max_samples, n_samples)

        # Height limit is ceiling(log2(sample_size))
        # We only grow trees to average height because we only care
        # about instances with shorter-than-average paths (anomalies)
        height_limit = int(np.ceil(np.log2(sample_size)))

        self.trees = []

        # Build each tree on a random subsample
        for _ in range(self.n_estimators):
            # Random sampling without replacement
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]

            # Create and fit an isolation tree
            tree = IsolationTree(height_limit)
            tree.root = tree.fit(X_sample)
            self.trees.append(tree)

        self._fitted = True
        self._sample_size = sample_size
        return self

    def _anomaly_score(self, avg_path_length):
        """
        Calculate anomaly score from average path length.

        This implements equation (2) from the paper:
        s(x,n) = 2^(-E(h(x))/c(n))

        The score is normalized to be between 0 and 1:
        - Scores close to 1 indicate anomalies
        - Scores close to 0 indicate normal instances
        - Scores around 0.5 mean the instance is ambiguous

        Args:
            avg_path_length: Average path length across all trees

        Returns:
            Anomaly score between 0 and 1
        """
        # Calculate c(n) for normalization
        c = calculate_c_factor(self._sample_size)

        # Return the anomaly score
        return np.power(2, -avg_path_length / c)

    def predict(self, X):
        """
        Predict anomaly labels (-1 for anomalies, 1 for normal).

        Args:
            X: Data to predict on

        Returns:
            Array of predictions: -1 for anomaly, 1 for normal
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        scores = self.score_samples(X)
        # Use 0.5 as threshold - instances with score > 0.5 are anomalies
        return np.where(scores > 0.5, -1, 1)

    def score_samples(self, X):
        """
        Calculate anomaly scores for samples.

        Args:
            X: Data to score

        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before scoring")

        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        # For each instance, get average path length across all trees
        for i in range(n_samples):
            path_lengths = []
            for tree in self.trees:
                path_length = tree.path_length(X[i], tree.root, 0)
                path_lengths.append(path_length)

            # Calculate average path length
            avg_path = np.mean(path_lengths)

            # Convert to anomaly score
            scores[i] = self._anomaly_score(avg_path)

        return scores

    def decision_function(self, X):
        """
        Average anomaly score - shifted to have negative scores for anomalies.

        This follows scikit-learn convention where more negative = more anomalous.

        Args:
            X: Data to score

        Returns:
            Array of shifted anomaly scores
        """
        return 0.5 - self.score_samples(X)

