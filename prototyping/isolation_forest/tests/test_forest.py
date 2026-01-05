"""
Unit tests for IsolationForest.
"""
import numpy as np
import pytest
from isoforest import IsolationForest


class TestIsolationForest:
    """Tests for IsolationForest class."""

    def test_initialization(self):
        """Test forest initialization."""
        forest = IsolationForest(n_estimators=50, max_samples=128, random_state=42)
        assert forest.n_estimators == 50
        assert forest.max_samples == 128
        assert forest.random_state == 42
        assert forest._fitted is False
        assert len(forest.trees) == 0

    def test_fit(self):
        """Test fitting the forest."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        forest = IsolationForest(n_estimators=10, random_state=42)
        forest.fit(X)

        assert forest._fitted is True
        assert len(forest.trees) == 10
        assert hasattr(forest, '_sample_size')

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        X = np.random.randn(100, 2)
        forest = IsolationForest(random_state=42)
        result = forest.fit(X)
        assert result is forest

    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error before fit."""
        forest = IsolationForest()
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="Model must be fitted"):
            forest.predict(X)

    def test_score_samples_before_fit_raises_error(self):
        """Test that score_samples raises error before fit."""
        forest = IsolationForest()
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="Model must be fitted"):
            forest.score_samples(X)

    def test_predict_shape(self):
        """Test that predict returns correct shape."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        X_test = np.random.randn(20, 2)

        forest = IsolationForest(random_state=42)
        forest.fit(X_train)
        predictions = forest.predict(X_test)

        assert predictions.shape == (20,)
        assert all(p in [-1, 1] for p in predictions)

    def test_score_samples_shape(self):
        """Test that score_samples returns correct shape."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        X_test = np.random.randn(20, 2)

        forest = IsolationForest(random_state=42)
        forest.fit(X_train)
        scores = forest.score_samples(X_test)

        assert scores.shape == (20,)
        assert all(0 <= s <= 1 for s in scores)

    def test_anomaly_detection(self):
        """Test that anomalies are detected."""
        np.random.seed(42)

        # Normal data clustered around origin
        X_normal = np.random.randn(200, 2) * 0.5

        # Anomalies far from origin
        X_anomalies = np.array([[5, 5], [-5, -5], [5, -5], [-5, 5]])

        X_train = X_normal
        X_test = np.vstack([X_normal[:10], X_anomalies])

        forest = IsolationForest(n_estimators=100, random_state=42)
        forest.fit(X_train)

        scores = forest.score_samples(X_test)

        # Anomalies should have higher scores than normal samples
        normal_scores = scores[:10]
        anomaly_scores = scores[10:]

        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    def test_decision_function_shape(self):
        """Test that decision_function returns correct shape."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        X_test = np.random.randn(20, 2)

        forest = IsolationForest(random_state=42)
        forest.fit(X_train)
        decisions = forest.decision_function(X_test)

        assert decisions.shape == (20,)

    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        X = np.random.randn(100, 2)

        forest1 = IsolationForest(n_estimators=10, random_state=42)
        forest1.fit(X)
        scores1 = forest1.score_samples(X)

        forest2 = IsolationForest(n_estimators=10, random_state=42)
        forest2.fit(X)
        scores2 = forest2.score_samples(X)

        np.testing.assert_array_equal(scores1, scores2)

