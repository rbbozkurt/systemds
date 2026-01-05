"""
Unit tests for utility functions.
"""
import numpy as np
from isoforest.utils import calculate_c_factor


class TestCalculateCFactor:
    """Tests for calculate_c_factor function."""

    def test_n_zero(self):
        """Test c factor for n=0."""
        result = calculate_c_factor(0)
        assert result == 0

    def test_n_one(self):
        """Test c factor for n=1."""
        result = calculate_c_factor(1)
        assert result == 1

    def test_n_two(self):
        """Test c factor for n=2."""
        result = calculate_c_factor(2)
        expected = 2.0 * (np.log(1) + 0.5772156649) - (2.0 * 1 / 2)
        assert np.isclose(result, expected)

    def test_large_n(self):
        """Test c factor for large n."""
        result = calculate_c_factor(256)
        # Should be positive and reasonable
        assert result > 0
        assert result < 20  # Sanity check

    def test_monotonic(self):
        """Test that c factor increases with n."""
        values = [calculate_c_factor(n) for n in [2, 10, 50, 100, 256]]
        # Should be monotonically increasing
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]

