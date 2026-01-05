"""
Utility functions for Isolation Forest algorithm.

This module contains helper functions used across the isolation forest implementation,
particularly for calculating average path lengths in binary search trees.
"""
import numpy as np


def calculate_c_factor(n):
    """
    Calculate the average path length of unsuccessful search in BST.

    This implements equation (1) from the Isolation Forest paper.
    It's used to normalize path lengths since we don't build complete trees.

    The formula: 2*H(n-1) - 2*(n-1)/n comes from BST analysis,
    where H(n-1) is the harmonic number, approximated by natural log.

    Args:
        n: Number of instances

    Returns:
        Average path length for n instances
    """
    if n <= 1:
        return 1 if n == 1 else 0
    # H(n-1) is the harmonic number, approximated by natural log + Euler's constant
    # 0.5772156649 is the Euler-Mascheroni constant
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

