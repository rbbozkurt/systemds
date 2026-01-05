"""
Isolation Tree implementation.

This module contains the IsolationTree class which implements a single
isolation tree that recursively partitions data.
"""
import numpy as np

from .nodes import ExternalNode, InternalNode
from .utils import calculate_c_factor


class IsolationTree:
    """
    A single Isolation Tree that recursively partitions data.

    The key insight: anomalies are easier to isolate (require fewer partitions)
    because they are 'few and different' from normal instances.

    Attributes:
        height_limit: Maximum depth the tree will grow to
        root: Root node of the tree
    """

    def __init__(self, height_limit):
        """
        Initialize an Isolation Tree.

        Args:
            height_limit: Maximum depth for tree growth
        """
        self.height_limit = height_limit
        self.root = None

    def fit(self, X, current_height=0):
        """
        Build the isolation tree by recursively partitioning X.

        This implements Algorithm 2 (iTree) from the paper.

        Args:
            X: Input data array of shape (n_samples, n_features)
            current_height: Current depth in the tree (starts at 0)

        Returns:
            The root node of the constructed tree
        """
        n_samples, n_features = X.shape

        # Termination conditions from the paper:
        # 1. Reached height limit (we only care about short paths)
        # 2. Only one instance left (fully isolated)
        # 3. All instances are identical (cannot split further)
        if current_height >= self.height_limit or n_samples <= 1:
            return ExternalNode(size=n_samples)

        # Check if all values are the same (cannot split)
        if len(np.unique(X, axis=0)) == 1:
            return ExternalNode(size=n_samples)

        # Randomly select an attribute to split on
        split_attr = np.random.randint(0, n_features)

        # Get min and max values for the selected attribute
        attr_values = X[:, split_attr]
        min_val = attr_values.min()
        max_val = attr_values.max()

        # If min equals max, this attribute cannot split the data
        if min_val == max_val:
            return ExternalNode(size=n_samples)

        # Randomly select a split point between min and max
        split_value = np.random.uniform(min_val, max_val)

        # Partition data based on the split
        left_mask = attr_values < split_value
        X_left = X[left_mask]
        X_right = X[~left_mask]

        # Recursively build left and right subtrees
        left_child = self.fit(X_left, current_height + 1)
        right_child = self.fit(X_right, current_height + 1)

        return InternalNode(left_child, right_child, split_attr, split_value)

    def path_length(self, x, node, current_height):
        """
        Calculate path length for a single instance.

        This implements Algorithm 3 (PathLength) from the paper.

        The path length is the number of edges traversed from root to
        termination. Shorter paths indicate anomalies.

        Args:
            x: Single instance to evaluate
            node: Current node in traversal
            current_height: Current path length (number of edges)

        Returns:
            Path length for this instance
        """
        # If we've reached an external node, return the path length
        # plus an adjustment for the unbuilt subtree
        if isinstance(node, ExternalNode):
            # c(n) adjustment accounts for the average path length
            # of a BST with n nodes that wasn't actually built
            return current_height + calculate_c_factor(node.size)

        # Internal node - continue traversal based on split
        split_attr = node.split_attr
        if x[split_attr] < node.split_value:
            return self.path_length(x, node.left, current_height + 1)
        else:
            return self.path_length(x, node.right, current_height + 1)

