"""
Unit tests for IsolationTree.
"""
import numpy as np
import pytest
from isoforest.tree import IsolationTree
from isoforest.nodes import ExternalNode, InternalNode


class TestIsolationTree:
    """Tests for IsolationTree class."""

    def test_initialization(self):
        """Test tree initialization."""
        tree = IsolationTree(height_limit=10)
        assert tree.height_limit == 10
        assert tree.root is None

    def test_fit_single_sample(self):
        """Test fitting with a single sample."""
        tree = IsolationTree(height_limit=10)
        X = np.array([[1.0, 2.0]])
        root = tree.fit(X)

        # Should create an external node
        assert isinstance(root, ExternalNode)
        assert root.size == 1

    def test_fit_at_height_limit(self):
        """Test that tree stops at height limit."""
        tree = IsolationTree(height_limit=0)
        X = np.random.randn(10, 2)
        root = tree.fit(X)

        # Should create an external node at height limit
        assert isinstance(root, ExternalNode)
        assert root.size == 10

    def test_fit_identical_samples(self):
        """Test fitting with identical samples."""
        tree = IsolationTree(height_limit=10)
        X = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        root = tree.fit(X)

        # Should create an external node (cannot split)
        assert isinstance(root, ExternalNode)
        assert root.size == 3

    def test_fit_creates_internal_node(self):
        """Test that fit creates internal nodes when possible."""
        np.random.seed(42)
        tree = IsolationTree(height_limit=10)
        X = np.random.randn(100, 2)
        root = tree.fit(X)

        # With 100 samples and height limit 10, should create internal node
        assert isinstance(root, InternalNode)
        assert hasattr(root, 'left')
        assert hasattr(root, 'right')
        assert hasattr(root, 'split_attr')
        assert hasattr(root, 'split_value')

    def test_path_length_external_node(self):
        """Test path length calculation for external node."""
        tree = IsolationTree(height_limit=10)
        node = ExternalNode(size=5)
        x = np.array([1.0, 2.0])

        path_len = tree.path_length(x, node, current_height=3)
        # Should be current_height + c(size)
        assert path_len > 3

    def test_path_length_internal_node(self):
        """Test path length calculation through internal node."""
        np.random.seed(42)
        tree = IsolationTree(height_limit=10)
        X = np.random.randn(100, 2)
        root = tree.fit(X)
        tree.root = root

        x = np.array([0.0, 0.0])
        path_len = tree.path_length(x, root, current_height=0)

        # Should be a reasonable path length
        assert path_len >= 0
        assert path_len < 20  # Sanity check

