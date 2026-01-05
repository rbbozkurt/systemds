"""
Unit tests for node structures.
"""
import pytest
from isoforest.nodes import ExternalNode, InternalNode


class TestExternalNode:
    """Tests for ExternalNode."""

    def test_creation(self):
        """Test ExternalNode creation."""
        node = ExternalNode(size=10)
        assert node.size == 10

    def test_immutability(self):
        """Test that ExternalNode is immutable (namedtuple)."""
        node = ExternalNode(size=10)
        with pytest.raises(AttributeError):
            node.size = 20


class TestInternalNode:
    """Tests for InternalNode."""

    def test_creation(self):
        """Test InternalNode creation."""
        left = ExternalNode(size=5)
        right = ExternalNode(size=5)
        node = InternalNode(left, right, split_attr=0, split_value=1.5)

        assert node.left == left
        assert node.right == right
        assert node.split_attr == 0
        assert node.split_value == 1.5

    def test_attributes_mutable(self):
        """Test that InternalNode attributes can be modified."""
        left = ExternalNode(size=5)
        right = ExternalNode(size=5)
        node = InternalNode(left, right, split_attr=0, split_value=1.5)

        # Should be able to modify attributes
        node.split_value = 2.0
        assert node.split_value == 2.0

