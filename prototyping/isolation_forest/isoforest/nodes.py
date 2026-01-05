"""
Node structures for Isolation Trees.

This module defines the node types used in isolation trees:
- ExternalNode: Leaf nodes where instances are isolated
- InternalNode: Internal nodes containing split information
"""
from collections import namedtuple


# External nodes represent leaf nodes where instances are isolated
ExternalNode = namedtuple('ExternalNode', ['size'])


class InternalNode:
    """
    Internal node containing split information and pointers to child nodes.

    Attributes:
        left: Left child node
        right: Right child node
        split_attr: Index of the attribute used for splitting
        split_value: Value used for the split decision
    """

    def __init__(self, left, right, split_attr, split_value):
        self.left = left
        self.right = right
        self.split_attr = split_attr
        self.split_value = split_value

