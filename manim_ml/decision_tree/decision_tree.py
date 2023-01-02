"""
    Module for visualizing decision trees in Manim. 
    It parses a decision tree classifier from sklearn. 
"""
from manim import *
from manim_ml.one_to_one_sync import OneToOneSync


class LeafNode(VGroup):
    pass


class NonLeafNode(VGroup):
    pass


class DecisionTreeDiagram(Graph):
    """Decision Tree Digram Class for Manim"""

    pass


class DecisionTreeEmbedding:
    """Embedding for the decision tree"""

    pass


class DecisionTreeContainer(OneToOneSync):
    """Connects the DecisionTreeDiagram to the DecisionTreeEmbedding"""

    def __init__(self, sklearn_tree, points, classes):
        self.sklearn_tree = sklearn_tree
        self.points = points
        self.classes = classes

    def make_unfold_tree_animation(self):
        """Unfolds the tree through an in order traversal

        This animations unfolds the tree diagram as well as showing the splitting
        of a shaded region in the Decision Tree embedding.
        """
        # Draw points in the embedding
        # Start the tree splitting animation
        pass
