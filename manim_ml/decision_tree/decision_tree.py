"""
    Module for visualizing decision trees in Manim. 
    It parses a decision tree classifier from sklearn. 
"""
from manim import *
from manim_ml.one_to_one_sync import OneToOneSync

class LeafNode(VGroup):
    pass

class SplitNode(VGroup):
    pass

class DecisionTreeDiagram(Graph):
    """Decision Tree Digram Class for Manim"""
    pass

class DecisionTreeEmbedding():
    """Embedding for the decision tree"""
    pass

class DecisionTreeContainer(OneToOneSync):
    """Connects the DecisionTreeDiagram to the DecisionTreeEmbedding"""

    def __init__(self):
        pass
