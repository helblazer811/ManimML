from manim import *
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.feed_forward_to_feed_forward import (
    FeedForwardToFeedForward,
)
from manim_ml.neural_network.layers.util import get_connective_layer


def test_get_connective_layer():
    """Tests get connective layer"""
    input_layer = FeedForwardLayer(3)
    output_layer = FeedForwardLayer(5)
    connective_layer = get_connective_layer(input_layer, output_layer)

    assert isinstance(connective_layer, FeedForwardToFeedForward)
