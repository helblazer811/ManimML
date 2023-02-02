from manim import *
from manim_ml.neural_network.layers.convolutional_2d_to_feed_forward import (
    Convolutional2DToFeedForward,
)
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.max_pooling_2d import MaxPooling2DLayer


class MaxPooling2DToFeedForward(Convolutional2DToFeedForward):
    """Feed Forward to Embedding Layer"""

    input_class = MaxPooling2DLayer
    output_class = FeedForwardLayer

    def __init__(
        self,
        input_layer: MaxPooling2DLayer,
        output_layer: FeedForwardLayer,
        passing_flash_color=ORANGE,
        **kwargs
    ):
        super().__init__(input_layer, output_layer, **kwargs)

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        return super().construct_layer(input_layer, output_layer, **kwargs)
