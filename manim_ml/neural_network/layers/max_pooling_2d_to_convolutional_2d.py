import numpy as np
from manim import *

from manim_ml.neural_network.layers.convolutional_2d_to_convolutional_2d import (
    Convolutional2DToConvolutional2D,
    Filters,
)
from manim_ml.neural_network.layers.max_pooling_2d import MaxPooling2DLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer, ThreeDLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer

from manim.utils.space_ops import rotation_matrix


class MaxPooling2DToConvolutional2D(Convolutional2DToConvolutional2D):
    """Feed Forward to Embedding Layer"""

    input_class = MaxPooling2DLayer
    output_class = Convolutional2DLayer

    def __init__(
        self,
        input_layer: MaxPooling2DLayer,
        output_layer: Convolutional2DLayer,
        passing_flash_color=ORANGE,
        cell_width=1.0,
        stroke_width=2.0,
        show_grid_lines=False,
        **kwargs
    ):
        input_layer.num_feature_maps = output_layer.num_feature_maps
        super().__init__(input_layer, output_layer, **kwargs)
        self.passing_flash_color = passing_flash_color
        self.cell_width = cell_width
        self.stroke_width = stroke_width
        self.show_grid_lines = show_grid_lines

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        """Constructs the MaxPooling to Convolution3D layer

        Parameters
        ----------
        input_layer : NeuralNetworkLayer
            input layer
        output_layer : NeuralNetworkLayer
            output layer
        """
        pass
