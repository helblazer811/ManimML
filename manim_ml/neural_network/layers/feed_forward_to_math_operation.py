from manim import *
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer
from manim_ml.neural_network.layers.math_operation_layer import MathOperationLayer
from manim_ml.utils.mobjects.connections import NetworkConnection

class FeedForwardToMathOperation(ConnectiveLayer):
    """Image Layer to FeedForward layer"""

    input_class = FeedForwardLayer
    output_class = MathOperationLayer

    def __init__(
        self,
        input_layer,
        output_layer,
        active_color=ORANGE,
        **kwargs
    ):
        self.active_color = active_color
        super().__init__(input_layer, output_layer, **kwargs)

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        # Draw an arrow from the output of the feed forward layer to the
        # input of the math operation layer
        self.connection = NetworkConnection(
            self.input_layer,
            self.output_layer,
            arc_direction="straight",
            buffer=0.05
        )
        self.add(self.connection)

        return super().construct_layer(input_layer, output_layer, **kwargs)

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        """Makes dots diverge from the given location and move to the feed forward nodes decoder"""
        # Make flashing pass animation on arrow
        passing_flash = ShowPassingFlash(
            self.connection.copy().set_color(self.active_color)
        )

        return passing_flash
