from manim import *
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer, ThreeDLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer


class Convolutional2DToFeedForward(ConnectiveLayer, ThreeDLayer):
    """Feed Forward to Embedding Layer"""

    input_class = Convolutional2DLayer
    output_class = FeedForwardLayer

    def __init__(
        self,
        input_layer: Convolutional2DLayer,
        output_layer: FeedForwardLayer,
        passing_flash_color=ORANGE,
        **kwargs
    ):
        super().__init__(input_layer, output_layer, **kwargs)
        self.passing_flash_color = passing_flash_color

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        return super().construct_layer(input_layer, output_layer, **kwargs)

    def make_forward_pass_animation(self, layer_args={}, run_time=1.5, **kwargs):
        """Forward pass animation from conv2d to conv2d"""
        animations = []
        # Get input layer final feature map
        final_feature_map = self.input_layer.feature_maps[-1]
        # Get output layer nodes
        feed_forward_nodes = self.output_layer.node_group
        # Go through each corner
        corners = final_feature_map.get_corners_dict().values()
        for corner in corners:
            # Go through each node
            for node in feed_forward_nodes:
                line = Line(corner, node, stroke_width=1.0)
                line.set_z_index(self.output_layer.node_group.get_z_index())
                anim = ShowPassingFlash(
                    line.set_color(self.passing_flash_color), time_width=0.2
                )
                animations.append(anim)

        return AnimationGroup(*animations)
