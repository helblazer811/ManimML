from manim import *
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer


class ImageToFeedForward(ConnectiveLayer):
    """Image Layer to FeedForward layer"""

    input_class = ImageLayer
    output_class = FeedForwardLayer

    def __init__(
        self,
        input_layer,
        output_layer,
        animation_dot_color=RED,
        dot_radius=0.05,
        **kwargs
    ):
        super().__init__(input_layer, output_layer, **kwargs)
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius

        self.feed_forward_layer = output_layer
        self.image_layer = input_layer

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        return super().construct_layer(input_layer, output_layer, **kwargs)

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        """Makes dots diverge from the given location and move to the feed forward nodes decoder"""
        animations = []
        dots = []
        image_mobject = self.image_layer.image_mobject
        # Move the dots to the centers of each of the nodes in the FeedForwardLayer
        image_location = image_mobject.get_center()
        for node in self.feed_forward_layer.node_group:
            new_dot = Dot(
                image_location, radius=self.dot_radius, color=self.animation_dot_color
            )
            per_node_succession = Succession(
                Create(new_dot),
                new_dot.animate.move_to(node.get_center()),
            )
            animations.append(per_node_succession)
            dots.append(new_dot)

        animation_group = AnimationGroup(*animations)
        return animation_group

    @override_animation(Create)
    def _create_override(self):
        return AnimationGroup()
