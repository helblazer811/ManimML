from manim import *
from manim_ml.neural_network.layers.embedding import EmbeddingLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer


class FeedForwardToEmbedding(ConnectiveLayer):
    """Feed Forward to Embedding Layer"""

    input_class = FeedForwardLayer
    output_class = EmbeddingLayer

    def __init__(
        self,
        input_layer,
        output_layer,
        animation_dot_color=RED,
        dot_radius=0.03,
        **kwargs
    ):
        super().__init__(input_layer, output_layer, **kwargs)
        self.feed_forward_layer = input_layer
        self.embedding_layer = output_layer
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        return super().construct_layer(input_layer, output_layer, **kwargs)

    def make_forward_pass_animation(self, layer_args={}, run_time=1.5, **kwargs):
        """Makes dots converge on a specific location"""
        # Find point to converge on by sampling from gaussian distribution
        location = self.embedding_layer.sample_point_location_from_distribution()
        # Set the embedding layer latent distribution
        # Move to location
        animations = []
        # Move the dots to the centers of each of the nodes in the FeedForwardLayer
        dots = []
        for node in self.feed_forward_layer.node_group:
            new_dot = Dot(
                node.get_center(),
                radius=self.dot_radius,
                color=self.animation_dot_color,
            )
            per_node_succession = Succession(
                Create(new_dot),
                new_dot.animate.move_to(location),
            )
            animations.append(per_node_succession)
            dots.append(new_dot)
        self.dots = VGroup(*dots)
        self.add(self.dots)
        # Follow up with remove animations
        remove_animations = []
        for dot in dots:
            remove_animations.append(FadeOut(dot))
        self.remove(self.dots)
        remove_animations = AnimationGroup(*remove_animations, run_time=0.2)
        animations = AnimationGroup(*animations)
        animation_group = Succession(animations, remove_animations, lag_ratio=1.0)

        return animation_group

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return AnimationGroup()
