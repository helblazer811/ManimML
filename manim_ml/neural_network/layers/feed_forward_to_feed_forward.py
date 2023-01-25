from typing import List, Union
import numpy as np

from manim import *
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer


class FeedForwardToFeedForward(ConnectiveLayer):
    """Layer for connecting FeedForward layer to FeedForwardLayer"""

    input_class = FeedForwardLayer
    output_class = FeedForwardLayer

    def __init__(
        self,
        input_layer,
        output_layer,
        passing_flash=True,
        dot_radius=0.05,
        animation_dot_color=RED,
        edge_color=WHITE,
        edge_width=1.5,
        camera=None,
        **kwargs
    ):
        super().__init__(input_layer, output_layer, **kwargs)
        self.passing_flash = passing_flash
        self.edge_color = edge_color
        self.dot_radius = dot_radius
        self.animation_dot_color = animation_dot_color
        self.edge_width = edge_width

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        self.edges = self.construct_edges()
        self.add(self.edges)

    def construct_edges(self):
        # Go through each node in the two layers and make a connecting line
        edges = []
        for node_i in self.input_layer.node_group:
            for node_j in self.output_layer.node_group:
                line = Line(
                    node_i.get_center(),
                    node_j.get_center(),
                    color=self.edge_color,
                    stroke_width=self.edge_width,
                )
                edges.append(line)

        edges = VGroup(*edges)
        return edges

    @override_animation(FadeOut)
    def _fadeout_animation(self):
        animations = []

        for edge in self.edges:
            animations.append(FadeOut(edge))

        animation_group = AnimationGroup(*animations)

        return animation_group

    def make_forward_pass_animation(
        self, layer_args={}, run_time=1, feed_forward_dropout=0.0, **kwargs
    ):
        """Animation for passing information from one FeedForwardLayer to the next"""
        path_animations = []
        dots = []
        for edge_index, edge in enumerate(self.edges):
            if (
                not "edge_indices_to_dropout" in layer_args
                or not edge_index in layer_args["edge_indices_to_dropout"]
            ):
                dot = Dot(
                    color=self.animation_dot_color,
                    fill_opacity=1.0,
                    radius=self.dot_radius,
                )
                # Add to dots group
                dots.append(dot)
                # Make the animation
                if self.passing_flash:
                    copy_edge = edge.copy()
                    anim = ShowPassingFlash(
                        copy_edge.set_color(self.animation_dot_color), time_width=0.2
                    )
                else:
                    anim = MoveAlongPath(
                        dot, edge, run_time=run_time, rate_function=sigmoid
                    )
                path_animations.append(anim)

        if not self.passing_flash:
            dots = VGroup(*dots)
            self.add(dots)

        path_animations = AnimationGroup(*path_animations)

        return path_animations

    def modify_edge_colors(self, colors=None, magnitudes=None, color_scheme="inferno"):
        """Changes the colors of edges"""
        # TODO implement
        pass

    def modify_edge_stroke_widths(self, widths):
        """Changes the widths of the edges"""
        assert len(widths) > 0
        # Note: 1d-arrays are assumed to be in row major order
        widths = np.array(widths)
        widths = np.flatten(widths)
        # Check thickness size
        assert np.shape(widths)[0] == len(self.edges)
        # Make animation
        animations = []

        for index, edge in enumerate(self.edges):
            width = widths[index]
            change_width = edge.animate.set_stroke_width(width)
            animations.append(change_width)

        animation_group = AnimationGroup(*animations)

        return animation_group

    @override_animation(Create)
    def _create_override(self, **kwargs):
        animations = []

        for edge in self.edges:
            animations.append(Create(edge))

        animation_group = AnimationGroup(*animations, lag_ratio=0.0)
        return animation_group
