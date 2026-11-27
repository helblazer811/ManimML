from typing import List, Union
import numpy as np

from manim import *
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer
import manim_ml

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
        animation_dot_color=manim_ml.config.color_scheme.active_color,
        edge_color=manim_ml.config.color_scheme.secondary_color,
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
        super().construct_layer(input_layer, output_layer, **kwargs)

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
                        copy_edge.set_color(self.animation_dot_color), time_width=0.3
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

    def modify_edge_colors(self, colors=None, magnitudes=None, positive_color="#38BDF8", negative_color="#F43F5E"):
        """Changes the colors of edges based on explicit colors or signed weight/gradient magnitudes."""
        animations = []
        if colors is not None:
            flat_colors = list(np.asarray(colors).flatten())
            assert len(flat_colors) == len(self.edges)
            for edge, col in zip(self.edges, flat_colors):
                animations.append(edge.animate.set_color(col))
        elif magnitudes is not None:
            flat_mags = np.asarray(magnitudes, dtype=float).flatten()
            assert len(flat_mags) == len(self.edges)
            for edge, val in zip(self.edges, flat_mags):
                col = positive_color if val >= 0 else negative_color
                animations.append(edge.animate.set_color(col))
        return AnimationGroup(*animations)

    def make_backward_pass_animation(self, layer_args={}, run_time=1.0, color="#F43F5E", **kwargs):
        """Animates error gradients flowing backward from output_layer to input_layer."""
        path_animations = []
        rt = run_time if run_time is not None else 1.0
        for edge_index, edge in enumerate(self.edges):
            if (
                "edge_indices_to_dropout" not in layer_args
                or edge_index not in layer_args["edge_indices_to_dropout"]
            ):
                rev_edge = Line(
                    edge.get_end(),
                    edge.get_start(),
                    color=color,
                    stroke_width=self.edge_width * 1.4,
                )
                if self.passing_flash:
                    anim = ShowPassingFlash(rev_edge, time_width=0.35, run_time=rt)
                else:
                    dot = Dot(color=color, fill_opacity=1.0, radius=self.dot_radius)
                    anim = MoveAlongPath(dot, rev_edge, run_time=rt, rate_function=sigmoid)
                path_animations.append(anim)
        return AnimationGroup(*path_animations)

    def modify_edge_stroke_widths(self, widths):
        """Changes the widths of the edges"""
        assert len(widths) > 0
        # Note: 1d-arrays are assumed to be in row major order
        widths = np.array(widths)
        widths = np.asarray(widths).flatten()
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
