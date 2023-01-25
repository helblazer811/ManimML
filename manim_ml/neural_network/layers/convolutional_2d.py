from typing import Union
from manim_ml.neural_network.activation_functions import get_activation_function_by_name
from manim_ml.neural_network.activation_functions.activation_function import (
    ActivationFunction,
)
import numpy as np
from manim import *

from manim_ml.neural_network.layers.parent_layers import (
    ThreeDLayer,
    VGroupNeuralNetworkLayer,
)
from manim_ml.gridded_rectangle import GriddedRectangle


class Convolutional2DLayer(VGroupNeuralNetworkLayer, ThreeDLayer):
    """Handles rendering a convolutional layer for a nn"""

    def __init__(
        self,
        num_feature_maps,
        feature_map_size=None,
        filter_size=None,
        cell_width=0.2,
        filter_spacing=0.1,
        color=BLUE,
        pulse_color=ORANGE,
        show_grid_lines=False,
        filter_color=ORANGE,
        stride=1,
        stroke_width=2.0,
        activation_function=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_feature_maps = num_feature_maps
        self.filter_color = filter_color
        if isinstance(feature_map_size, int):
            self.feature_map_size = (feature_map_size, feature_map_size)
        else:
            self.feature_map_size = feature_map_size
        if isinstance(filter_size, int):
            self.filter_size = (filter_size, filter_size)
        else:
            self.filter_size = filter_size
        self.cell_width = cell_width
        self.filter_spacing = filter_spacing
        self.color = color
        self.pulse_color = pulse_color
        self.stride = stride
        self.stroke_width = stroke_width
        self.show_grid_lines = show_grid_lines
        self.activation_function = activation_function

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs,
    ):
        # Make the feature maps
        self.feature_maps = self.construct_feature_maps()
        self.add(self.feature_maps)
        # Rotate stuff properly
        # normal_vector = self.feature_maps[0].get_normal_vector()
        self.rotate(
            ThreeDLayer.rotation_angle,
            about_point=self.get_center(),
            axis=ThreeDLayer.rotation_axis,
        )

        self.construct_activation_function()

    def construct_activation_function(self):
        """Construct the activation function"""
        # Add the activation function
        if not self.activation_function is None:
            # Check if it is a string
            if isinstance(self.activation_function, str):
                activation_function = get_activation_function_by_name(
                    self.activation_function
                )()
            else:
                assert isinstance(self.activation_function, ActivationFunction)
                activation_function = self.activation_function
            # Plot the function above the rest of the layer
            self.activation_function = activation_function
            self.add(self.activation_function)

    def construct_feature_maps(self):
        """Creates the neural network layer"""
        # Draw rectangles that are filled in with opacity
        feature_maps = []
        for filter_index in range(self.num_feature_maps):
            rectangle = GriddedRectangle(
                color=self.color,
                height=self.feature_map_size[1] * self.cell_width,
                width=self.feature_map_size[0] * self.cell_width,
                fill_color=self.color,
                fill_opacity=0.2,
                stroke_color=self.color,
                stroke_width=self.stroke_width,
                grid_xstep=self.cell_width,
                grid_ystep=self.cell_width,
                grid_stroke_width=self.stroke_width / 2,
                grid_stroke_color=self.color,
                show_grid_lines=self.show_grid_lines,
            )
            # Move the feature map
            rectangle.move_to([0, 0, filter_index * self.filter_spacing])
            # rectangle.set_z_index(4)
            feature_maps.append(rectangle)

        return VGroup(*feature_maps)

    def highlight_and_unhighlight_feature_maps(self):
        """Highlights then unhighlights feature maps"""
        return Succession(
            ApplyMethod(self.feature_maps.set_color, self.pulse_color),
            ApplyMethod(self.feature_maps.set_color, self.color),
        )

    def make_forward_pass_animation(
        self, run_time=5, corner_pulses=False, layer_args={}, **kwargs
    ):
        """Convolution forward pass animation"""
        # Note: most of this animation is done in the Convolution3DToConvolution3D layer
        if corner_pulses:
            raise NotImplementedError()
            passing_flashes = []
            for line in self.corner_lines:
                pulse = ShowPassingFlash(
                    line.copy().set_color(self.pulse_color).set_stroke(opacity=1.0),
                    time_width=0.5,
                    run_time=run_time,
                    rate_func=rate_functions.linear,
                )
                passing_flashes.append(pulse)

            # per_filter_run_time = run_time / len(self.feature_maps)
            # Make animation group
            animation_group = AnimationGroup(
                *passing_flashes,
                #    filter_flashes
            )
        else:
            if not self.activation_function is None:
                animation_group = AnimationGroup(
                    self.activation_function.make_evaluate_animation(),
                    self.highlight_and_unhighlight_feature_maps(),
                    lag_ratio=0.0,
                )
            else:
                animation_group = AnimationGroup()

        return animation_group

    def scale(self, scale_factor, **kwargs):
        self.cell_width *= scale_factor
        super().scale(scale_factor, **kwargs)

    def get_center(self):
        """Overrides function for getting center

        The reason for this is so that the center calculation
        does not include the activation function.
        """
        return self.feature_maps.get_center()

    def get_width(self):
        """Overrides get width function"""
        return self.feature_maps.length_over_dim(0)

    def get_height(self):
        """Overrides get height function"""
        return self.feature_maps.length_over_dim(1)

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return FadeIn(self.feature_maps)
