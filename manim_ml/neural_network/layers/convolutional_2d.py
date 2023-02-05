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
from manim_ml.utils.mobjects.gridded_rectangle import GriddedRectangle


class FeatureMap(VGroup):
    """Class for making a feature map"""

    def __init__(
        self,
        color=ORANGE,
        feature_map_size=None,
        fill_color=ORANGE,
        fill_opacity=0.2,
        cell_width=0.2,
        padding=(0, 0),
        stroke_width=2.0,
        show_grid_lines=False,
        padding_dashed=False,
    ):
        super().__init__()
        self.color = color
        self.feature_map_size = feature_map_size
        self.fill_color = fill_color
        self.fill_opacity = fill_opacity
        self.cell_width = cell_width
        self.padding = padding
        self.stroke_width = stroke_width
        self.show_grid_lines = show_grid_lines
        self.padding_dashed = padding_dashed
        # Check if we have non-zero padding
        if padding[0] > 0 or padding[1] > 0:
            # Make the exterior rectangle dashed
            width_with_padding = (
                self.feature_map_size[0] + self.padding[0] * 2
            ) * self.cell_width
            height_with_padding = (
                self.feature_map_size[1] + self.padding[1] * 2
            ) * self.cell_width
            self.untransformed_width = width_with_padding
            self.untransformed_height = height_with_padding

            self.exterior_rectangle = GriddedRectangle(
                color=self.color,
                width=width_with_padding,
                height=height_with_padding,
                fill_color=self.color,
                fill_opacity=self.fill_opacity,
                stroke_color=self.color,
                stroke_width=self.stroke_width,
                grid_xstep=self.cell_width,
                grid_ystep=self.cell_width,
                grid_stroke_width=self.stroke_width / 2,
                grid_stroke_color=self.color,
                show_grid_lines=self.show_grid_lines,
                dotted_lines=self.padding_dashed,
            )
            self.add(self.exterior_rectangle)
            # Add an interior rectangle with no fill color
            self.interior_rectangle = GriddedRectangle(
                color=self.color,
                fill_opacity=0.0,
                width=self.feature_map_size[0] * self.cell_width,
                height=self.feature_map_size[1] * self.cell_width,
                stroke_width=self.stroke_width,
            )
            self.add(self.interior_rectangle)
        else:
            # Just make an exterior rectangle with no dashes.
            self.untransformed_height = (self.feature_map_size[1] * self.cell_width,)
            self.untransformed_width = (self.feature_map_size[0] * self.cell_width,)
            # Make the exterior rectangle
            self.exterior_rectangle = GriddedRectangle(
                color=self.color,
                height=self.feature_map_size[1] * self.cell_width,
                width=self.feature_map_size[0] * self.cell_width,
                fill_color=self.color,
                fill_opacity=self.fill_opacity,
                stroke_color=self.color,
                stroke_width=self.stroke_width,
                grid_xstep=self.cell_width,
                grid_ystep=self.cell_width,
                grid_stroke_width=self.stroke_width / 2,
                grid_stroke_color=self.color,
                show_grid_lines=self.show_grid_lines,
            )
            self.add(self.exterior_rectangle)

    def get_corners_dict(self):
        """Returns a dictionary of the corners"""
        # Sort points through clockwise rotation of a vector in the xy plane
        return self.exterior_rectangle.get_corners_dict()


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
        active_color=ORANGE,
        filter_color=ORANGE,
        show_grid_lines=False,
        fill_opacity=0.3,
        stride=1,
        stroke_width=2.0,
        activation_function=None,
        padding=0,
        padding_dashed=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_feature_maps = num_feature_maps
        self.filter_color = filter_color
        if isinstance(padding, tuple):
            assert len(padding) == 2
            self.padding = padding
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            raise Exception(f"Unrecognized type for padding: {type(padding)}")

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
        self.active_color = active_color
        self.stride = stride
        self.stroke_width = stroke_width
        self.show_grid_lines = show_grid_lines
        self.activation_function = activation_function
        self.fill_opacity = fill_opacity
        self.padding_dashed = padding_dashed

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
        super().construct_layer(input_layer, output_layer, **kwargs)

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
            # Check if we need to add padding
            """
            feature_map = GriddedRectangle(
                color=self.color,
                height=self.feature_map_size[1] * self.cell_width,
                width=self.feature_map_size[0] * self.cell_width,
                fill_color=self.color,
                fill_opacity=self.fill_opacity,
                stroke_color=self.color,
                stroke_width=self.stroke_width,
                grid_xstep=self.cell_width,
                grid_ystep=self.cell_width,
                grid_stroke_width=self.stroke_width / 2,
                grid_stroke_color=self.color,
                show_grid_lines=self.show_grid_lines,
            )
            """
            # feature_map = GriddedRectangle()
            feature_map = FeatureMap(
                color=self.color,
                feature_map_size=self.feature_map_size,
                cell_width=self.cell_width,
                fill_color=self.color,
                fill_opacity=self.fill_opacity,
                padding=self.padding,
                padding_dashed=self.padding_dashed,
            )
            # Move the feature map
            feature_map.move_to([0, 0, filter_index * self.filter_spacing])
            # rectangle.set_z_index(4)
            feature_maps.append(feature_map)

        return VGroup(*feature_maps)

    def highlight_and_unhighlight_feature_maps(self):
        """Highlights then unhighlights feature maps"""
        return Succession(
            ApplyMethod(self.feature_maps.set_color, self.active_color),
            ApplyMethod(self.feature_maps.set_color, self.color),
        )

    def make_forward_pass_animation(self, run_time=5, layer_args={}, **kwargs):
        """Convolution forward pass animation"""
        # Note: most of this animation is done in the Convolution3DToConvolution3D layer
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

    def move_to(self, mobject_or_point):
        """Moves the center of the layer to the given mobject or point"""
        layer_center = self.feature_maps.get_center()
        if isinstance(mobject_or_point, Mobject):
            target_center = mobject_or_point.get_center() 
        else:
            target_center = mobject_or_point

        self.shift(target_center - layer_center)

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return FadeIn(self.feature_maps)
