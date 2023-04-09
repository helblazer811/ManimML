from manim import *

from manim_ml.utils.mobjects.gridded_rectangle import GriddedRectangle
from manim_ml.neural_network.layers.parent_layers import (
    ThreeDLayer,
    VGroupNeuralNetworkLayer,
)
from manim_ml.neural_network.layers.convolutional_2d import FeatureMap

class TransposeConvolution2DLayer (VGroupNeuralNetworkLayer, ThreeDLayer):
    """Transpose convolution layer for Convolutional2DLayer"""

    def __init__(
            self,
            num_feature_maps,
            feature_map_size=None,
            filter_size=None,
            in_pad=1,
            padding=0,
            padding_dashed=True,
            kernel_size=3,
            stride=1,
            cell_highlight_color=ORANGE,
            cell_width=0.2,
            filter_spacing=0.1,
            color=BLUE,
            active_color=ORANGE,
            filter_color=ORANGE,
            show_grid_lines=False,
            fill_opacity=0.3,
            stroke_width=2.0,
            **kwargs
    ):
        """Layer object for animating 2D Transpose Convolution

        Parameters
        ----------
        kernel_size : int or tuple, optional
            Width/Height of transpose convolution kernel, by default 3
        stride : int, optional
            Stride of the transpose convolution operation, by default 1
        """

        super().__init__(**kwargs)

        if isinstance(padding, tuple):
            assert len(padding) == 2
            self.padding = padding
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            raise Exception(f"Unrecognized type for padding: {type(padding)}")

        # Add check for internal padding of transpose convolution
        if isinstance(in_pad, tuple):
            assert len(in_pad) == 2
            self.in_padding = in_pad
        elif isinstance(in_pad, int):
            self.in_padding = (in_pad, in_pad)
        else:
            raise Exception(f"Unrecognized type for padding: {type(in_pad)}")


        if isinstance(feature_map_size, int):
            self.feature_map_size = (feature_map_size, feature_map_size)
        else:
            self.feature_map_size = feature_map_size

        if isinstance(filter_size, int):
            self.filter_size = (filter_size, filter_size)
        else:
            self.filter_size = filter_size

        self.num_feature_maps = num_feature_maps
        self.kernel_size = kernel_size
        self.stride = stride
        self.cell_highlight_color = cell_highlight_color
        self.cell_width = cell_width
        self.filter_spacing = filter_spacing
        self.color = color
        self.active_color = active_color
        self.filter_color = filter_color
        self.show_grid_lines = show_grid_lines
        self.stroke_width = stroke_width
        self.fill_opacity = fill_opacity
        self.padding_dashed = padding_dashed

    def construct_layer(
            self,
            input_layer: "NeuralNetworkLayer",
            output_layer: "NeuralNetworkLayer",
            **kwargs
    ):
        # Make the output feature maps
        # self.feature_maps = self._make_output_feature_maps(
        #     input_layer.num_feature_maps, input_layer.feature_map_size
        # )

        self.feature_maps = self.construct_feature_maps(input_layer)

        self.add(self.feature_maps)
        self.rotate(
            ThreeDLayer.rotation_angle,
            about_point=self.get_center(),
            axis=ThreeDLayer.rotation_axis,
        )
        super().construct_layer(input_layer, output_layer, **kwargs)


    def construct_feature_maps(self, input_layer):
        """Creates the neural network layer"""
        # Draw rectangles that are filled in with opacity
        feature_maps = []

        feature_map_size = (
            (input_layer.feature_map_size[0] * (self.in_padding[0]+1) + self.in_padding[0] - self.filter_size[0] + 1)/ self.stride,
            (input_layer.feature_map_size[1] * (self.in_padding[1]+1) + self.in_padding[1] - self.filter_size[1] + 1)/ self.stride,
        )

        for filter_index in range(self.num_feature_maps):
            feature_map = FeatureMap(
                color=self.color,
                feature_map_size=feature_map_size,
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

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        """Makes forward pass of Max Pooling Layer.

        Parameters
        ----------
        layer_args : dict, optional
            _description_, by default {}
        """
        return AnimationGroup()

    @override_animation(Create)
    def _create_override(self, **kwargs):
        """Create animation for the MaxPooling operation"""
        pass
