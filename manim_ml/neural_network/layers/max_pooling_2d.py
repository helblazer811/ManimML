from manim import *
from manim_ml.gridded_rectangle import GriddedRectangle

from manim_ml.neural_network.layers.parent_layers import (
    ThreeDLayer,
    VGroupNeuralNetworkLayer,
)


class MaxPooling2DLayer(VGroupNeuralNetworkLayer, ThreeDLayer):
    """Max pooling layer for Convolutional2DLayer

    Note: This is for a Convolutional2DLayer even though
    it is called MaxPooling2DLayer because the 2D corresponds
    to the 2 spatial dimensions of the convolution.
    """

    def __init__(
        self,
        kernel_size=2,
        stride=1,
        cell_highlight_color=ORANGE,
        cell_width=0.2,
        filter_spacing=0.1,
        color=BLUE,
        show_grid_lines=False,
        stroke_width=2.0,
        **kwargs
    ):
        """Layer object for animating 2D Convolution Max Pooling

        Parameters
        ----------
        kernel_size : int or tuple, optional
            Width/Height of max pooling kernel, by default 2
        stride : int, optional
            Stride of the max pooling operation, by default 1
        """
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.cell_highlight_color = cell_highlight_color
        self.cell_width = cell_width
        self.filter_spacing = filter_spacing
        self.color = color
        self.show_grid_lines = show_grid_lines
        self.stroke_width = stroke_width

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        # Make the output feature maps
        self.feature_maps = self._make_output_feature_maps(
            input_layer.num_feature_maps, input_layer.feature_map_size
        )
        self.add(self.feature_maps)
        self.rotate(
            ThreeDLayer.rotation_angle,
            about_point=self.get_center(),
            axis=ThreeDLayer.rotation_axis,
        )
        self.feature_map_size = (
            input_layer.feature_map_size[0] / self.kernel_size,
            input_layer.feature_map_size[1] / self.kernel_size,
        )

    def _make_output_feature_maps(self, num_input_feature_maps, input_feature_map_size):
        """Makes a set of output feature maps"""
        # Compute the size of the feature maps
        output_feature_map_size = (
            input_feature_map_size[0] / self.kernel_size,
            input_feature_map_size[1] / self.kernel_size,
        )
        # Draw rectangles that are filled in with opacity
        feature_maps = []
        for filter_index in range(num_input_feature_maps):
            rectangle = GriddedRectangle(
                color=self.color,
                height=output_feature_map_size[1] * self.cell_width,
                width=output_feature_map_size[0] * self.cell_width,
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
