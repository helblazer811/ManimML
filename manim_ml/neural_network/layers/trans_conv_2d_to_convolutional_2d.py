from manim import *

from manim_ml.neural_network.layers.convolutional_2d_to_convolutional_2d import Convolutional2DToConvolutional2D
from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.trans_conv_2d import TransposeConvolution2DLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer, ThreeDLayer


class TransConv2DToConvolutional2D(Convolutional2DToConvolutional2D):

    input_class = TransposeConvolution2DLayer
    output_class = Convolutional2DLayer

    def __init__(
        self,
        input_layer: TransposeConvolution2DLayer,
        output_layer: Convolutional2DLayer,
        color=ORANGE,
        filter_opacity=0.3,
        line_color=ORANGE,
        active_color=ORANGE,
        cell_width=0.2,
        show_grid_lines=True,
        highlight_color=ORANGE,
        **kwargs,
    ):
        super().__init__(
            input_layer,
            output_layer,
            **kwargs,
        )
        self.color = color
        self.filter_color = self.output_layer.filter_color
        self.filter_size = self.output_layer.filter_size
        self.feature_map_size = self.input_layer.feature_map_size
        self.num_input_feature_maps = self.input_layer.num_feature_maps
        self.num_output_feature_maps = self.output_layer.num_feature_maps
        self.cell_width = self.output_layer.cell_width
        self.stride = self.output_layer.stride
        self.padding = self.input_layer.padding
        self.filter_opacity = filter_opacity
        self.cell_width = cell_width
        self.line_color = line_color
        self.active_color = active_color
        self.show_grid_lines = show_grid_lines
        self.highlight_color = highlight_color

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs,
    ):
        return super().construct_layer(input_layer, output_layer, **kwargs)
    
    def make_forward_pass_animation(
        self,
        layer_args={},
        all_filters_at_once=False,
        highlight_active_feature_map=True,
        run_time=10.5,
        **kwargs,
        ):

        # Just run the usual 2D convolution (this may need to have parameters tweaked)
        super().make_forward_pass_animation(
            layer_args, all_filters_at_once, highlight_active_feature_map, run_time,
            **kwargs)