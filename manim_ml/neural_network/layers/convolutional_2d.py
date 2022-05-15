from manim import *
from matplotlib import animation
from xarray import align
from manim_ml.neural_network.layers.parent_layers import VGroupNeuralNetworkLayer

class Convolutional2DLayer(VGroupNeuralNetworkLayer):
    
    def __init__(self, feature_map_height, feature_map_width, filter_width, filter_height,
            stride=1, cell_width=0.5, pixel_width=0.5, feature_map_color=BLUE, filter_color=ORANGE, 
            **kwargs):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        self.feature_map_height = feature_map_height
        self.feature_map_width = feature_map_width
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.pixel_width = pixel_width
        self.feature_map_color = feature_map_color
        self.filter_color = filter_color
        self.stride = stride
        self.cell_width = cell_width
        # Construct the input
        self.construct_feature_map()

    def construct_feature_map(self):
        """Makes feature map"""
        # Make feature map rectangle
        self.feature_map = Rectangle(
            width=self.feature_map_width * self.cell_width,
            height=self.feature_map_height * self.cell_width, 
            color=self.feature_map_color,
            grid_xstep=self.cell_width,
            grid_ystep=self.cell_width
        )

        self.add(self.feature_map)

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return FadeIn(self.feature_map)

    def make_forward_pass_animation(self, **kwargs):
        """Make feed forward animation"""
        return AnimationGroup()