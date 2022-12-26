from manim import *
from manim_ml.neural_network.layers.parent_layers import ThreeDLayer, VGroupNeuralNetworkLayer
from manim_ml.gridded_rectangle import GriddedRectangle
import numpy as np

class Convolutional3DLayer(VGroupNeuralNetworkLayer, ThreeDLayer):
    """Handles rendering a convolutional layer for a nn"""

    def __init__(self, num_feature_maps, feature_map_width, feature_map_height, 
                filter_width, filter_height, cell_width=0.2, filter_spacing=0.1, color=BLUE, 
                pulse_color=ORANGE, filter_color=ORANGE, stride=1, stroke_width=2.0, **kwargs):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        self.num_feature_maps = num_feature_maps
        self.feature_map_height = feature_map_height
        self.filter_color = filter_color
        self.feature_map_width = feature_map_width
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.cell_width = cell_width
        self.filter_spacing = filter_spacing
        self.color = color
        self.pulse_color = pulse_color
        self.stride = stride
        self.stroke_width = stroke_width
        # Make the feature maps
        self.feature_maps = self.construct_feature_maps()
        self.add(self.feature_maps)

    def construct_feature_maps(self):
        """Creates the neural network layer"""
        # Draw rectangles that are filled in with opacity
        feature_maps = VGroup()
        for filter_index in range(self.num_feature_maps):
            rectangle = GriddedRectangle(
                center=[0, 0, filter_index * self.filter_spacing], # Center coordinate
                color=self.color, 
                height=self.feature_map_height * self.cell_width,
                width=self.feature_map_width * self.cell_width,
                fill_color=self.color,
                fill_opacity=0.2, 
                stroke_color=self.color,
                stroke_width=self.stroke_width,
                # grid_xstep=self.cell_width,
                # grid_ystep=self.cell_width,
                # grid_stroke_width=DEFAULT_STROKE_WIDTH/2
            )
            # Rotate about z axis
            rectangle.rotate_about_origin(
                90 * DEGREES, 
                np.array([0, 1, 0])
            ) 
            feature_maps.add(rectangle)
        
        return feature_maps

    def make_forward_pass_animation(
            self, 
            run_time=5, 
            corner_pulses=False,
            layer_args={}, 
            **kwargs
        ):
        """Convolution forward pass animation"""
        # Note: most of this animation is done in the Convolution3DToConvolution3D layer
        print(f"Corner pulses: {corner_pulses}")
        if corner_pulses:
            passing_flashes = []
            for line in self.corner_lines:
                pulse = ShowPassingFlash(
                    line.copy()
                        .set_color(self.pulse_color)
                        .set_stroke(opacity=1.0), 
                    time_width=0.5,
                    run_time=run_time,
                    rate_func=rate_functions.linear
                )
                passing_flashes.append(pulse)

            # per_filter_run_time = run_time / len(self.feature_maps)
            # Make animation group
            animation_group = AnimationGroup(
                *passing_flashes,
            #    filter_flashes
            )
        else:
            animation_group = AnimationGroup()

        return animation_group

    def scale(self, scale_factor, **kwargs):
        self.cell_width *= scale_factor
        super().scale(scale_factor, **kwargs)

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return FadeIn(self.feature_maps)
