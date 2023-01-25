import numpy as np

from manim import *
from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.layers.parent_layers import (
    ThreeDLayer,
    VGroupNeuralNetworkLayer,
)
from manim_ml.gridded_rectangle import GriddedRectangle


class ImageToConvolutional2DLayer(VGroupNeuralNetworkLayer, ThreeDLayer):
    """Handles rendering a convolutional layer for a nn"""

    input_class = ImageLayer
    output_class = Convolutional2DLayer

    def __init__(
        self, input_layer: ImageLayer, output_layer: Convolutional2DLayer, **kwargs
    ):
        super().__init__(input_layer, output_layer, **kwargs)
        self.input_layer = input_layer
        self.output_layer = output_layer

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs,
    ):
        return super().construct_layer(input_layer, output_layer, **kwargs)

    def make_forward_pass_animation(self, run_time=5, layer_args={}, **kwargs):
        """Maps image to convolutional layer"""
        # Transform the image from the input layer to the
        num_image_channels = self.input_layer.num_channels
        if num_image_channels == 1 or num_image_channels == 3:  # TODO fix this later
            return self.grayscale_image_forward_pass_animation()
        elif num_image_channels == 3:
            return self.rbg_image_forward_pass_animation()
        else:
            raise Exception(
                f"Unrecognized number of image channels: {num_image_channels}"
            )

    def rbg_image_forward_pass_animation(self):
        """Handles forward pass animation for 3 channel image"""
        image_mobject = self.input_layer.image_mobject
        # TODO get each color channel and turn it into an image
        # TODO create image mobjects for each channel and transform
        # it to the feature maps of the output_layer
        raise NotImplementedError()

    def grayscale_image_forward_pass_animation(self):
        """Handles forward pass animation for 1 channel image"""
        animations = []
        image_mobject = self.input_layer.image_mobject
        target_feature_map = self.output_layer.feature_maps[0]
        # Map image mobject to feature map
        # Make rotation of image
        rotation = ApplyMethod(
            image_mobject.rotate,
            ThreeDLayer.rotation_angle,
            ThreeDLayer.rotation_axis,
            image_mobject.get_center(),
            run_time=0.5,
        )
        """
        x_rotation = ApplyMethod(
            image_mobject.rotate,
            ThreeDLayer.three_d_x_rotation,
            [1, 0, 0], 
            image_mobject.get_center(),
            run_time=0.5
        )
        y_rotation = ApplyMethod(
            image_mobject.rotate,
            ThreeDLayer.three_d_y_rotation,
            [0, 1, 0], 
            image_mobject.get_center(),
            run_time=0.5
        )
        """
        # Set opacity
        set_opacity = ApplyMethod(image_mobject.set_opacity, 0.2, run_time=0.5)
        # Scale the max of width or height to the
        # width of the feature_map
        def scale_image_func(image_mobject):
            max_width_height = max(image_mobject.width, image_mobject.height)
            scale_factor = target_feature_map.untransformed_width / max_width_height
            image_mobject.scale(scale_factor)

            return image_mobject

        scale_image = ApplyFunction(scale_image_func, image_mobject)
        # scale_image = ApplyMethod(image_mobject.scale, scale_factor, run_time=0.5)
        # Move the image
        move_image = ApplyMethod(image_mobject.move_to, target_feature_map)
        # Compose the animations
        animation = Succession(
            rotation,
            scale_image,
            set_opacity,
            move_image,
        )
        return animation

    def scale(self, scale_factor, **kwargs):
        super().scale(scale_factor, **kwargs)

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return AnimationGroup()
