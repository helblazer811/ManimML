import numpy as np

from manim import *
from manim_ml.neural_network.layers.convolutional3d import Convolutional3DLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.layers.parent_layers import ThreeDLayer, VGroupNeuralNetworkLayer
from manim_ml.gridded_rectangle import GriddedRectangle

class ImageToConvolutional3DLayer(VGroupNeuralNetworkLayer, ThreeDLayer):
    """Handles rendering a convolutional layer for a nn"""
    input_class = ImageLayer
    output_class = Convolutional3DLayer

    def __init__(self, input_layer: ImageLayer, output_layer: Convolutional3DLayer, **kwargs):
        super().__init__(input_layer, output_layer, **kwargs)
        self.input_layer = input_layer
        self.output_layer = output_layer

    def make_forward_pass_animation(
            self, 
            run_time=5, 
            layer_args={}, 
            **kwargs
        ):
        """Maps image to convolutional layer"""
        # Transform the image from the input layer to the
        num_image_channels = self.input_layer.num_channels
        if num_image_channels == 3:
            return self.rbg_image_animation()
        elif num_image_channels == 1:
            return self.grayscale_image_animation()
        else:
            raise Exception(f"Unrecognized number of image channels: {num_image_channels}")

    def rbg_image_animation(self):
        """Handles animation for 3 channel image"""
        image_mobject = self.input_layer.image_mobject
        # TODO get each color channel and turn it into an image
        # TODO create image mobjects for each channel and transform
        # it to the feature maps of the output_layer
        raise NotImplementedError()
        pass

    def grayscale_image_animation(self):
        """Handles animation for 1 channel image"""
        animations = []
        image_mobject = self.input_layer.image_mobject
        target_feature_map = self.output_layer.feature_maps[0]
        # Make the object 3D by adding it back into camera frame
        def remove_fixed_func(image_mobject):
            # self.camera.remove_fixed_orientation_mobjects(image_mobject)
            # self.camera.remove_fixed_in_frame_mobjects(image_mobject)
            return image_mobject

        remove_fixed = ApplyFunction(
            remove_fixed_func,
            image_mobject
        )
        animations.append(remove_fixed)
        # Make a transformation of the image_mobject to the first feature map
        input_to_feature_map_transformation = Transform(image_mobject, target_feature_map)
        animations.append(input_to_feature_map_transformation)
        # Make the object fixed in 2D again
        def make_fixed_func(image_mobject):
            # self.camera.add_fixed_orientation_mobjects(image_mobject)
            # self.camera.add_fixed_in_frame_mobjects(image_mobject)
            return image_mobject
            
        make_fixed = ApplyFunction(
            make_fixed_func,
            image_mobject
        )
        animations.append(make_fixed)

        return AnimationGroup()
        
        return AnimationGroup(*animations)

    def scale(self, scale_factor, **kwargs):
        super().scale(scale_factor, **kwargs)

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return AnimationGroup()