from manim import *
import numpy as np
from manim_ml.image import GrayscaleImageMobject
from manim_ml.neural_network.layers.parent_layers import NeuralNetworkLayer

from PIL import Image


class ImageLayer(NeuralNetworkLayer):
    """Single Image Layer for Neural Network"""

    def __init__(self, numpy_image, height=1.5, show_image_on_create=True, **kwargs):
        super().__init__(**kwargs)
        self.image_height = height
        self.numpy_image = numpy_image
        self.show_image_on_create = show_image_on_create

    def construct_layer(self, input_layer, output_layer):
        """Construct layer method

        Parameters
        ----------
        input_layer :
            Input layer
        output_layer :
            Output layer
        """
        if len(np.shape(self.numpy_image)) == 2:
            # Assumed Grayscale
            self.num_channels = 1
            self.image_mobject = GrayscaleImageMobject(
                self.numpy_image, height=self.image_height
            )
        elif len(np.shape(self.numpy_image)) == 3:
            # Assumed RGB
            self.num_channels = 3
            self.image_mobject = ImageMobject(self.numpy_image).scale_to_fit_height(
                self.image_height
            )
        self.add(self.image_mobject)

    @classmethod
    def from_path(cls, image_path, grayscale=True, **kwargs):
        """Creates a query using the paths"""
        # Load images from path
        image = Image.open(image_path)
        numpy_image = np.asarray(image)
        # Make the layer
        image_layer = cls(numpy_image, **kwargs)

        return image_layer

    @override_animation(Create)
    def _create_override(self, **kwargs):
        debug_mode = False
        if debug_mode:
            return FadeIn(SurroundingRectangle(self.image_mobject))
        if self.show_image_on_create:
            return FadeIn(self.image_mobject)
        else:
            return AnimationGroup()

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        return AnimationGroup()

    def get_right(self):
        """Override get right"""
        return self.image_mobject.get_right()

    def scale(self, scale_factor, **kwargs):
        """Scales the image mobject"""
        self.image_mobject.scale(scale_factor)

    @property
    def width(self):
        return self.image_mobject.width

    @property
    def height(self):
        return self.image_mobject.height
