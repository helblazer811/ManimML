from manim import *
from manim_ml.image import GrayscaleImageMobject
from manim_ml.neural_network.layers.parent_layers import NeuralNetworkLayer

class ImageLayer(NeuralNetworkLayer):
    """Single Image Layer for Neural Network"""

    def __init__(self, numpy_image, height=1.5, **kwargs):
        super().__init__(**kwargs)
        self.numpy_image = numpy_image
        if len(np.shape(self.numpy_image)) == 2:
            # Assumed Grayscale
            self.image_mobject = GrayscaleImageMobject(self.numpy_image, height=height)
        elif len(np.shape(self.numpy_image)) == 3:
            # Assumed RGB
            self.image_mobject = ImageMobject(self.numpy_image)
        self.add(self.image_mobject)

    @override_animation(Create)
    def _create_animation(self, **kwargs):
        return FadeIn(self.image_mobject)

    def make_forward_pass_animation(self):
        return Create(self.image_mobject)

    def move_to(self, location):
        """Override of move to"""
        self.image_mobject.move_to(location)

    def get_right(self):
        """Override get right"""
        return self.image_mobject.get_right()

    @property
    def width(self):
        return self.image_mobject.width