from manim import *
import numpy as np

def construct_image_mobject(input_image, height=2.3):
    """Constructs an ImageMobject from a numpy grayscale image"""
    # Convert image to rgb
    if len(input_image.shape) == 2:
        input_image = np.repeat(input_image, 3, axis=0)
        input_image = np.rollaxis(input_image, 0, start=3)
    #  Make the ImageMobject
    image_mobject = ImageMobject(input_image, image_mode="RGB")
    image_mobject.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    image_mobject.height = height

    return image_mobject

class NumpyImageMobject(ImageMobject):
    """Mobject for creating images in Manim from numpy arrays"""

    def __init__(self, numpy_image, height=2.3, grayscale=False):
        self.numpy_image = numpy_image
        self.height = height

        if grayscale:
            assert len(input_image.shape) == 2 
            input_image = np.repeat(self.numpy_image, 3, axis=0)
            input_image = np.rollaxis(input_image, 0, start=3)

        super().__init__(input_image, image_mode="RGB")

        self.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.height = height

