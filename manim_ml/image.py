from manim import *
import numpy as np
from PIL import Image

class GrayscaleImageMobject(ImageMobject):
    """Mobject for creating images in Manim from numpy arrays"""

    def __init__(self, numpy_image, height=2.3):
        self.numpy_image = numpy_image

        assert len(np.shape(self.numpy_image)) == 2 
        input_image = self.numpy_image[None, :, :]
        # Convert grayscale to rgb version of grayscale
        input_image = np.repeat(input_image, 3, axis=0)
        input_image = np.rollaxis(input_image, 0, start=3)

        super().__init__(input_image, image_mode="RGB")

        self.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.scale_to_fit_height(height)

    @classmethod
    def from_path(cls, path, height=2.3):
        """Loads image from path"""
        image = Image.open(path)
        numpy_image = np.asarray(image)

        return cls(numpy_image, height=height)

    @override_animation(Create)
    def create(self, run_time=2):
        return FadeIn(self)
