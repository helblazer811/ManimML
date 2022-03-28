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
