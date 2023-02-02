from PIL import Image

from manim import *
from manim_ml.utils.mobjects.image import GrayscaleImageMobject
from manim_ml.neural_network.layers.parent_layers import ThreeDLayer


class TestImageHomotopy(Scene):
    def compute_shape_at_time(self, time):
        """Computes the shape of a transformed image at a given time"""

    def compute_center_offset_at_time(self, x, y, z, time):
        """Computes the center offset of a point at a given time"""
        pass

    def construct(self):
        image = Image.open("../assets/mnist/digit.jpeg")
        numpy_image = np.asarray(image)
        # Make nn
        image_mobject = GrayscaleImageMobject(numpy_image)
        self.add(image_mobject)
        self.wait(1)

        rot = image_mobject.animate.rotate(
            axis=[0, 1, 0], angle=ThreeDLayer.three_d_y_rotation
        )
        move = image_mobject.animate.move_to()
        self.play(rot)
        """
        # Make square
        square = Square()
        self.add(square)
        # Make polygon
        polygon = Polygon(
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
        )
        polygon.shift(RIGHT)
        self.play(
            Transform(square, polygon)
        )
        # Make the homotopy

        def shift_right_homotopy(x, y, z, t):
            return x + 1, y, z
        # Make the animation
        animation = Homotopy(
            mobject=image_mobject,
            homotopy=shift_right_homotopy
        )

        self.play(animation, run_time=1)
        """
