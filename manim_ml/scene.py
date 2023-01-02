from manim import *


class ManimML3DScene(ThreeDScene):
    """
    This is a wrapper class for the Manim ThreeDScene

    Note: the primary purpose of this is to make it so
    that everything inside of a layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def play(self):
        """ """
        pass
