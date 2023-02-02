from manim import *
from manim_ml.utils.mobjects.gridded_rectangle import GriddedRectangle


class TestGriddedRectangleScene(ThreeDScene):
    def construct(self):
        rect = GriddedRectangle(color=ORANGE, width=3, height=3)
        self.add(rect)
