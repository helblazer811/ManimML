from manim import *


class SurroundingRectangleTest(Scene):
    def construct(self):
        rectangle = Rectangle(width=1, height=1, color=WHITE, fill_opacity=1.0)
        self.add(rectangle)

        surrounding_rectangle = SurroundingRectangle(rectangle, color=GREEN, buff=0.0)
        self.add(surrounding_rectangle)
