from manim import *


class TestSuccession(Scene):
    def construct(self):
        white_dot = Dot(color=WHITE)
        white_dot.shift(UP)

        red_dot = Dot(color=RED)

        self.play(
            Succession(
                Create(white_dot),
                white_dot.animate.shift(RIGHT),
                Create(red_dot),
                Wait(1),
                Uncreate(red_dot),
            )
        )
