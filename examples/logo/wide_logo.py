"""
    Logo for Manim Machine Learning
"""
from manim import *

import manim_ml
manim_ml.config.color_scheme = "light_mode"

from manim_ml.neural_network.architectures.feed_forward import FeedForwardNeuralNetwork

config.pixel_height = 1000
config.pixel_width = 2000
config.frame_height = 4.0
config.frame_width = 8.0


class ManimMLLogo(Scene):
    def construct(self):
        self.text = Text("ManimML", color=manim_ml.config.color_scheme.text_color)
        self.text.scale(1.0)
        self.neural_network = FeedForwardNeuralNetwork(
            [3, 5, 3, 6, 3], layer_spacing=0.3, node_color=BLUE
        )
        self.neural_network.scale(0.8)
        self.neural_network.next_to(self.text, RIGHT, buff=0.5)
        # self.neural_network.move_to(self.text.get_right())
        # self.neural_network.shift(1.25 * DOWN)
        self.logo_group = Group(self.text, self.neural_network)
        self.logo_group.scale(1.0)
        self.logo_group.move_to(ORIGIN)
        self.play(Write(self.text), run_time=1.0)
        self.play(Create(self.neural_network), run_time=3.0)
        # self.surrounding_rectangle = SurroundingRectangle(self.logo_group, buff=0.3, color=BLUE)
        underline = Underline(self.text, color=BLACK)
        animation_group = AnimationGroup(
            self.neural_network.make_forward_pass_animation(run_time=5),
            Create(underline),
            # Create(self.surrounding_rectangle)
        )
        # self.surrounding_rectangle = SurroundingRectangle(self.logo_group, buff=0.3, color=BLUE)
        underline = Underline(self.text, color=BLACK)
        animation_group = AnimationGroup(
            self.neural_network.make_forward_pass_animation(run_time=5),
            Create(underline),
            #    Create(self.surrounding_rectangle)
        )
        self.play(animation_group, runtime=5.0)
        self.wait(5)
