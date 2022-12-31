"""
    Logo for Manim Machine Learning
"""
from manim import *
from manim_ml.neural_network.neural_network import FeedForwardNeuralNetwork

config.pixel_height = 500
config.pixel_width = 500
config.frame_height = 4.0
config.frame_width = 4.0

class ManimMLLogo(Scene):

    def construct(self):
        self.text = Text("ManimML")
        self.text.scale(1.0)
        self.neural_network = FeedForwardNeuralNetwork([3, 5, 3, 6, 3], layer_spacing=0.3, node_color=BLUE)
        self.neural_network.scale(1.0)
        self.neural_network.move_to(self.text.get_bottom())
        self.neural_network.shift(1.25 * DOWN)
        self.logo_group = Group(self.text, self.neural_network) 
        self.logo_group.scale(1.0)
        self.logo_group.move_to(ORIGIN)
        self.play(Write(self.text))
        self.play(Create(self.neural_network))
        # self.surrounding_rectangle = SurroundingRectangle(self.logo_group, buff=0.3, color=BLUE)
        underline = Underline(self.text, color=BLUE)
        animation_group = AnimationGroup(
            self.neural_network.make_forward_pass_animation(run_time=5),
            Create(underline),
        #    Create(self.surrounding_rectangle)
        )
        # self.surrounding_rectangle = SurroundingRectangle(self.logo_group, buff=0.3, color=BLUE)
        underline = Underline(self.text, color=BLUE)
        animation_group = AnimationGroup(
            self.neural_network.make_forward_pass_animation(run_time=5),
            Create(underline),
        #    Create(self.surrounding_rectangle)
        )
        self.play(animation_group)
        self.wait(5)
