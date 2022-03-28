"""
    Logo for Manim Machine Learning
"""
from manim import *
from neural_network import NeuralNetwork

config.pixel_height = 500 
config.pixel_width = 1920 
config.frame_height = 10.0
config.frame_width = 10.0

class ManimMLLogo(Scene):

    def construct(self):
        self.text = Text("ManimML")
        self.text.scale(1.3)
        self.neural_network = NeuralNetwork([3, 5, 3, 6, 3], layer_spacing=0.6, node_color=BLUE)
        self.neural_network.scale(0.8)
        self.neural_network.move_to(self.text.get_right())
        self.neural_network.shift(RIGHT * 1.3)
        self.logo_group = VGroup(self.text, self.neural_network) 
        self.logo_group.scale(1.5)
        self.logo_group.move_to(ORIGIN)
        self.play(Write(self.text))
        self.play(Create(self.neural_network))
        # self.surrounding_rectangle = SurroundingRectangle(self.logo_group, buff=0.3, color=BLUE)
        underline = Underline(self.text, color=BLUE)
        animation_group = AnimationGroup(
            self.neural_network.make_forward_propagation_animation(run_time=5),
            Create(underline),
        #    Create(self.surrounding_rectangle)
        )
        # self.surrounding_rectangle = SurroundingRectangle(self.logo_group, buff=0.3, color=BLUE)
        underline = Underline(self.text, color=BLUE)
        animation_group = AnimationGroup(
            self.neural_network.make_forward_propagation_animation(run_time=5),
            Create(underline),
        #    Create(self.surrounding_rectangle)
        )
        self.play(animation_group)
        self.wait(5)
