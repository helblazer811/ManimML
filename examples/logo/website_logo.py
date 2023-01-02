"""
    Logo for Manim Machine Learning
"""
from manim import *
from manim_ml.neural_network.neural_network import FeedForwardNeuralNetwork

config.pixel_height = 400
config.pixel_width = 600
config.frame_height = 8.0
config.frame_width = 10.0


class ManimMLLogo(Scene):
    def construct(self):
        self.neural_network = FeedForwardNeuralNetwork(
            [3, 5, 3, 5], layer_spacing=0.6, node_color=BLUE, edge_width=6
        )
        self.neural_network.scale(3)
        self.neural_network.move_to(ORIGIN)
        self.play(Create(self.neural_network))
        # self.surrounding_rectangle = SurroundingRectangle(self.logo_group, buff=0.3, color=BLUE)
        animation_group = AnimationGroup(
            self.neural_network.make_forward_pass_animation(run_time=5),
        )
        self.play(animation_group)
        self.wait(5)
