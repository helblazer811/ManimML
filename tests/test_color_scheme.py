from manim import *

from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer
import manim_ml

manim_ml.config.color_scheme = "light_mode"

config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0

class FeedForwardScene(Scene):
    def construct(self):
        nn = NeuralNetwork([
            FeedForwardLayer(3), 
            FeedForwardLayer(5), 
            FeedForwardLayer(3)
        ])

        self.add(nn)
        self.play(nn.make_forward_pass_animation())
