from manim import *

from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer

class FeedForwardScene(Scene):

    def construct(self):
        nn = NeuralNetwork([
            FeedForwardLayer(3),
            FeedForwardLayer(5),
            FeedForwardLayer(3)
        ])

        self.add(nn)