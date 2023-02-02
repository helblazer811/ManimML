from manim import *

from manim_ml.neural_network import FeedForwardLayer, NeuralNetwork

# Make the specific scene
config.pixel_height = 700
config.pixel_width = 1200
config.frame_height = 4.0
config.frame_width = 4.0


class CombinedScene(ThreeDScene):
    def construct(self):
        # Make nn
        nn = NeuralNetwork(
            [
                FeedForwardLayer(num_nodes=3),
                FeedForwardLayer(num_nodes=5),
                FeedForwardLayer(num_nodes=3),
            ]
        )
        self.add(nn)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
