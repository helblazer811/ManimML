from manim import *

from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0


class CombinedScene(ThreeDScene):
    def construct(self):
        # Make nn
        nn = NeuralNetwork(
            [
                Convolutional2DLayer(1, 7, filter_spacing=0.32),
                Convolutional2DLayer(
                    3, 5, 3, filter_spacing=0.32, activation_function="ReLU"
                ),
                FeedForwardLayer(3, activation_function="Sigmoid"),
            ],
            layer_spacing=0.25,
        )
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        forward_pass = nn.make_forward_pass_animation()
        self.play(ChangeSpeed(forward_pass, speedinfo={}), run_time=10)
        self.wait(1)
