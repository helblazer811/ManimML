from manim import *

from manim_ml.neural_network import (
    Convolutional2DLayer,
    FeedForwardLayer,
    NeuralNetwork,
)

# Make the specific scene
config.pixel_height = 700
config.pixel_width = 1900
config.frame_height = 7.0
config.frame_width = 7.0


class CombinedScene(ThreeDScene):
    def construct(self):
        # Make nn
        nn = NeuralNetwork(
            [
                Convolutional2DLayer(1, 7, 3, filter_spacing=0.32),
                Convolutional2DLayer(3, 5, 3, filter_spacing=0.32),
                Convolutional2DLayer(5, 3, 3, filter_spacing=0.18),
                FeedForwardLayer(3),
                FeedForwardLayer(3),
            ],
            layer_spacing=0.25,
        )
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        forward_pass = nn.make_forward_pass_animation()
        self.play(ChangeSpeed(forward_pass, speedinfo={}), run_time=10)
