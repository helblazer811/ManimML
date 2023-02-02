from manim import *
from manim_ml.neural_network.animations.dropout import (
    make_neural_network_dropout_animation,
)
from manim_ml.neural_network import FeedForwardLayer, NeuralNetwork

config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 5.0
config.frame_width = 5.0


class DropoutNeuralNetworkScene(Scene):
    def construct(self):
        # Make nn
        nn = NeuralNetwork(
            [
                FeedForwardLayer(3, rectangle_color=BLUE),
                FeedForwardLayer(5, rectangle_color=BLUE),
                FeedForwardLayer(3, rectangle_color=BLUE),
                FeedForwardLayer(5, rectangle_color=BLUE),
                FeedForwardLayer(4, rectangle_color=BLUE),
            ],
            layer_spacing=0.4,
        )
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        self.play(
            make_neural_network_dropout_animation(
                nn, dropout_rate=0.25, do_forward_pass=True
            )
        )
        self.wait(1)
