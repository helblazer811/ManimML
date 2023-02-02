from manim import *
from PIL import Image
import numpy as np

from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.max_pooling_2d import MaxPooling2DLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0


class MaxPoolingScene(ThreeDScene):
    def construct(self):
        # Make nn
        nn = NeuralNetwork(
            [
                Convolutional2DLayer(1, 8),
                Convolutional2DLayer(3, 6, 3),
                MaxPooling2DLayer(kernel_size=2),
                Convolutional2DLayer(5, 2, 2),
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
