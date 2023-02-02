from pathlib import Path

from manim import *
from PIL import Image
import numpy as np

from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.layers.max_pooling_2d import MaxPooling2DLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 20.0
config.frame_width = 20.0

ROOT_DIR = Path(__file__).parents[2]


class CombinedScene(ThreeDScene):
    def construct(self):
        image = Image.open(ROOT_DIR / "assets/mnist/digit.jpeg")
        numpy_image = np.asarray(image)
        # Make nn
        nn = NeuralNetwork(
            [
                ImageLayer(numpy_image, height=4.5),
                Convolutional2DLayer(1, 28),
                Convolutional2DLayer(6, 28, 5),
                MaxPooling2DLayer(kernel_size=2),
                Convolutional2DLayer(16, 10, 5),
                MaxPooling2DLayer(kernel_size=2),
                FeedForwardLayer(8),
                FeedForwardLayer(3),
                FeedForwardLayer(2),
            ],
            layer_spacing=0.25,
        )
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Make code snippet
        # code = make_code_snippet()
        # code.next_to(nn, DOWN)
        # self.add(code)
        # Group it all
        # group = Group(nn, code)
        # group.move_to(ORIGIN)
        nn.move_to(ORIGIN)
        # Play animation
        # forward_pass = nn.make_forward_pass_animation()
        # self.wait(1)
        # self.play(forward_pass)
