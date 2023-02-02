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
config.frame_height = 6.0
config.frame_width = 6.0


def make_code_snippet():
    code_str = """
        # Make the neural network
        nn = NeuralNetwork([
            ImageLayer(image),
            Convolutional2DLayer(1, 8),
            MaxPooling2DLayer(kernel_size=2),
            Convolutional2DLayer(3, 2, 3),
        ])
        # Play the animation
        self.play(nn.make_forward_pass_animation()) 
    """

    code = Code(
        code=code_str,
        tab_width=4,
        background_stroke_width=1,
        background_stroke_color=WHITE,
        insert_line_no=False,
        style="monokai",
        font="Monospace",
        background="window",
        language="py",
    )
    code.scale(0.4)

    return code


class CombinedScene(ThreeDScene):
    def construct(self):
        image = Image.open("../../assets/mnist/digit.jpeg")
        numpy_image = np.asarray(image)
        # Make nn
        nn = NeuralNetwork(
            [
                ImageLayer(numpy_image, height=1.5),
                Convolutional2DLayer(1, 8, filter_spacing=0.32),
                MaxPooling2DLayer(kernel_size=2),
                Convolutional2DLayer(3, 2, 3, filter_spacing=0.32),
            ],
            layer_spacing=0.25,
        )
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Make code snippet
        code = make_code_snippet()
        code.next_to(nn, DOWN)
        Group(code, nn).move_to(ORIGIN)
        self.add(code)
        self.wait(5)
        # Play animation
        forward_pass = nn.make_forward_pass_animation()
        self.wait(1)
        self.play(forward_pass)
