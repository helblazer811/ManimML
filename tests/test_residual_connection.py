from manim import *
from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.utils.testing.frames_comparison import frames_comparison

from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer, ImageLayer

from PIL import Image
import numpy as np

__module_test__ = "residual"


@frames_comparison
def test_ResidualConnectionScene(scene):
    """Tests the appearance of a residual connection"""
    nn = NeuralNetwork(
        {
            "layer1": FeedForwardLayer(3),
            "layer2": FeedForwardLayer(5),
            "layer3": FeedForwardLayer(3),
        }
    )

    scene.add(nn)


# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0


class FeedForwardScene(Scene):
    def construct(self):
        nn = NeuralNetwork(
            {
                "layer1": FeedForwardLayer(4),
                "layer2": FeedForwardLayer(4),
                "layer3": FeedForwardLayer(4),
            },
            layer_spacing=0.45,
        )

        nn.add_connection("layer1", "layer3")

        self.add(nn)

        self.play(nn.make_forward_pass_animation(), run_time=8)


class ConvScene(ThreeDScene):
    def construct(self):
        image = Image.open("../assets/mnist/digit.jpeg")
        numpy_image = np.asarray(image)

        nn = NeuralNetwork(
            {
                "layer1": Convolutional2DLayer(1, 5, padding=1),
                "layer2": Convolutional2DLayer(1, 5, 3, padding=1),
                "layer3": Convolutional2DLayer(1, 5, 3, padding=1),
            },
            layer_spacing=0.25,
        )

        nn.add_connection("layer1", "layer3")

        self.add(nn)

        self.play(nn.make_forward_pass_animation(), run_time=8)
