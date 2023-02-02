from manim import *
from manim_ml.neural_network.layers.image import ImageLayer
import numpy as np
from PIL import Image

from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

ROOT_DIR = Path(__file__).parents[2]

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0


def make_code_snippet():
    code_str = """
        # Make nn
        nn = NeuralNetwork([
            ImageLayer(numpy_image),
            Convolutional2DLayer(1, 6, 1, padding=1),
            Convolutional2DLayer(3, 6, 3),
            FeedForwardLayer(3),
            FeedForwardLayer(1),
        ])
        # Play animation
        self.play(nn.make_forward_pass_animation()) 
    """

    code = Code(
        code=code_str,
        tab_width=4,
        background_stroke_width=1,
        background_stroke_color=WHITE,
        insert_line_no=False,
        style="monokai",
        # background="window",
        language="py",
    )
    code.scale(0.38)

    return code


class CombinedScene(ThreeDScene):
    def construct(self):
        # Make nn
        image = Image.open(ROOT_DIR / "assets/mnist/digit.jpeg")
        numpy_image = np.asarray(image)
        # Make nn
        nn = NeuralNetwork(
            [
                ImageLayer(numpy_image, height=1.5),
                Convolutional2DLayer(
                    num_feature_maps=1,
                    feature_map_size=6,
                    padding=1,
                    padding_dashed=True,
                ),
                Convolutional2DLayer(
                    num_feature_maps=3,
                    feature_map_size=6,
                    filter_size=3,
                    padding=0,
                    padding_dashed=False,
                ),
                FeedForwardLayer(3),
                FeedForwardLayer(1),
            ],
            layer_spacing=0.25,
        )
        # Center the nn
        self.add(nn)
        code = make_code_snippet()
        code.next_to(nn, DOWN)
        self.add(code)
        Group(code, nn).move_to(ORIGIN)
        # Play animation
        forward_pass = nn.make_forward_pass_animation()
        self.wait(1)
        self.play(forward_pass, run_time=20)
