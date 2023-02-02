from pathlib import Path

from manim import *
from PIL import Image

from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 7.0
config.frame_width = 7.0
ROOT_DIR = Path(__file__).parents[2]


def make_code_snippet():
    code_str = """
        # Make the neural network
        nn = NeuralNetwork([
            # ... Layers at start
            Convolutional2DLayer(3, 5, 3, activation_function="ReLU"),
            # ... Layers at end
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
    code.scale(0.45)

    return code


class CombinedScene(ThreeDScene):
    def construct(self):
        image = Image.open(ROOT_DIR / "assets/mnist/digit.jpeg")
        numpy_image = np.asarray(image)
        # Make nn
        nn = NeuralNetwork(
            [
                ImageLayer(numpy_image, height=1.5),
                Convolutional2DLayer(1, 7),
                Convolutional2DLayer(3, 5, 3, activation_function="ReLU"),
                Convolutional2DLayer(5, 3, 3, activation_function="ReLU"),
                FeedForwardLayer(3),
                FeedForwardLayer(1),
            ],
            layer_spacing=0.25,
        )
        # nn.scale(0.7)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Make code snippet
        # code = make_code_snippet()
        # code.next_to(nn, DOWN)
        # self.add(code)
        # nn.move_to(ORIGIN)
        # Move everything up
        # Group(nn, code).move_to(ORIGIN)
        # Play animation
        forward_pass = nn.make_forward_pass_animation()
        self.wait(1)
        self.play(forward_pass)
