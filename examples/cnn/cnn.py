from pathlib import Path

from manim import *
from PIL import Image

<<<<<<< HEAD
from manim_ml.neural_network.layers.convolutional3d import Convolutional3DLayer
=======
from manim_ml.neural_network.layers import Convolutional3DLayer
>>>>>>> 0bc3ad561ba224f3d33e9f843665c1d50d64a68b
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

<<<<<<< HEAD
# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 7.0
config.frame_width = 7.0
=======
ROOT_DIR = Path(__file__).parents[2]
>>>>>>> 0bc3ad561ba224f3d33e9f843665c1d50d64a68b

def make_code_snippet():
    code_str = """
        # Make nn
        nn = NeuralNetwork([
<<<<<<< HEAD
            ImageLayer(numpy_image, height=1.5),
            Convolutional3DLayer(1, 7, 7, 3, 3),
            Convolutional3DLayer(3, 5, 5, 3, 3),
            Convolutional3DLayer(5, 3, 3, 1, 1),
=======
            ImageLayer(numpy_image),
            Convolutional3DLayer(3, 3, 3),
            Convolutional3DLayer(5, 2, 2),
            Convolutional3DLayer(10, 2, 1),
>>>>>>> 0bc3ad561ba224f3d33e9f843665c1d50d64a68b
            FeedForwardLayer(3),
            FeedForwardLayer(3),
        ])
        # Play animation
        self.play(nn.make_forward_pass_animation()) 
    """

    code = Code(
        code = code_str, 
        tab_width=4,
        background_stroke_width=1,
        background_stroke_color=WHITE,
        insert_line_no=False,
        style='monokai',
        #background="window",
        language="py",
    )
    code.scale(0.50)

    return code

class CombinedScene(ThreeDScene):
    def construct(self):
        image = Image.open(ROOT_DIR / 'assets/mnist/digit.jpeg')
        numpy_image = np.asarray(image)
        # Make nn
        nn = NeuralNetwork([
<<<<<<< HEAD
                ImageLayer(numpy_image, height=1.5),
                Convolutional3DLayer(1, 7, 7, 3, 3, filter_spacing=0.32),
                Convolutional3DLayer(3, 5, 5, 3, 3, filter_spacing=0.32),
                Convolutional3DLayer(5, 3, 3, 1, 1, filter_spacing=0.18),
                FeedForwardLayer(3),
                FeedForwardLayer(3),
            ], 
            layer_spacing=0.25,
        )
        # Center the nn
=======
            ImageLayer(numpy_image, height=3.5),
            Convolutional3DLayer(3, 3, 3, filter_spacing=0.2),
            Convolutional3DLayer(5, 2, 2, filter_spacing=0.2),
            Convolutional3DLayer(10, 2, 1, filter_spacing=0.2),
            FeedForwardLayer(3, rectangle_stroke_width=4, node_stroke_width=4).scale(2),
            FeedForwardLayer(1, rectangle_stroke_width=4, node_stroke_width=4).scale(2)
        ], layer_spacing=0.2)
        nn.scale(0.9)
>>>>>>> 0bc3ad561ba224f3d33e9f843665c1d50d64a68b
        nn.move_to(ORIGIN)
        self.add(nn)
        # Make code snippet
        code = make_code_snippet()
        code.next_to(nn, DOWN)
        self.add(code)
        # Group it all
        group = Group(nn, code)
        group.move_to(ORIGIN)
        # Play animation
        forward_pass = nn.make_forward_pass_animation(
            corner_pulses=False,
            all_filters_at_once=False
        )
        self.wait(1)
        self.play(
           forward_pass
        ) 
