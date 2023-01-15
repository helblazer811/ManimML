from manim import *
from PIL import Image

from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork


class SingleConvolutionalLayerScene(ThreeDScene):
    def construct(self):
        # Make nn
        layers = [Convolutional2DLayer(3, 4)]
        nn = NeuralNetwork(layers)
        nn.scale(1.3)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        self.set_camera_orientation(
            phi=280 * DEGREES, theta=-10 * DEGREES, gamma=90 * DEGREES
        )
        # self.play(nn.make_forward_pass_animation(run_time=5))


class Simple3DConvScene(ThreeDScene):
    def construct(self):
        """
        TODO
        - [X] Make grid lines for the CNN filters
        - [ ] Make Scanning filter effect
        - [ ] Have filter box go accross each input feature map
        - [ ] Make filter lines effect
        - [ ] Make flowing animation down filter lines
        """
        # Make nn
        layers = [
            Convolutional2DLayer(
                1, 5, 5, 5, 5, feature_map_height=3, filter_width=3, filter_height=3
            ),
            Convolutional2DLayer(
                1, 3, 3, 1, 1, feature_map_width=3, filter_width=3, filter_height=3
            ),
        ]
        nn = NeuralNetwork(layers)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        # self.set_camera_orientation(phi=280*DEGREES, theta=-10*DEGREES, gamma=90*DEGREES)
        self.play(nn.make_forward_pass_animation(run_time=30))

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0

class CombinedScene(ThreeDScene):
    def construct(self):
        image = Image.open("../assets/mnist/digit.jpeg")
        numpy_image = np.asarray(image)
        # Make nn
        nn = NeuralNetwork(
            [
                ImageLayer(numpy_image, height=1.5),
                Convolutional2DLayer(1, 7, 7, 3, 3, filter_spacing=0.32),
                Convolutional2DLayer(3, 5, 5, 3, 3, filter_spacing=0.32),
                Convolutional2DLayer(5, 3, 3, 1, 1, filter_spacing=0.18),
                FeedForwardLayer(3),
                FeedForwardLayer(3),
            ],
            layer_spacing=0.25,
        )
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        forward_pass = nn.make_forward_pass_animation(
            corner_pulses=False, 
            all_filters_at_once=False
        )
        self.wait(1)
        self.play(forward_pass)