from manim import *
from PIL import Image

from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.layers.parent_layers import ThreeDLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 800
config.frame_height = 6.0
config.frame_width = 6.0


class CombinedScene(ThreeDScene):
    def construct(self):
        image = Image.open("../../assets/doggo.jpeg")
        numpy_image = np.asarray(image)
        # Rotate the Three D layer position
        ThreeDLayer.rotation_angle = 15 * DEGREES
        ThreeDLayer.rotation_axis = [1, -1.0, 0]
        # Make nn
        nn = NeuralNetwork(
            [
                ImageLayer(numpy_image, height=1.5),
                Convolutional2DLayer(1, 7, 7, 3, 3, filter_spacing=0.32),
                Convolutional2DLayer(3, 5, 5, 1, 1, filter_spacing=0.18),
            ],
            layer_spacing=0.25,
            layout_direction="top_to_bottom",
        )
        # Center the nn
        nn.move_to(ORIGIN)
        nn.scale(1.5)
        self.add(nn)
        # Play animation
        forward_pass = nn.make_forward_pass_animation(
            highlight_active_feature_map=True,
        )
        self.wait(1)
        self.play(forward_pass)
