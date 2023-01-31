from manim import *
from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork
from PIL import Image
import numpy as np

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0

class NeuralNetworkScene(ThreeDScene):
    """Test Scene for the Neural Network"""

    def play_camera_follow_forward_pass(
        self, 
        neural_network,
        buffer=0.1
    ):
        per_layer_animations = neural_network.make_forward_pass_animation(
            return_per_layer_animations=True
        )
        all_layers = neural_network.all_layers
        # Compute the width and height of the frame
        max_width = 0
        max_height = 0
        for layer_index in range(1, len(all_layers) - 1):
            prev_layer = all_layers[layer_index - 1]
            current_layer = all_layers[layer_index]
            next_layer = all_layers[layer_index + 1]
            group = Group(prev_layer, current_layer, next_layer)
            
            max_width = max(max_width, group.width)
            max_height = max(max_height, group.height)

        frame_width = max_width * (1 + buffer)
        frame_height = max_height * (1 + buffer)
        # Go through each animation
        for layer_index in range(1, len(all_layers)):
            layer_animation = per_layer_animations[layer_index]

    def construct(self):
        # Make the Layer object
        image = Image.open("../assets/mnist/digit.jpeg")
        numpy_image = np.asarray(image)
        nn = NeuralNetwork([
                ImageLayer(numpy_image, height=1.5),
                Convolutional2DLayer(1, 7, filter_spacing=0.32),
                Convolutional2DLayer(3, 5, 3, filter_spacing=0.32),
                Convolutional2DLayer(5, 3, 3, filter_spacing=0.18),
                FeedForwardLayer(3),
                FeedForwardLayer(3),
            ],
            layer_spacing=0.25,
        )
        nn.move_to(ORIGIN)
        # Make Animation
        self.add(nn)
        # self.play(Create(nn))
        self.play_camera_follow_forward_pass(nn)