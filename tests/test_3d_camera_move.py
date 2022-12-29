from manim import *
from PIL import Image

from manim_ml.neural_network.layers.convolutional3d import Convolutional3DLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0

class CombinedScene(ThreeDScene):
    def construct(self):
        image = Image.open('../assets/mnist/digit.jpeg')
        numpy_image = np.asarray(image)
        # Make nn
        nn = NeuralNetwork(
            [
                ImageLayer(numpy_image, height=1.4),
                Convolutional3DLayer(1, 7, 7, 3, 3, filter_spacing=0.2),
                Convolutional3DLayer(3, 5, 5, 3, 3, filter_spacing=0.2),
                Convolutional3DLayer(5, 3, 3, 1, 1, filter_spacing=0.2),
                FeedForwardLayer(3, rectangle_stroke_width=4, node_stroke_width=4),
                FeedForwardLayer(3, rectangle_stroke_width=4, node_stroke_width=4),
            ], 
            layer_spacing=0.5,
            # camera=self.camera
        )
        # Center the nn
        self.add(nn)
        nn.move_to(ORIGIN)
        # Play animation
        forward_pass = nn.make_forward_pass_animation(
            corner_pulses=False,
            layer_args={
                "all_filters_at_once": True
            }
        )
        self.play(
            forward_pass
        ) 
