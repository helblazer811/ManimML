from manim import *
from PIL import Image

from manim_ml.neural_network.layers.convolutional_3d import Convolutional3DLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

class SingleConvolutionalLayerScene(ThreeDScene):

    def construct(self):
        # Make nn
        layers = [
            Convolutional3DLayer(3, 4)
        ]
        nn = NeuralNetwork(layers)
        nn.scale(1.3)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        self.set_camera_orientation(phi=280*DEGREES, theta=-10*DEGREES, gamma=90*DEGREES)
        # self.play(nn.make_forward_pass_animation(run_time=5))

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 12.0
config.frame_width = 12.0

class CombinedScene(ThreeDScene, Scene):
    def construct(self):
        image = Image.open('../assets/mnist/digit.jpeg')
        numpy_image = np.asarray(image)
        # Make nn
        nn = NeuralNetwork([
            ImageLayer(numpy_image, height=1.4),
            Convolutional3DLayer(3, 3, 3, filter_spacing=0.2),
            Convolutional3DLayer(5, 2, 2, filter_spacing=0.2),
            Convolutional3DLayer(10, 2, 1, filter_spacing=0.2),
            FeedForwardLayer(3, rectangle_stroke_width=4, node_stroke_width=4).scale(2),
            FeedForwardLayer(1, rectangle_stroke_width=4, node_stroke_width=4).scale(2)
        ], layer_spacing=0.2)

        nn.scale(1.3)
        # Center the nn
        nn.move_to(ORIGIN)
        self.play(Create(nn))
        # Play animation
        # self.set_camera_orientation(phi=280* DEGREES, theta=-20*DEGREES, gamma=90 * DEGREES)
        # self.begin_ambient_camera_rotation()
        forward_pass = nn.make_forward_pass_animation(run_time=10)
        print(forward_pass)
        self.play(
            forward_pass
        ) 
