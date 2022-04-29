from manim import *
from PIL import Image

from manim_ml.neural_network.layers.convolutional import ConvolutionalLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

def make_code_snippet():
    code_str = """
        # Make nn
        nn = NeuralNetwork([
            ImageLayer(numpy_image),
            ConvolutionalLayer(3, 3, 3),
            ConvolutionalLayer(5, 2, 2),
            ConvolutionalLayer(10, 2, 1),
            FeedForwardLayer(3),
            FeedForwardLayer(1)
        ], layer_spacing=0.2)
        # Center the nn
        self.play(Create(nn))
        # Play animation
        self.play(nn.make_forward_pass_animation(run_time=5)) 
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
    code.scale(0.6)

    return code


# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 12.0
config.frame_width = 12.0

class CombinedScene(ThreeDScene, Scene):
    def construct(self):
        image = Image.open('../../assets/mnist/digit.jpeg')
        numpy_image = np.asarray(image)
        # Make nn
        nn = NeuralNetwork([
            ImageLayer(numpy_image, height=3.5),
            ConvolutionalLayer(3, 3, 3, filter_spacing=0.2),
            ConvolutionalLayer(5, 2, 2, filter_spacing=0.2),
            ConvolutionalLayer(10, 2, 1, filter_spacing=0.2),
            FeedForwardLayer(3, rectangle_stroke_width=4, node_stroke_width=4).scale(2),
            FeedForwardLayer(1, rectangle_stroke_width=4, node_stroke_width=4).scale(2)
        ], layer_spacing=0.2)
        nn.scale(0.9)
        nn.move_to(ORIGIN)
        nn.shift(UP*1.8)
        # Make code snippet
        code = make_code_snippet()
        code.shift(DOWN*1.8)
        # Center the nn
        self.play(Create(nn))
        self.add(code)
        # Play animation
        # self.set_camera_orientation(phi=280* DEGREES, theta=-20*DEGREES, gamma=90 * DEGREES)
        # self.begin_ambient_camera_rotation()
        self.play(nn.make_forward_pass_animation(run_time=5)) 
