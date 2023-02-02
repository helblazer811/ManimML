from manim import *
from PIL import Image
import numpy as np
from manim_ml.neural_network import Convolutional2DLayer, NeuralNetwork

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0


def make_code_snippet():
    code_str = """
        # Make the neural network
        nn = NeuralNetwork({
            "layer1": Convolutional2DLayer(1, 5, padding=1),
            "layer2": Convolutional2DLayer(1, 5, 3, padding=1),
            "layer3": Convolutional2DLayer(1, 5, 3, padding=1)
        })
        # Add the residual connection
        nn.add_connection("layer1", "layer3")
        # Make the animation
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


class ConvScene(ThreeDScene):
    def construct(self):
        image = Image.open("../../assets/mnist/digit.jpeg")
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

        code = make_code_snippet()
        code.next_to(nn, DOWN)
        self.add(code)
        Group(code, nn).move_to(ORIGIN)

        self.play(nn.make_forward_pass_animation(), run_time=8)
