from manim import *
from manim_ml.neural_network.layers import FeedForwardLayer, ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork
from PIL import Image
import numpy as np

config.pixel_height = 720
config.pixel_width = 1280
config.frame_height = 6.0
config.frame_width = 6.0


class ImageNeuralNetworkScene(Scene):
    def make_code_snippet(self):
        code_str = """
            # Make image object
            image = Image.open('images/image.jpeg')
            numpy_image = np.asarray(image)
            # Make Neural Network
            layers = [
                ImageLayer(numpy_image, height=1.4),
                FeedForwardLayer(3), 
                FeedForwardLayer(5),
                FeedForwardLayer(3)
            ]
            nn = NeuralNetwork(layers)
            self.add(nn)
            # Play animation
            self.play(
                nn.make_forward_pass_animation()
            )
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
        code.scale(0.2)

        return code

    def construct(self):
        image = Image.open("../../tests/images/image.jpeg")
        numpy_image = np.asarray(image)
        # Make nn
        layers = [
            ImageLayer(numpy_image, height=1.4),
            FeedForwardLayer(3),
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            FeedForwardLayer(6),
        ]
        nn = NeuralNetwork(layers)
        nn.scale(0.9)
        # Center the nn
        nn.move_to(ORIGIN)
        nn.rotate(-PI / 2)
        nn.layers[0].image_mobject.rotate(PI / 2)
        nn.layers[0].image_mobject.shift([0, -0.4, 0])
        nn.shift([1.5, 0.3, 0])
        self.add(nn)
        # Make code snippet
        code_snippet = self.make_code_snippet()
        code_snippet.scale(1.9)
        code_snippet.shift([-1.25, 0, 0])
        self.add(code_snippet)
        # Play animation
        self.play(nn.make_forward_pass_animation(run_time=10))


if __name__ == "__main__":
    """Render all scenes"""
    # Feed Forward Neural Network
    ffnn_scene = FeedForwardNeuralNetworkScene()
    ffnn_scene.render()
    # Neural Network
    nn_scene = NeuralNetworkScene()
    nn_scene.render()
