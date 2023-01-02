from manim import *
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from PIL import Image
from manim_ml.neural_network.neural_network import NeuralNetwork
import numpy as np

config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 4.0
config.frame_width = 4.0


class DropoutNeuralNetworkScene(Scene):
    def construct(self):
        image = Image.open("../assets/gan/real_image.jpg")
        numpy_image = np.asarray(image)
        # Make nn
        layers = [
            FeedForwardLayer(3, rectangle_color=BLUE),
            FeedForwardLayer(5, rectangle_color=BLUE),
            FeedForwardLayer(3, rectangle_color=BLUE),
            FeedForwardLayer(6, rectangle_color=BLUE),
        ]
        nn = NeuralNetwork(layers)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        for i in range(5):
            self.play(
                nn.make_forward_pass_animation(run_time=5, feed_forward_dropout=True)
            )


if __name__ == "__main__":
    """Render all scenes"""
    dropout_nn_scene = DropoutNeuralNetworkScene()
    dropout_nn_scene.render()
