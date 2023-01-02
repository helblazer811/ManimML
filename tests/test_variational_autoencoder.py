from manim import *
from PIL import Image
from manim_ml.neural_network.layers import EmbeddingLayer, FeedForwardLayer, ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

config.pixel_height = 720
config.pixel_width = 1280
config.frame_height = 8.0
config.frame_width = 8.0


class VariationalAutoencoderScene(Scene):
    def construct(self):
        embedding_layer = EmbeddingLayer(dist_theme="ellipse").scale(2)

        image = Image.open("images/image.jpeg")
        numpy_image = np.asarray(image)
        # Make nn
        neural_network = NeuralNetwork(
            [
                ImageLayer(numpy_image, height=1.4),
                FeedForwardLayer(5),
                FeedForwardLayer(3),
                embedding_layer,
                FeedForwardLayer(3),
                FeedForwardLayer(5),
                ImageLayer(numpy_image, height=1.4),
            ],
            layer_spacing=0.1,
        )

        neural_network.scale(1.3)

        self.play(Create(neural_network), run_time=3)
        self.play(neural_network.make_forward_pass_animation(), run_time=5)
        self.play(neural_network.make_forward_pass_animation(), run_time=5)
