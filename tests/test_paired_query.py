from manim import *
from manim_ml.neural_network.layers import PairedQueryLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

config.pixel_height = 720
config.pixel_width = 1280
config.frame_height = 6.0
config.frame_width = 6.0


class PairedQueryScene(Scene):
    def construct(self):
        positive_path = "../assets/triplet/positive.jpg"
        negative_path = "../assets/triplet/negative.jpg"

        paired_layer = PairedQueryLayer.from_paths(
            positive_path, negative_path, grayscale=False
        )

        paired_layer.scale(0.08)

        neural_network = NeuralNetwork(
            [paired_layer, FeedForwardLayer(5), FeedForwardLayer(3)]
        )

        neural_network.scale(1)

        self.play(Create(neural_network), run_time=3)

        self.play(neural_network.make_forward_pass_animation(), run_time=10)
