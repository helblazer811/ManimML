from manim import *
from manim_ml.neural_network.layers import TripletLayer, triplet
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

config.pixel_height = 720
config.pixel_width = 1280
config.frame_height = 6.0
config.frame_width = 6.0


class TripletScene(Scene):
    def construct(self):
        anchor_path = "../assets/triplet/anchor.jpg"
        positive_path = "../assets/triplet/positive.jpg"
        negative_path = "../assets/triplet/negative.jpg"

        triplet_layer = TripletLayer.from_paths(
            anchor_path, positive_path, negative_path, grayscale=False
        )

        triplet_layer.scale(0.08)

        neural_network = NeuralNetwork(
            [triplet_layer, FeedForwardLayer(5), FeedForwardLayer(3)]
        )

        neural_network.scale(1)

        self.play(Create(neural_network), run_time=3)

        self.play(neural_network.make_forward_pass_animation(), run_time=10)
