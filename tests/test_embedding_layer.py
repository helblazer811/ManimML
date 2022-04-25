from manim import *

from manim_ml.neural_network.layers.embedding import EmbeddingLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

config.pixel_height = 720 
config.pixel_width = 1280 
config.frame_height = 5.0
config.frame_width = 5.0

class EmbeddingNNScene(Scene):

    def construct(self):
        embedding_layer = EmbeddingLayer()

        neural_network = NeuralNetwork([
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            embedding_layer,
            FeedForwardLayer(3),
            FeedForwardLayer(5)
        ])

        self.play(Create(neural_network))

        self.play(neural_network.make_forward_pass_animation(run_time=5))

class QueryEmbeddingNNScene(Scene):

    def construct(self):
        embedding_layer = EmbeddingLayer()
        embedding_layer.paired_query_mode = True

        neural_network = NeuralNetwork([
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            embedding_layer,
            FeedForwardLayer(3),
            FeedForwardLayer(5)
        ])

        self.play(Create(neural_network), run_time=2)

        self.play(
            neural_network.make_forward_pass_animation(
                run_time=5,
                layer_args={
                    embedding_layer: {
                        "query_locations": (np.array([2, 2]), np.array([1, 1]))
                    }
                } 
            )
        )