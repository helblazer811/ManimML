from manim import *
from manim_ml.neural_network.embedding import EmbeddingLayer
from manim_ml.neural_network.feed_forward import FeedForwardLayer
from manim_ml.neural_network.image import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork, FeedForwardNeuralNetwork
from PIL import Image
import numpy as np

config.pixel_height = 720
config.pixel_width = 1280
config.frame_height = 6.0
config.frame_width = 6.0

class FeedForwardNeuralNetworkScene(Scene):

    def construct(self):
        nn = FeedForwardNeuralNetwork([3, 5, 3])
        self.play(Create(nn))
        self.play(Wait(3))

class NeuralNetworkScene(Scene):
    """Test Scene for the Neural Network"""

    def construct(self):
        # Make the Layer object
        layers = [
            FeedForwardLayer(3), 
            FeedForwardLayer(5), 
            FeedForwardLayer(3)
        ]
        nn = NeuralNetwork(layers)
        nn.move_to(ORIGIN)
        # Make Animation
        self.add(nn)
        forward_propagation_animation = nn.make_forward_pass_animation(run_time=5, passing_flash=True)

        self.play(forward_propagation_animation)

class ImageNeuralNetworkScene(Scene):

    def construct(self):
        image = Image.open('images/image.jpeg')
        numpy_image = np.asarray(image)
        # Make nn
        layers = [
            ImageLayer(numpy_image, height=1.4),
            FeedForwardLayer(3), 
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            FeedForwardLayer(6)
        ]
        nn = NeuralNetwork(layers)
        nn.scale(1.3)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        self.play(nn.make_forward_pass_animation(run_time=5))
        self.play(nn.make_forward_pass_animation(run_time=5))


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

class RecursiveNNScene(Scene):

    def construct(self):
        nn = NeuralNetwork([
            NeuralNetwork([
                FeedForwardLayer(3),
                FeedForwardLayer(2)
            ]),
            NeuralNetwork([
                FeedForwardLayer(2),
                FeedForwardLayer(3)
            ])
        ])

        self.play(Create(nn))

if __name__ == "__main__":
    """Render all scenes"""
    # Feed Forward Neural Network
    ffnn_scene = FeedForwardNeuralNetworkScene()
    ffnn_scene.render()
    # Neural Network 
    nn_scene = NeuralNetworkScene()
    nn_scene.render()
