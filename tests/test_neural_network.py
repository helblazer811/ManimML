from manim import *
from manim_ml.neural_network.layers import FeedForwardLayer, ImageLayer
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
            ImageLayer(numpy_image, height=1.0),
            FeedForwardLayer(3), 
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            FeedForwardLayer(3)
        ]
        nn = NeuralNetwork(layers)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        self.play(nn.make_forward_pass_animation())

if __name__ == "__main__":
    """Render all scenes"""
    # Feed Forward Neural Network
    ffnn_scene = FeedForwardNeuralNetworkScene()
    ffnn_scene.render()
    # Neural Network 
    nn_scene = NeuralNetworkScene()
    nn_scene.render()
