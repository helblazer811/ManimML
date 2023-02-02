from manim import *
from manim_ml.neural_network.layers import FeedForwardLayer
from manim_ml.neural_network.neural_network import NeuralNetwork


class NeuralNetworkScene(Scene):
    """Test Scene for the Neural Network"""

    def construct(self):
        # Make the Layer object
        layers = [FeedForwardLayer(3), FeedForwardLayer(5), FeedForwardLayer(3)]
        nn = NeuralNetwork(layers)
        nn.scale(2)
        nn.move_to(ORIGIN)
        # Make Animation
        self.add(nn)
        # self.play(Create(nn))
        forward_propagation_animation = nn.make_forward_pass_animation(
            run_time=5, passing_flash=True
        )

        self.play(forward_propagation_animation)
