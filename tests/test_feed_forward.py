from manim import *
from manim_ml.utils.testing.frames_comparison import frames_comparison

from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer

__module_test__ = "feed_forward"


@frames_comparison
def test_FeedForwardScene(scene):
    """Tests the appearance of a feed forward network"""
    nn = NeuralNetwork([FeedForwardLayer(3), FeedForwardLayer(5), FeedForwardLayer(3)])

    scene.add(nn)


class FeedForwardScene(Scene):
    def construct(self):
        nn = NeuralNetwork(
            [FeedForwardLayer(3), FeedForwardLayer(5), FeedForwardLayer(3)]
        )

        self.add(nn)

        self.play(nn.make_forward_pass_animation())
