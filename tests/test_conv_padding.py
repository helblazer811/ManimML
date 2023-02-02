from manim import *

from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

from manim_ml.utils.testing.frames_comparison import frames_comparison

__module_test__ = "padding"

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0


class CombinedScene(ThreeDScene):
    def construct(self):
        # Make nn
        nn = NeuralNetwork(
            [
                Convolutional2DLayer(
                    num_feature_maps=1,
                    feature_map_size=7,
                    padding=1,
                    padding_dashed=True,
                ),
                Convolutional2DLayer(
                    num_feature_maps=3,
                    feature_map_size=7,
                    filter_size=3,
                    padding=0,
                    padding_dashed=False,
                ),
                FeedForwardLayer(3),
            ],
            layer_spacing=0.25,
        )
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        forward_pass = nn.make_forward_pass_animation()
        self.wait(1)
        self.play(forward_pass, run_time=30)


@frames_comparison
def test_ConvPadding(scene):
    # Make nn
    nn = NeuralNetwork(
        [
            Convolutional2DLayer(
                num_feature_maps=1, feature_map_size=7, padding=1, padding_dashed=True
            ),
            Convolutional2DLayer(
                num_feature_maps=3,
                feature_map_size=7,
                filter_size=3,
                padding=1,
                filter_spacing=0.35,
                padding_dashed=False,
            ),
            FeedForwardLayer(3),
        ],
        layer_spacing=0.25,
    )
    # Center the nn
    nn.move_to(ORIGIN)
    scene.add(nn)
    # Play animation
    forward_pass = nn.make_forward_pass_animation()
    scene.play(forward_pass, run_time=30)
