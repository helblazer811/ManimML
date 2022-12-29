from manim import *
from manim_ml.neural_network.layers import Convolutional2DLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 12.0
config.frame_width = 12.0

class TestConv2d(Scene):

    def construct(self):
        nn = NeuralNetwork(
            [
                Convolutional2DLayer(5, 5, 3, 3, cell_width=0.5, stride=1),
                Convolutional2DLayer(3, 3, 2, 2, cell_width=0.5, stride=1),
            ], 
            layer_spacing=1.5,
            camera=self.camera
        )
        # Center the nn
        nn.scale(1.3)
        nn.move_to(ORIGIN)
        self.play(Create(nn), run_time=2)
        # Play animation
        forward_pass = nn.make_forward_pass_animation(run_time=19)
        self.play(
            forward_pass,
        )
        self.wait(1)