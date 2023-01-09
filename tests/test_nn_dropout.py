from manim import *
from manim_ml.neural_network.animations.dropout import (
    make_neural_network_dropout_animation,
)
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from PIL import Image
from manim_ml.neural_network.neural_network import NeuralNetwork
import numpy as np

config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 5.0
config.frame_width = 5.0


def make_code_snippet():
    code_str = """ 
        nn = NeuralNetwork([
            FeedForwardLayer(3),
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            FeedForwardLayer(5),
            FeedForwardLayer(4),
        ])
        self.play(
            make_neural_network_dropout_animation(
                nn, dropout_rate=0.25, do_forward_pass=True
            )
        )
    """

    code = Code(
        code=code_str,
        tab_width=4,
        background_stroke_width=1,
        background_stroke_color=WHITE,
        insert_line_no=False,
        style="monokai",
        language="py",
    )
    code.scale(0.28)

    return code


class DropoutNeuralNetworkScene(Scene):
    def construct(self):
        # Make nn
        nn = NeuralNetwork(
            [
                FeedForwardLayer(3, rectangle_color=BLUE),
                FeedForwardLayer(5, rectangle_color=BLUE),
                FeedForwardLayer(3, rectangle_color=BLUE),
                FeedForwardLayer(5, rectangle_color=BLUE),
                FeedForwardLayer(4, rectangle_color=BLUE),
            ],
            layer_spacing=0.4,
        )
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Make code snippet
        code_snippet = make_code_snippet()
        self.add(code_snippet)
        code_snippet.next_to(nn, DOWN * 0.7)
        Group(code_snippet, nn).move_to(ORIGIN)
        # Play animation
        self.play(
            make_neural_network_dropout_animation(
                nn, dropout_rate=0.25, do_forward_pass=True
            )
        )
        self.wait(1)


if __name__ == "__main__":
    """Render all scenes"""
    dropout_nn_scene = DropoutNeuralNetworkScene()
    dropout_nn_scene.render()
