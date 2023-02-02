from manim import ORIGIN, ThreeDScene
from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer, MaxPooling2DLayer, Convolutional2DLayer
from manim_ml.neural_network.animations.dropout import make_neural_network_dropout_animation
# Make nn

class BasicScene(ThreeDScene):
    def construct(self):
        nn = NeuralNetwork([
                FeedForwardLayer(8),
                FeedForwardLayer(12),
                FeedForwardLayer(8),
                FeedForwardLayer(5),
            ],
            layer_spacing=0.6,
        )
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        self.play(
            make_neural_network_dropout_animation(
                nn, dropout_rate=0.75, do_forward_pass=True, last_layer_stable=True
            )
        )
        self.wait(1)