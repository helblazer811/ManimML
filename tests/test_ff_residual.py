from manim import *
from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer, MathOperationLayer

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0

class TestFeedForwardResidualNetwork(Scene):

    def construct(self):
        # Add the network
        nn = NeuralNetwork({
                "feed_forward_1": FeedForwardLayer(3),
                "feed_forward_2": FeedForwardLayer(3, activation_function="ReLU"),
                "feed_forward_3": FeedForwardLayer(3),
                "sum_operation": MathOperationLayer("+", activation_function="ReLU"),
            },
            layer_spacing=0.38
        )
        self.add(nn)
        # Make connections
        input_blank_dot = Dot(
            nn.input_layers_dict["feed_forward_1"].get_left() - np.array([0.65, 0.0, 0.0])
        )
        nn.add_connection(input_blank_dot, "feed_forward_1", arc_direction="straight")
        nn.add_connection("feed_forward_1", "sum_operation")
        output_blank_dot = Dot(
            nn.input_layers_dict["sum_operation"].get_right() + np.array([0.65, 0.0, 0.0])
        )
        nn.add_connection("sum_operation", output_blank_dot, arc_direction="straight")

        # Make forward pass animation
        self.play(
            nn.make_forward_pass_animation()
        )
    