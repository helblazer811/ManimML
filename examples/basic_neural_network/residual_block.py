from manim import *

from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.math_operation_layer import MathOperationLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0

def make_code_snippet():
    code_str = """
        nn = NeuralNetwork({
            "feed_forward_1": FeedForwardLayer(3),
            "feed_forward_2": FeedForwardLayer(3, activation_function="ReLU"),
            "feed_forward_3": FeedForwardLayer(3),
            "sum_operation": MathOperationLayer("+", activation_function="ReLU"),
        })
        nn.add_connection("feed_forward_1", "sum_operation")
        self.play(nn.make_forward_pass_animation()) 
    """

    code = Code(
        code=code_str,
        tab_width=4,
        background_stroke_width=1,
        background_stroke_color=WHITE,
        insert_line_no=False,
        style="monokai",
        # background="window",
        language="py",
    )
    code.scale(0.38)

    return code


class CombinedScene(ThreeDScene):
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
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Make code snippet
        code = make_code_snippet()
        code.next_to(nn, DOWN)
        self.add(code)
        # Group it all
        group = Group(nn, code)
        group.move_to(ORIGIN)
        # Play animation
        forward_pass = nn.make_forward_pass_animation()
        self.wait(1)
        self.play(forward_pass)