from manim import *

from manim_ml.neural_network.activation_functions import get_activation_function_by_name
from manim_ml.neural_network.activation_functions.activation_function import ActivationFunction
from manim_ml.neural_network.layers.parent_layers import VGroupNeuralNetworkLayer

class FeedForwardLayer(VGroupNeuralNetworkLayer):
    """Handles rendering a layer for a neural network"""

    def __init__(
        self,
        num_nodes,
        layer_buffer=SMALL_BUFF / 2,
        node_radius=0.08,
        node_color=BLUE,
        node_outline_color=WHITE,
        rectangle_color=WHITE,
        node_spacing=0.3,
        rectangle_fill_color=BLACK,
        node_stroke_width=2.0,
        rectangle_stroke_width=2.0,
        animation_dot_color=RED,
        activation_function=None,
        **kwargs
    ):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.layer_buffer = layer_buffer
        self.node_radius = node_radius
        self.node_color = node_color
        self.node_stroke_width = node_stroke_width
        self.node_outline_color = node_outline_color
        self.rectangle_stroke_width = rectangle_stroke_width
        self.rectangle_color = rectangle_color
        self.node_spacing = node_spacing
        self.rectangle_fill_color = rectangle_fill_color
        self.animation_dot_color = animation_dot_color
        self.activation_function = activation_function

        self.node_group = VGroup()

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        """Creates the neural network layer"""
        # Add Nodes
        for node_number in range(self.num_nodes):
            node_object = Circle(
                radius=self.node_radius,
                color=self.node_color,
                stroke_width=self.node_stroke_width,
            )
            self.node_group.add(node_object)
        # Space the nodes
        # Assumes Vertical orientation
        for node_index, node_object in enumerate(self.node_group):
            location = node_index * self.node_spacing
            node_object.move_to([0, location, 0])
        # Create Surrounding Rectangle
        self.surrounding_rectangle = SurroundingRectangle(
            self.node_group,
            color=self.rectangle_color,
            fill_color=self.rectangle_fill_color,
            fill_opacity=1.0,
            buff=self.layer_buffer,
            stroke_width=self.rectangle_stroke_width,
        )
        self.surrounding_rectangle.set_z_index(1)
        # Add the objects to the class
        self.add(self.surrounding_rectangle, self.node_group)

        self.construct_activation_function()

    def construct_activation_function(self):
        """Construct the activation function"""
        # Add the activation function
        if not self.activation_function is None:
            # Check if it is a string
            if isinstance(self.activation_function, str):
                activation_function = get_activation_function_by_name(
                    self.activation_function
                )()
            else:
                assert isinstance(self.activation_function, ActivationFunction)
                activation_function = self.activation_function
            # Plot the function above the rest of the layer
            self.activation_function = activation_function
            self.add(self.activation_function)

    def make_dropout_forward_pass_animation(self, layer_args, **kwargs):
        """Makes a forward pass animation with dropout"""
        # Make sure proper dropout information was passed
        assert "dropout_node_indices" in layer_args
        dropout_node_indices = layer_args["dropout_node_indices"]
        # Only highlight nodes that were note dropped out
        nodes_to_highlight = []
        for index, node in enumerate(self.node_group):
            if not index in dropout_node_indices:
                nodes_to_highlight.append(node)
        nodes_to_highlight = VGroup(*nodes_to_highlight)
        # Make highlight animation
        succession = Succession(
            ApplyMethod(
                nodes_to_highlight.set_color, self.animation_dot_color, run_time=0.25
            ),
            Wait(1.0),
            ApplyMethod(nodes_to_highlight.set_color, self.node_color, run_time=0.25),
        )

        return succession

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        # Check if dropout is a thing
        if "dropout_node_indices" in layer_args:
            # Drop out certain nodes
            return self.make_dropout_forward_pass_animation(
                layer_args=layer_args, **kwargs
            )
        else:
            # Make highlight animation
            succession = Succession(
                ApplyMethod(
                    self.node_group.set_color, self.animation_dot_color, run_time=0.25
                ),
                Wait(1.0),
                ApplyMethod(self.node_group.set_color, self.node_color, run_time=0.25),
            )
            if not self.activation_function is None:
                animation_group = AnimationGroup(
                    succession,
                    self.activation_function.make_evaluate_animation(),
                    lag_ratio=0.0,
                )
                return animation_group
            else:
                return succession

    @override_animation(Create)
    def _create_override(self, **kwargs):
        animations = []

        animations.append(Create(self.surrounding_rectangle))

        for node in self.node_group:
            animations.append(Create(node))

        animation_group = AnimationGroup(*animations, lag_ratio=0.0)
        return animation_group
