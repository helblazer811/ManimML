from manim import *

from manim_ml.neural_network.activation_functions import get_activation_function_by_name
from manim_ml.neural_network.activation_functions.activation_function import (
    ActivationFunction,
)
from manim_ml.neural_network.layers.parent_layers import VGroupNeuralNetworkLayer

class MathOperationLayer(VGroupNeuralNetworkLayer):
    """Handles rendering a layer for a neural network"""
    valid_operations = ["+", "-", "*", "/"]

    def __init__(
        self,
        operation_type: str,
        node_radius=0.5,
        node_color=BLUE,
        node_stroke_width=2.0,
        active_color=ORANGE,
        activation_function=None,
        font_size=20,
        **kwargs
    ):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        # Ensure operation type is valid
        assert operation_type in MathOperationLayer.valid_operations
        self.operation_type = operation_type
        self.node_radius = node_radius
        self.node_color = node_color
        self.node_stroke_width = node_stroke_width
        self.active_color = active_color
        self.font_size = font_size
        self.activation_function = activation_function

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        """Creates the neural network layer"""
        # Draw the operation
        self.operation_text = Text(
            self.operation_type,
            font_size=self.font_size
        )
        self.add(self.operation_text)
        # Make the surrounding circle
        self.surrounding_circle = Circle(
            color=self.node_color,
            stroke_width=self.node_stroke_width
        ).surround(self.operation_text)
        self.add(self.surrounding_circle)
        # Make the activation function
        self.construct_activation_function()
        super().construct_layer(input_layer, output_layer, **kwargs)

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

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        """Makes the forward pass animation

        Parameters
        ----------
        layer_args : dict, optional
            layer specific arguments, by default {}

        Returns
        -------
        AnimationGroup
            Forward pass animation
        """
        # Make highlight animation
        succession = Succession(
            ApplyMethod(
                self.surrounding_circle.set_color, 
                self.active_color, 
                run_time=0.25
            ),
            Wait(1.0),
            ApplyMethod(
                self.surrounding_circle.set_color, 
                self.node_color, 
                run_time=0.25
            ),
        )
        # Animate the activation function
        if not self.activation_function is None:
            animation_group = AnimationGroup(
                succession,
                self.activation_function.make_evaluate_animation(),
                lag_ratio=0.0,
            )
            return animation_group
        else:
            return succession

    def get_center(self):
        return self.surrounding_circle.get_center()

    def get_left(self):
        return self.surrounding_circle.get_left()

    def get_right(self):
        return self.surrounding_circle.get_right()
    
    def move_to(self, mobject_or_point):
        """Moves the center of the layer to the given mobject or point"""
        layer_center = self.surrounding_circle.get_center()
        if isinstance(mobject_or_point, Mobject):
            target_center = mobject_or_point.get_center() 
        else:
            target_center = mobject_or_point

        self.shift(target_center - layer_center)