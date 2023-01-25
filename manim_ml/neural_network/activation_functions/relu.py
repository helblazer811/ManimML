from manim import *

from manim_ml.neural_network.activation_functions.activation_function import (
    ActivationFunction,
)


class ReLUFunction(ActivationFunction):
    """Rectified Linear Unit Activation Function"""

    def __init__(self, function_name="ReLU", x_range=[-1, 1], y_range=[-1, 1]):
        super().__init__(function_name, x_range, y_range)

    def apply_function(self, x_val):
        if x_val < 0:
            return 0
        else:
            return x_val
