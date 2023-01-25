from manim import *
from abc import ABC, abstractmethod
import random

import manim_ml.neural_network.activation_functions.relu as relu

class ActivationFunction(ABC, VGroup):
    """Abstract parent class for defining activation functions"""

    def __init__(
        self,
        function_name=None,
        x_range=[-1, 1],
        y_range=[-1, 1],
        x_length=0.5,
        y_length=0.3,
        show_function_name=True,
        active_color=ORANGE,
        plot_color=BLUE,
        rectangle_color=WHITE,
    ):
        super(VGroup, self).__init__()
        self.function_name = function_name
        self.x_range = x_range
        self.y_range = y_range
        self.x_length = x_length
        self.y_length = y_length
        self.show_function_name = show_function_name
        self.active_color = active_color
        self.plot_color = plot_color
        self.rectangle_color = rectangle_color

        self.construct_activation_function()

    def construct_activation_function(self):
        """Makes the activation function"""
        # Make an axis
        self.axes = Axes(
            x_range=self.x_range,
            y_range=self.y_range,
            x_length=self.x_length,
            y_length=self.y_length,
            tips=False,
            axis_config={
                "include_numbers": False,
                "stroke_width": 0.5,
                "include_ticks": False,
            },
        )
        self.add(self.axes)
        # Surround the axis with a rounded rectangle.
        self.surrounding_rectangle = SurroundingRectangle(
            self.axes,
            corner_radius=0.05,
            buff=0.05,
            stroke_width=2.0,
            stroke_color=self.rectangle_color,
        )
        self.add(self.surrounding_rectangle)
        # Plot function on axis by applying it and showing in given range
        self.graph = self.axes.plot(
            lambda x: self.apply_function(x),
            x_range=self.x_range,
            stroke_color=self.plot_color,
            stroke_width=2.0,
        )
        self.add(self.graph)
        # Add the function name
        if self.show_function_name:
            function_name_text = Text(
                self.function_name, font_size=12, font="sans-serif"
            )
            function_name_text.next_to(self.axes, UP * 0.5)
            self.add(function_name_text)

    @abstractmethod
    def apply_function(self, x_val):
        """Evaluates function at given x_val"""
        if x_val == None:
            x_val = random.uniform(self.x_range[0], self.x_range[1])

    def make_evaluate_animation(self, x_val=None):
        """Evaluates the function at a random point in the x_range"""
        # Highlight the graph
        # TODO: Evaluate the function at the x_val and show a highlighted dot
        animation_group = Succession(
            AnimationGroup(
                ApplyMethod(self.graph.set_color, self.active_color),
                ApplyMethod(
                    self.surrounding_rectangle.set_stroke_color, self.active_color
                ),
                lag_ratio=0.0,
            ),
            Wait(1),
            AnimationGroup(
                ApplyMethod(self.graph.set_color, self.plot_color),
                ApplyMethod(
                    self.surrounding_rectangle.set_stroke_color, self.rectangle_color
                ),
                lag_ratio=0.0,
            ),
            lag_ratio=1.0,
        )

        return animation_group
