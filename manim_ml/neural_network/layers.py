from typing import overload
from manim import *
from abc import ABC, abstractmethod

from matplotlib import animation
from manim_ml.image import GrayscaleImageMobject

class NeuralNetworkLayer(ABC, VGroup):
    """Abstract Neural Network Layer class"""

    @abstractmethod
    def make_forward_pass_animation(self):
        pass

class FeedForwardLayer(NeuralNetworkLayer):
    """Handles rendering a layer for a neural network"""

    def __init__(self, num_nodes, layer_buffer=SMALL_BUFF/2, node_radius=0.08,
                node_color=BLUE, node_outline_color=WHITE, rectangle_color=WHITE,
                node_spacing=0.3, rectangle_fill_color=BLACK, node_stroke_width=2.0,
                rectangle_stroke_width=2.0, animation_dot_color=RED):
        super(NeuralNetworkLayer, self).__init__()
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

        self.node_group = VGroup()

        self._construct_neural_network_layer()

    def _construct_neural_network_layer(self):
        """Creates the neural network layer"""
        # Add Nodes
        for node_number in range(self.num_nodes):
            node_object = Circle(radius=self.node_radius, color=self.node_color, 
                                stroke_width=self.node_stroke_width)
            self.node_group.add(node_object)
        # Space the nodes
        # Assumes Vertical orientation
        for node_index, node_object in enumerate(self.node_group):
            location = node_index * self.node_spacing
            node_object.move_to([0, location, 0])
        # Create Surrounding Rectangle
        self.surrounding_rectangle = SurroundingRectangle(self.node_group, color=self.rectangle_color, 
                                                    fill_color=self.rectangle_fill_color, fill_opacity=1.0, 
                                                    buff=self.layer_buffer, stroke_width=self.rectangle_stroke_width)
        # Add the objects to the class
        self.add(self.surrounding_rectangle, self.node_group)

    def make_forward_pass_animation(self):
        # make highlight animation
        succession = Succession(
            ApplyMethod(self.node_group.set_color, self.animation_dot_color, run_time=0.25),
            Wait(1.0),
            ApplyMethod(self.node_group.set_color, self.node_color, run_time=0.25),
        )

        return succession

    @override_animation(Create)
    def _create_animation(self, **kwargs):
        animations = []

        animations.append(Create(self.surrounding_rectangle))

        for node in self.node_group:
            animations.append(Create(node))

        animation_group = AnimationGroup(*animations, lag_ratio=0.0)
        return animation_group
 
class ImageLayer(NeuralNetworkLayer):
    """Image Layer for Neural Network"""

    def __init__(self, numpy_image, height=1.5):
        super().__init__()
        self.numpy_image = numpy_image
        if len(np.shape(self.numpy_image)) == 2:
            # Assumed Grayscale
            self.image_mobject = GrayscaleImageMobject(self.numpy_image, height=height)
        elif len(np.shape(self.numpy_image)) == 3:
            # Assumed RGB
            self.image_mobject = ImageMobject(self.numpy_image)

    @override_animation(Create)
    def _create_animation(self, **kwargs):
        return FadeIn(self.image_mobject)

    def make_forward_pass_animation(self):
        return Create(self.image_mobject)

    @property
    def width(self):
        return self.image_mobject.width
