"""
    Layers that describe the connections between user layers.
"""
from manim import *
from manim_ml.neural_network.layers import NeuralNetworkLayer
from abc import ABC, abstractmethod

class ConnectiveLayer(NeuralNetworkLayer):
    """Forward pass animation for a given pair of layers"""

    @abstractmethod
    def __init__(self, input_layer, output_layer):
        super(NeuralNetworkLayer, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer

    @abstractmethod
    def make_forward_pass_animation(self):
        pass

class FeedForwardToFeedForward(ConnectiveLayer):
    """Layer for connecting FeedForward layer to FeedForwardLayer"""

    def __init__(self, input_layer, output_layer, passing_flash=True,
                dot_radius=0.05, animation_dot_color=RED, edge_color=WHITE,
                edge_width=0.5):
        super().__init__(input_layer, output_layer)
        self.passing_flash = passing_flash
        self.edge_color = edge_color
        self.dot_radius = dot_radius
        self.animation_dot_color = animation_dot_color
        self.edge_width = edge_width

        self.edges = self.construct_edges()
        self.add(self.edges)

    def construct_edges(self):
        # Go through each node in the two layers and make a connecting line
        edges = []
        for node_i in self.input_layer.node_group:
            for node_j in self.output_layer.node_group:
                line = Line(node_i.get_center(), node_j.get_center(), 
                            color=self.edge_color, stroke_width=self.edge_width)
                edges.append(line)

        edges = Group(*edges)
        return edges

    def make_forward_pass_animation(self, run_time=1):
        """Animation for passing information from one FeedForwardLayer to the next"""
        path_animations = []
        dots = []
        for edge in self.edges:
            dot = Dot(color=self.animation_dot_color, fill_opacity=1.0, radius=self.dot_radius)
            # Handle layering
            dot.set_z_index(1)
            # Add to dots group
            dots.append(dot)
            # Make the animation
            if self.passing_flash:
                print("passing flash")
                anim = ShowPassingFlash(edge.copy().set_color(self.animation_dot_color), time_width=0.2, run_time=3)
            else:
                anim = MoveAlongPath(dot, edge, run_time=run_time, rate_function=sigmoid)
            path_animations.append(anim)

        if not self.passing_flash:
            dots = Group(*dots)
            self.add(dots)

        path_animations = AnimationGroup(*path_animations)

        return path_animations

class ImageToFeedForward(ConnectiveLayer):
    """Image Layer to FeedForward layer"""

    def __init__(self, input_layer, output_layer, animation_dot_color=RED,
                dot_radius=0.05):
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius
        # Input assumed to be ImageLayer
        # Output assumed to be FeedForwardLayer
        super().__init__(input_layer, output_layer)

    def make_forward_pass_animation(self):
        """Makes dots diverge from the given location and move to the feed forward nodes decoder"""
        animations = []
        image_mobject = self.input_layer.image_mobject
        # Move the dots to the centers of each of the nodes in the FeedForwardLayer
        image_location  = image_mobject.get_center()
        for node in self.output_layer.node_group:
            new_dot = Dot(image_location, radius=self.dot_radius, color=self.animation_dot_color)
            per_node_succession = Succession(
                Create(new_dot),
                new_dot.animate.move_to(node.get_center()),
            )
            animations.append(per_node_succession)

        animation_group = AnimationGroup(*animations)
        return animation_group
