from manim import *
from abc import ABC, abstractmethod

class NeuralNetworkLayer(ABC, VGroup):
    """Abstract Neural Network Layer class"""

    @abstractmethod
    def make_forward_pass_animation(self):
        pass

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

    def __init__(self, input_layer, output_layer, passing_flash=True,
                dot_radius=0.05, animation_dot_count=RED, edge_color=WHITE,
                edge_width=0.5):
        super(FeedForwardToFeedForward, self).__init__(input_layer, output_layer)
        self.passing_flash = passing_flash
        self.edge_color = edge_color
        self.dot_radius = dot_radius
        self.animation_dot_count = animation_dot_count
        self.edge_width = edge_width

        self.construct_edges()

    def construct_edges(self):
        # Go through each node in the two layers and make a connecting line
        edges = []
        for node_i in self.input_layer.node_group:
            for node_j in self.output_layer.node_group:
                line = Line(node_i.get_center(), node_j.get_center(), 
                            color=self.edge_color, stroke_width=self.edge_width)
                self.add(line)

        self.edges = VGroup(*edges)

    def make_forward_pass_animation(self, run_time=1):
        """Animation for passing information from one FeedForwardLayer to the next"""
        path_animations = []
        for edge in self.edges:
            dot = Dot(color=self.animation_dot_color, fill_opacity=1.0, radius=self.dot_radius)
            # Handle layering
            dot.set_z_index(1)
            # Add to dots group
            self.dots.add(dot)
            # Make the animation
            if self.passing_flash:
                anim = ShowPassingFlash(edge.copy().set_color(self.animation_dot_color), time_width=0.2, run_time=3)
            else:
                anim = MoveAlongPath(dot, edge, run_time=run_time, rate_function=sigmoid)
            path_animations.append(anim)

        path_animations = AnimationGroup(*path_animations)

        return path_animations

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
        surrounding_rectangle = SurroundingRectangle(self.node_group, color=self.rectangle_color, 
                                                    fill_color=self.rectangle_fill_color, fill_opacity=1.0, 
                                                    buff=self.layer_buffer, stroke_width=self.rectangle_stroke_width)
        # Add the objects to the class
        self.add(surrounding_rectangle, self.node_group)

    def make_forward_pass_animation(self):
        # make highlight animation
        succession = Succession(
            ApplyMethod(self.node_group.set_color, self.animation_dot_color, run_time=0.25),
            Wait(1.0),
            ApplyMethod(self.node_group.set_color, self.node_color, run_time=0.25),
        )

        return succession
 
