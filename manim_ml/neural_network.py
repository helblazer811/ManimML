"""Neural Network Manim Visualization

This module is responsible for generating a neural network visualization with
manim, specifically a fully connected neural network diagram.

Example:
    # Specify how many nodes are in each node layer
    layer_node_count = [5, 3, 5]
    # Create the object with default style settings
    NeuralNetwork(layer_node_count)
"""
from manim import *

class NeuralNetworkLayer(VGroup):
    """Handles rendering a layer for a neural network"""

    def __init__(
            self, num_nodes, layer_buffer=SMALL_BUFF/2, node_radius=0.08,
            node_color=BLUE, node_outline_color=WHITE, rectangle_color=WHITE,
            node_spacing=0.3, rectangle_fill_color=BLACK, node_stroke_width=2.0,
            rectangle_stroke_width=2.0):
        super(VGroup, self).__init__()
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

        self.node_group = VGroup()

        self._construct_neural_network_layer()

    def _construct_neural_network_layer(self):
        """Creates the neural network layer"""
        # Add Nodes
        for node_number in range(self.num_nodes):
            node_object = Circle(radius=self.node_radius, color=self.node_color, stroke_width=self.node_stroke_width)
            self.node_group.add(node_object)
        # Space the nodes
        # Assumes Vertical orientation
        for node_index, node_object in enumerate(self.node_group):
            location = node_index * self.node_spacing
            node_object.move_to([0, location, 0])
        # Create Surrounding Rectangle
        surrounding_rectangle = SurroundingRectangle(
            self.node_group, color=self.rectangle_color, fill_color=self.rectangle_fill_color,
            fill_opacity=1.0, buff=self.layer_buffer, stroke_width=self.rectangle_stroke_width
        )
        # Add the objects to the class
        self.add(surrounding_rectangle, self.node_group)

class NeuralNetwork(VGroup):

    def __init__(
            self, layer_node_count, layer_width=0.6, node_radius=1.0,
            node_color=BLUE, edge_color=WHITE, layer_spacing=0.8,
            animation_dot_color=RED, edge_width=2.0, dot_radius=0.05):
        super(VGroup, self).__init__()
        self.layer_node_count = layer_node_count
        self.layer_width = layer_width
        self.node_radius = node_radius
        self.edge_width = edge_width
        self.node_color = node_color
        self.edge_color = edge_color
        self.layer_spacing = layer_spacing
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius

        # TODO take layer_node_count [0, (1, 2), 0] 
        # and make it have explicit distinct subspaces
        self.layers = self._construct_layers()
        self.edge_layers = self._construct_edges()

        self.add(self.edge_layers)
        self.add(self.layers)

    def _construct_layers(self):
        """Creates the neural network"""
        layers = VGroup()
        # Create each layer
        for layer_index, node_count in enumerate(self.layer_node_count):
            layer = NeuralNetworkLayer(node_count, node_color=self.node_color)
            # Manage spacing
            layer.move_to([self.layer_spacing * layer_index, 0, 0])
            # Add layer to VGroup
            layers.add(layer)
        # Handle layering
        layers.set_z_index(2)
        return layers

    def _construct_edges(self):
        """Draws connecting lines between layers"""
        edge_layers = VGroup()
        for layer_index in range(len(self.layer_node_count) - 1):
            current_layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]
            edge_layer = VGroup()
            # Go through each node in the two layers and make a connecting line
            for node_i in current_layer.node_group:
                for node_j in next_layer.node_group:
                    line = Line(node_i.get_center(), node_j.get_center(), color=self.edge_color, stroke_width=self.edge_width)
                    edge_layer.add(line)
            edge_layers.add(edge_layer)
        # Handle layering
        edge_layers.set_z_index(0)
        return edge_layers

    def make_forward_propagation_animation(self, run_time=2):
        """Generates an animation for feed forward propogation"""
        all_animations = []
        per_layer_run_time = run_time / len(self.edge_layers)
        self.dots = VGroup()
        for edge_layer in self.edge_layers:
            path_animations = []
            for edge in edge_layer:
                dot = Dot(color=self.animation_dot_color, fill_opacity=1.0, radius=self.dot_radius)
                # Handle layering
                dot.set_z_index(1)
                # Add to dots group
                self.dots.add(dot)
                # Make the animation
                anim = MoveAlongPath(dot, edge, run_time=per_layer_run_time, rate_function=sigmoid)
                path_animations.append(anim)
            path_animation_group = AnimationGroup(*path_animations)
            all_animations.append(path_animation_group)

        animation_group = AnimationGroup(*all_animations, run_time=run_time, lag_ratio=1)

        return animation_group

config.pixel_height = 720
config.pixel_width = 1280
config.frame_height = 6.0
config.frame_width = 6.0

class TestNeuralNetworkScene(Scene):
    """Test Scene for the Neural Network"""

    def construct(self):
        # Make the Layer object
        num_nodes = [8, 5, 3, 5]
        nn = NeuralNetwork(num_nodes)
        nn.move_to(ORIGIN)
        # Make Animation
        self.add(nn)
        forward_propagation_animation = nn.make_forward_propagation_animation()

        second_nn = NeuralNetwork([3, 4])
        self.add(second_nn)

        self.play(forward_propagation_animation)
        self.play(second_nn.make_forward_propagation_animation())
