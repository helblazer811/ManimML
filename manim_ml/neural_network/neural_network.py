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
from matplotlib import animation
from manim_ml.neural_network.layers import FeedForwardToFeedForward, FeedForwardLayer

class NeuralNetwork(VGroup):

    def __init__(self, layers, edge_color=WHITE, layer_spacing=0.8,
                    animation_dot_color=RED, edge_width=1.5, dot_radius=0.05):
        super().__init__()
        self.layers = layers
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.layer_spacing = layer_spacing
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius

        # TODO take layer_node_count [0, (1, 2), 0] 
        # and make it have explicit distinct subspaces
        self.layers = self._construct_layers()
        self.connective_layers = self._construct_connective_layers()

        self.add(self.connective_layers)
        self.add(self.layers)

    def _construct_layers(self):
        """Creates the neural network"""
        layers = VGroup()
        # Create each layer
        for layer_index, layer in enumerate(self.layers):
            # Manage spacing
            layer.move_to([self.layer_spacing * layer_index, 0, 0])
            # Add layer to VGroup
            layers.add(layer)
        # Handle layering
        layers.set_z_index(2)
        return layers

    def _construct_connective_layers(self):
        """Draws connecting lines between layers"""
        connective_layers = VGroup()
        for layer_index in range(len(self.layers) - 1):
            current_layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]
            
            if isinstance(current_layer, FeedForwardLayer) \
                and isinstance(next_layer, FeedForwardLayer):
                edge_layer = FeedForwardToFeedForward(current_layer, next_layer, 
                                                    edge_width=self.edge_width)

                connective_layers.add(edge_layer)
            else:
                raise Exception(f"Unimplemented connection for layer types: {type(current_layer)} and {type(next_layer)}")

        # Handle layering
        connective_layers.set_z_index(0)
        return connective_layers

    def make_forward_pass_animation(self, run_time=2, passing_flash=True):
        """Generates an animation for feed forward propogation"""
        all_animations = []

        for layer_index, layer in enumerate(self.layers[:-1]):
            layer_forward_pass = layer.make_forward_pass_animation()
            all_animations.append(layer_forward_pass)

            connective_layer = self.connective_layers[layer_index]
            connective_forward_pass = connective_layer.make_forward_pass_animation()
            all_animations.append(connective_forward_pass)
            
        # Do last layer animation
        last_layer_forward_pass = self.layers[-1].make_forward_pass_animation()
        all_animations.append(last_layer_forward_pass)
        # Make the animation group
        animation_group = AnimationGroup(*all_animations, run_time=run_time, lag_ratio=1.0)

        return animation_group

class FeedForwardNeuralNetwork(NeuralNetwork):
    """NeuralNetwork with just feed forward layers"""

    def __init__(self, layer_node_count, node_radius=1.0, 
                node_color=BLUE, **kwargs):
        # construct layers
        layers = []
        for num_nodes in layer_node_count:
            layer = FeedForwardLayer(num_nodes, node_color=node_color, node_radius=node_radius)
            layers.append(layer)
        # call super class
        super().__init__(layers, **kwargs)