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
import warnings
import textwrap

from manim_ml.neural_network.layers import \
    FeedForwardLayer, FeedForwardToFeedForward, ImageLayer, \
    ImageToFeedForward, FeedForwardToImage, EmbeddingLayer, \
    EmbeddingToFeedForward, FeedForwardToEmbedding, TripletLayer, \
    TripletToFeedForward
    
class NeuralNetwork(Group):

    def __init__(self, input_layers, edge_color=WHITE, layer_spacing=0.8,
                    animation_dot_color=RED, edge_width=1.5, dot_radius=0.03):
        super(Group, self).__init__()
        self.input_layers = Group(*input_layers)
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.layer_spacing = layer_spacing
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius
        self.created = False
        # TODO take layer_node_count [0, (1, 2), 0] 
        # and make it have explicit distinct subspaces
        self._place_layers()
        self.connective_layers, self.all_layers = self._construct_connective_layers()
        # Center the whole diagram by default
        self.all_layers.move_to(ORIGIN)
        self.add(self.all_layers)
        # print nn
        print(repr(self))

    def _place_layers(self):
        """Creates the neural network"""
        # TODO implement more sophisticated custom layouts
        # Default: Linear layout
        for layer_index in range(1, len(self.input_layers)):
            previous_layer = self.input_layers[layer_index - 1]
            current_layer = self.input_layers[layer_index]

            current_layer.move_to(previous_layer)
            shift_vector = np.array([(previous_layer.get_width()/2 + current_layer.get_width()/2) + 0.2, 0, 0])
            current_layer.shift(shift_vector)
        # Handle layering
        self.input_layers.set_z_index(2)

    def _construct_connective_layers(self):
        """Draws connecting lines between layers"""
        connective_layers = Group()
        all_layers = Group()
        for layer_index in range(len(self.input_layers) - 1):
            current_layer = self.input_layers[layer_index]
            all_layers.add(current_layer)
            next_layer = self.input_layers[layer_index + 1]
            # Check if layer is actually a nested NeuralNetwork
            if isinstance(current_layer, NeuralNetwork):
                # Last layer of the current layer
                current_layer = current_layer.all_layers[-1]
            if isinstance(next_layer, NeuralNetwork):
                # First layer of the next layer
                next_layer = next_layer.all_layers[0]
            if isinstance(current_layer, FeedForwardLayer) \
                and isinstance(next_layer, FeedForwardLayer):
                # FeedForward to Image
                edge_layer = FeedForwardToFeedForward(current_layer, next_layer, 
                                                    edge_width=self.edge_width)
                connective_layers.add(edge_layer)
                all_layers.add(edge_layer)
            elif isinstance(current_layer, ImageLayer) \
                and isinstance(next_layer, FeedForwardLayer):
                # Image to FeedForward
                image_to_feedforward = ImageToFeedForward(current_layer, next_layer, dot_radius=self.dot_radius)
                connective_layers.add(image_to_feedforward)
                all_layers.add(image_to_feedforward)
            elif isinstance(current_layer, FeedForwardLayer) \
                and isinstance(next_layer, ImageLayer):
                # Image to FeedForward
                feed_forward_to_image = FeedForwardToImage(current_layer, next_layer, dot_radius=self.dot_radius)
                connective_layers.add(feed_forward_to_image)
                all_layers.add(feed_forward_to_image) 
            elif isinstance(current_layer, FeedForwardLayer) \
                and isinstance(next_layer, EmbeddingLayer):
                # FeedForward to Embedding
                layer = FeedForwardToEmbedding(current_layer, next_layer, 
                                                animation_dot_color=self.animation_dot_color, dot_radius=self.dot_radius)
                connective_layers.add(layer)
                all_layers.add(layer)
            elif isinstance(current_layer, EmbeddingLayer) \
                and isinstance(next_layer, FeedForwardLayer): 
                # Embedding to FeedForward
                layer = EmbeddingToFeedForward(current_layer, next_layer, 
                                                animation_dot_color=self.animation_dot_color, dot_radius=self.dot_radius)
                connective_layers.add(layer)
                all_layers.add(layer)
            elif isinstance(current_layer, TripletLayer) \
                and isinstance(next_layer, FeedForwardLayer):
                # TripletLayer to FeedForwardLayer
                layer = TripletToFeedForward(current_layer, next_layer)
                connective_layers.add(layer)
                all_layers.add(layer)
            else:
                warnings.warn(f"Warning: unimplemented connection for layer types: {type(current_layer)} and {type(next_layer)}")
        # Add final layer
        all_layers.add(self.input_layers[-1])
        # Handle layering
        return connective_layers, all_layers

    def make_forward_pass_animation(self, run_time=10, passing_flash=True):
        """Generates an animation for feed forward propogation"""
        all_animations = []
        for layer_index, layer in enumerate(self.input_layers[:-1]):
            layer_forward_pass = layer.make_forward_pass_animation()
            all_animations.append(layer_forward_pass)
            connective_layer = self.connective_layers[layer_index]
            connective_forward_pass = connective_layer.make_forward_pass_animation()
            all_animations.append(connective_forward_pass)
        # Do last layer animation
        last_layer_forward_pass = self.input_layers[-1].make_forward_pass_animation()
        all_animations.append(last_layer_forward_pass)
        # Make the animation group
        animation_group = AnimationGroup(*all_animations, run_time=run_time, lag_ratio=1.0)

        return animation_group

    @override_animation(Create)
    def _create_override(self, **kwargs):
        """Overrides Create animation"""
        # Stop the neural network from being created twice
        if self.created:
            return AnimationGroup()
        self.created = True
        # Create each layer one by one
        animations = []
        for layer in self.all_layers:
            print(layer)
            animation = Create(layer)
            animations.append(animation)

        animation_group = AnimationGroup(*animations, lag_ratio=1.0)
        
        return animation_group

    def remove_layer(self, layer_index):
        """Removes layer at given index and returns animation for removing the layer"""
        raise NotImplementedError()

    def add_layer(self, layer):
        """Adds layer and returns animation for adding action"""
        raise NotImplementedError()

    def __repr__(self):
        """Print string representation of layers"""
        inner_string = ""
        for layer in self.all_layers:
            inner_string += f"{repr(layer)},\n"
        inner_string = textwrap.indent(inner_string, "    ")

        string_repr = "NeuralNetwork([\n" + inner_string + "])"
        return string_repr

class FeedForwardNeuralNetwork(NeuralNetwork):
    """NeuralNetwork with just feed forward layers"""

    def __init__(self, layer_node_count, node_radius=0.08, 
                node_color=BLUE, **kwargs):
        # construct layers
        layers = []
        for num_nodes in layer_node_count:
            layer = FeedForwardLayer(num_nodes, node_color=node_color, node_radius=node_radius)
            layers.append(layer)
        # call super class
        super().__init__(layers, **kwargs)