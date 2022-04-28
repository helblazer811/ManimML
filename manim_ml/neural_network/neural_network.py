"""Neural Network Manim Visualization

This module is responsible for generating a neural network visualization with
manim, specifically a fully connected neural network diagram.

Example:
    # Specify how many nodes are in each node layer
    layer_node_count = [5, 3, 5]
    # Create the object with default style settings
    NeuralNetwork(layer_node_count)
"""
from cv2 import AGAST_FEATURE_DETECTOR_NONMAX_SUPPRESSION
from manim import *
import warnings
import textwrap
from manim_ml.neural_network.layers.embedding import EmbeddingLayer

from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer
from manim_ml.neural_network.layers.util import get_connective_layer
from manim_ml.list_group import ListGroup
from manim_ml.neural_network.neural_network_transformations import InsertLayer, RemoveLayer

class NeuralNetwork(Group):

    def __init__(self, input_layers, edge_color=WHITE, layer_spacing=0.2,
                    animation_dot_color=RED, edge_width=2.5, dot_radius=0.03,
                    title=" "):
        super(Group, self).__init__()
        self.input_layers = ListGroup(*input_layers)
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.layer_spacing = layer_spacing
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius
        self.title_text = title
        self.created = False
        # TODO take layer_node_count [0, (1, 2), 0] 
        # and make it have explicit distinct subspaces
        self._place_layers()
        self.connective_layers, self.all_layers = self._construct_connective_layers()
        # Make overhead title
        self.title = Text(self.title_text, font_size=DEFAULT_FONT_SIZE/2)
        self.title.next_to(self, UP, 1.0)
        self.add(self.title)
        # Place layers at correct z index
        self.connective_layers.set_z_index(2)
        self.input_layers.set_z_index(3)
        # Center the whole diagram by default
        self.all_layers.move_to(ORIGIN)
        self.add(self.all_layers)
        # Print neural network
        print(repr(self))

    def _place_layers(self):
        """Creates the neural network"""
        # TODO implement more sophisticated custom layouts
        # Default: Linear layout
        for layer_index in range(1, len(self.input_layers)):
            previous_layer = self.input_layers[layer_index - 1]
            current_layer = self.input_layers[layer_index]
            current_layer.move_to(previous_layer)
            # TODO Temp fix
            if isinstance(current_layer, EmbeddingLayer) or isinstance(previous_layer, EmbeddingLayer):
                shift_vector = np.array([(previous_layer.get_width()/2 + current_layer.get_width()/2 - 0.2), 0, 0]) 
            else:
                shift_vector = np.array([(previous_layer.get_width()/2 + current_layer.get_width()/2) + self.layer_spacing, 0, 0])
            current_layer.shift(shift_vector)

    def _construct_connective_layers(self):
        """Draws connecting lines between layers"""
        connective_layers = ListGroup()
        all_layers = ListGroup()
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
            # Find connective layer with correct layer pair
            connective_layer = get_connective_layer(current_layer, next_layer)
            connective_layers.add(connective_layer)
            all_layers.add(connective_layer)
        # Add final layer
        all_layers.add(self.input_layers[-1])
        # Handle layering
        return connective_layers, all_layers

    def insert_layer(self, layer, insert_index):
        """Inserts a layer at the given index"""
        neural_network = self
        insert_animation = InsertLayer(layer, insert_index, neural_network)
        return insert_animation

    def remove_layer(self, layer):
        """Removes layer object if it exists"""
        neural_network = self
        return RemoveLayer(layer, neural_network, layer_spacing=self.layer_spacing)

    def replace_layer(self, old_layer, new_layer):
        """Replaces given layer object"""
        remove_animation = self.remove_layer(insert_index)
        insert_animation = self.insert_layer(layer, insert_index)
        # Make the animation
        animation_group = AnimationGroup(
            FadeOut(self.all_layers[insert_index]),
            FadeIn(layer),
            lag_ratio=1.0
        )

        return animation_group

    def make_forward_pass_animation(self, run_time=10, passing_flash=True, layer_args={},
                                    **kwargs):
        """Generates an animation for feed forward propagation"""
        all_animations = []
        for layer_index, layer in enumerate(self.all_layers):
            # Get the layer args
            if isinstance(layer, ConnectiveLayer):
                """
                    NOTE: By default a connective layer will get the combined
                    layer_args of the layers it is connecting.
                """
                before_layer_args = {}
                after_layer_args = {}
                if layer.input_layer in layer_args:
                    before_layer_args = layer_args[layer.input_layer]
                if layer.output_layer in layer_args:
                    after_layer_args = layer_args[layer.output_layer]
                # Merge the two dicts
                current_layer_args = {**before_layer_args, **after_layer_args}
            else:
                current_layer_args = {}
                if layer in layer_args:
                    current_layer_args = layer_args[layer]
            # Perform the forward pass of the current layer
            layer_forward_pass = layer.make_forward_pass_animation(layer_args=current_layer_args, **kwargs)
            all_animations.append(layer_forward_pass)
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

        animations = []
        # Create the overhead title
        animations.append(Create(self.title))
        # Create each layer one by one
        for layer in self.all_layers:
            layer_animation = Create(layer)
            # Make titles
            create_title = Create(layer.title)
            # Create layer animation group
            animation_group = AnimationGroup(
                layer_animation, 
                create_title
            )
            animations.append(animation_group)

        animation_group = AnimationGroup(*animations, lag_ratio=1.0)
        
        return animation_group

    def set_z_index(self, z_index_value: float, family=False):
        """Overriden set_z_index"""
        # Setting family=False stops sub-neural networks from inheriting parent z_index
        return super().set_z_index(z_index_value, family=False)

    def __repr__(self, metadata=["z_index", "title_text"]):
        """Print string representation of layers"""
        inner_string = ""
        for layer in self.all_layers:
            inner_string += f"{repr(layer)} ("
            for key in metadata: 
                value = getattr(layer, key)
                if not value is "":
                    inner_string += f"{key}={value}, "
            inner_string += "),\n"
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