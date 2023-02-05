"""Neural Network Manim Visualization

This module is responsible for generating a neural network visualization with
manim, specifically a fully connected neural network diagram.

Example:
    # Specify how many nodes are in each node layer
    layer_node_count = [5, 3, 5]
    # Create the object with default style settings
    NeuralNetwork(layer_node_count)
"""
import textwrap
from manim_ml.neural_network.layers.embedding import EmbeddingLayer
from manim_ml.utils.mobjects.connections import NetworkConnection
import numpy as np
from manim import *

from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer, ThreeDLayer
from manim_ml.neural_network.layers.util import get_connective_layer
from manim_ml.utils.mobjects.list_group import ListGroup
from manim_ml.neural_network.animations.neural_network_transformations import (
    InsertLayer,
    RemoveLayer,
)


class NeuralNetwork(Group):
    """Neural Network Visualization Container Class"""

    def __init__(
        self,
        input_layers,
        edge_color=WHITE,
        layer_spacing=0.2,
        animation_dot_color=RED,
        edge_width=2.5,
        dot_radius=0.03,
        title=" ",
        layout="linear",
        layout_direction="left_to_right",
        debug_mode=False
    ):
        super(Group, self).__init__()
        self.input_layers_dict = self.make_input_layers_dict(input_layers)
        self.input_layers = ListGroup(*self.input_layers_dict.values())
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.layer_spacing = layer_spacing
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius
        self.title_text = title
        self.created = False
        self.layout = layout
        self.layout_direction = layout_direction
        self.debug_mode = debug_mode
        # TODO take layer_node_count [0, (1, 2), 0]
        # and make it have explicit distinct subspaces
        # Construct all of the layers
        self._construct_input_layers()
        # Place the layers
        self._place_layers(layout=layout, layout_direction=layout_direction)
        # Make the connective layers
        self.connective_layers, self.all_layers = self._construct_connective_layers()
        # Make overhead title
        self.title = Text(self.title_text, font_size=DEFAULT_FONT_SIZE / 2)
        self.title.next_to(self, UP, 1.0)
        self.add(self.title)
        # Place layers at correct z index
        self.connective_layers.set_z_index(2)
        self.input_layers.set_z_index(3)
        # Center the whole diagram by default
        self.all_layers.move_to(ORIGIN)
        self.add(self.all_layers)
        # Make container for connections
        self.connections = []
        # Print neural network
        print(repr(self))

    def make_input_layers_dict(self, input_layers):
        """Make dictionary of input layers"""
        if isinstance(input_layers, dict):
            # If input layers is dictionary then return it
            return input_layers
        elif isinstance(input_layers, list):
            # If input layers is a list then make a dictionary with default
            return_dict = {}
            for layer_index, input_layer in enumerate(input_layers):
                return_dict[f"layer{layer_index}"] = input_layer

            return return_dict
        else:
            raise Exception(f"Uncrecognized input layers type: {type(input_layers)}")

    def add_connection(
        self,
        start_mobject_or_name,
        end_mobject_or_name,
        connection_style="default",
        connection_position="bottom",
        arc_direction="down"
    ):
        """Add connection from start layer to end layer"""
        assert connection_style in ["default"]
        if connection_style == "default":
            # Make arrow connection from start layer to end layer
            # Add the connection
            if isinstance(start_mobject_or_name, Mobject):
                input_mobject = start_mobject_or_name
            else:
                input_mobject = self.input_layers_dict[start_mobject_or_name]
            if isinstance(end_mobject_or_name, Mobject):
                output_mobject = end_mobject_or_name
            else:
                output_mobject = self.input_layers_dict[end_mobject_or_name]
            
            connection = NetworkConnection(
                input_mobject,
                output_mobject,
                arc_direction=arc_direction,
                buffer=0.05
            )
            self.connections.append(connection)
            self.add(connection)

    def _construct_input_layers(self):
        """Constructs each of the input layers in context
        of their adjacent layers"""
        prev_layer = None
        next_layer = None
        # Go through all the input layers and run their construct method
        print("Constructing layers")
        for layer_index in range(len(self.input_layers)):
            current_layer = self.input_layers[layer_index]
            print(f"Current layer: {current_layer}")
            if layer_index < len(self.input_layers) - 1:
                next_layer = self.input_layers[layer_index + 1]
            if layer_index > 0:
                prev_layer = self.input_layers[layer_index - 1]
            # Run the construct layer method for each
            current_layer.construct_layer(
                prev_layer, 
                next_layer,
                debug_mode=self.debug_mode
            )

    def _place_layers(
        self, 
        layout="linear", 
        layout_direction="top_to_bottom"
    ):
        """Creates the neural network"""
        # TODO implement more sophisticated custom layouts
        # Default: Linear layout
        for layer_index in range(1, len(self.input_layers)):
            previous_layer = self.input_layers[layer_index - 1]
            current_layer = self.input_layers[layer_index]
            current_layer.move_to(previous_layer.get_center())

            if layout_direction == "left_to_right":
                x_shift = (
                    previous_layer.get_width() / 2
                    + current_layer.get_width() / 2
                    + self.layer_spacing
                )
                shift_vector = np.array([x_shift, 0, 0])
            elif layout_direction == "top_to_bottom":
                y_shift = -(
                    (previous_layer.get_width() / 2 + current_layer.get_width() / 2)
                    + self.layer_spacing
                )
                shift_vector = np.array([0, y_shift, 0])
            else:
                raise Exception(f"Unrecognized layout direction: {layout_direction}")
            current_layer.shift(shift_vector)

        # After all layers have been placed place their activation functions
        for current_layer in self.input_layers:
            # Place activation function
            if hasattr(current_layer, "activation_function"):
                if not current_layer.activation_function is None:
                    up_movement = np.array(
                        [
                            0,
                            current_layer.get_height() / 2
                            + current_layer.activation_function.get_height() / 2
                            + 0.5 * self.layer_spacing,
                            0,
                        ]
                    )
                    current_layer.activation_function.move_to(
                        current_layer,
                    )
                    current_layer.activation_function.shift(up_movement)
                    self.add(current_layer.activation_function)

    def _construct_connective_layers(self):
        """Draws connecting lines between layers"""
        connective_layers = ListGroup()
        all_layers = ListGroup()
        for layer_index in range(len(self.input_layers) - 1):
            current_layer = self.input_layers[layer_index]
            # Add the layer to the list of layers
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
            # Construct the connective layer
            connective_layer.construct_layer(current_layer, next_layer)
            # Add the layer to the list of layers
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
        raise NotImplementedError()
        remove_animation = self.remove_layer(insert_index)
        insert_animation = self.insert_layer(layer, insert_index)
        # Make the animation
        animation_group = AnimationGroup(
            FadeOut(self.all_layers[insert_index]), FadeIn(layer), lag_ratio=1.0
        )

        return animation_group

    def make_forward_pass_animation(
        self, 
        run_time=None, 
        passing_flash=True, 
        layer_args={}, 
        per_layer_animations=False,
        **kwargs
    ):
        """Generates an animation for feed forward propagation"""
        all_animations = []
        per_layer_animation_map = {}
        per_layer_runtime = (
            run_time / len(self.all_layers) if not run_time is None else None
        )
        for layer_index, layer in enumerate(self.all_layers):
            # Get the layer args
            if isinstance(layer, ConnectiveLayer):
                """
                NOTE: By default a connective layer will get the combined
                layer_args of the layers it is connecting and itself.
                """
                before_layer_args = {}
                current_layer_args = {}
                after_layer_args = {}
                if layer.input_layer in layer_args:
                    before_layer_args = layer_args[layer.input_layer]
                if layer in layer_args:
                    current_layer_args = layer_args[layer]
                if layer.output_layer in layer_args:
                    after_layer_args = layer_args[layer.output_layer]
                # Merge the two dicts
                current_layer_args = {
                    **before_layer_args,
                    **current_layer_args,
                    **after_layer_args,
                }
            else:
                current_layer_args = {}
                if layer in layer_args:
                    current_layer_args = layer_args[layer]
            # Perform the forward pass of the current layer
            layer_forward_pass = layer.make_forward_pass_animation(
                layer_args=current_layer_args, run_time=per_layer_runtime, **kwargs
            )
            # Animate a forward pass for incoming connections
            connection_input_pass = AnimationGroup()
            for connection in self.connections:
                if isinstance(layer, ConnectiveLayer):
                    output_layer = layer.output_layer
                    if connection.end_mobject == output_layer:
                        connection_input_pass = ShowPassingFlash(
                            connection,
                            run_time=layer_forward_pass.run_time,
                            time_width=0.2,
                        )
                        break

            layer_forward_pass = AnimationGroup(
                layer_forward_pass, 
                connection_input_pass, 
                lag_ratio=0.0
            )
            all_animations.append(layer_forward_pass)
            # Add the animation to per layer animation
            per_layer_animation_map[layer] = layer_forward_pass
        # Make the animation group
        animation_group = Succession(*all_animations, lag_ratio=1.0)
        if per_layer_animations:
            return per_layer_animation_map
        else:
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
            animation_group = AnimationGroup(layer_animation, create_title)
            animations.append(animation_group)

        animation_group = AnimationGroup(*animations, lag_ratio=1.0)

        return animation_group

    def set_z_index(self, z_index_value: float, family=False):
        """Overriden set_z_index"""
        # Setting family=False stops sub-neural networks from inheriting parent z_index
        for layer in self.all_layers:
            if not isinstance(NeuralNetwork):
                layer.set_z_index(z_index_value)

    def scale(self, scale_factor, **kwargs):
        """Overriden scale"""
        for layer in self.all_layers:
            layer.scale(scale_factor, **kwargs)
        # Place layers with scaled spacing
        self.layer_spacing *= scale_factor
        self._place_layers(layout=self.layout, layout_direction=self.layout_direction)

    def filter_layers(self, function):
        """Filters layers of the network given function"""
        layers_to_return = []
        for layer in self.all_layers:
            func_out = function(layer)
            assert isinstance(
                func_out, bool
            ), "Filter layers function returned a non-boolean type."
            if func_out:
                layers_to_return.append(layer)

        return layers_to_return

    def __repr__(self, metadata=["z_index", "title_text"]):
        """Print string representation of layers"""
        inner_string = ""
        for layer in self.all_layers:
            inner_string += f"{repr(layer)}("
            for key in metadata:
                value = getattr(layer, key)
                if not value is "":
                    inner_string += f"{key}={value}, "
            inner_string += "),\n"
        inner_string = textwrap.indent(inner_string, "    ")

        string_repr = "NeuralNetwork([\n" + inner_string + "])"
        return string_repr
