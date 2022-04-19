"""Neural Network Manim Visualization

This module is responsible for generating a neural network visualization with
manim, specifically a fully connected neural network diagram.

Example:
    # Specify how many nodes are in each node layer
    layer_node_count = [5, 3, 5]
    # Create the object with default style settings
    NeuralNetwork(layer_node_count)
"""
from socket import create_connection
from urllib.parse import non_hierarchical
from manim import *
import warnings
import textwrap

from manim_ml.neural_network.layers import connective_layers_list
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.util import get_connective_layer
from manim_ml.list_group import ListGroup
    
class NeuralNetwork(Group):

    def __init__(self, input_layers, edge_color=WHITE, layer_spacing=0.8,
                    animation_dot_color=RED, edge_width=2.5, dot_radius=0.03):
        super(Group, self).__init__()
        self.input_layers = ListGroup(*input_layers)
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
            shift_vector = np.array([(previous_layer.get_width()/2 + current_layer.get_width()/2) + 0.2, 0, 0])
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
        layers_before = self.all_layers[:insert_index]
        layers_after = self.all_layers[insert_index:]
        # Make connective layers and shift animations
        # Before layer
        if len(layers_before) > 0:
            before_connective = get_connective_layer(layers_before[-1], layer)
            before_shift = np.array([-layer.width/2, 0, 0])
            # Shift layers before
            before_shift_animation = Group(*layers_before).animate.shift(before_shift)
        else:
            before_connective = AnimationGroup()
        # After layer
        if len(layers_after) > 0:
            after_connective = get_connective_layer(layer, layers_after[0])
            after_shift = np.array([layer.width/2, 0, 0])
            # Shift layers after
            after_shift_animation = Group(*layers_after).animate.shift(after_shift)
        else:
            after_connective = AnimationGroup

        # Make animation group
        shift_animations = AnimationGroup(
            before_shift_animation, 
            after_shift_animation
        )

        insert_animation = Create(layer)
        animation_group = AnimationGroup(
            shift_animations, 
            insert_animation,
            lag_ratio=1.0
        )

        return animation_group

    def remove_layer(self, layer):
        """Removes layer object if it exists"""
        # Get layer index
        layer_index = self.all_layers.index_of(layer)
        if layer_index == -1:
            raise Exception("Layer object not found")
        # Get the layers before and after
        before_layer = None
        after_layer = None
        if layer_index - 2 >= 0:
            before_layer = self.all_layers[layer_index - 2]
        if layer_index + 2 < len(self.all_layers): 
            after_layer = self.all_layers[layer_index + 2]
        # Remove the layer
        self.all_layers.remove(layer)
        # Remove surrounding connective layers from self.all_layers
        before_connective = None
        after_connective = None
        if layer_index - 1 >= 0:
            # There is a layer before
            before_connective = self.all_layers.remove_at_index(layer_index - 1)
        if layer_index + 1 < len(self.all_layers):
            # There is a layer after
            after_connective = self.all_layers.remove_at_index(layer_index + 1)
        # Make animations
        # Fade out the removed layer
        fade_out_removed = FadeOut(layer)
        # Fade out the removed connective layers
        fade_out_before_connective = Animation()
        if not before_connective is None:
            fade_out_before_connective = FadeOut(before_connective)
        fade_out_after_connective = Animation()
        if not after_connective is None:
            fade_out_after_connective = FadeOut(after_connective)
        # Create new connective layer
        new_connective = None
        if not before_layer is None and not after_layer is None:
            new_connective = get_connective_layer(before_layer, after_layer)
            before_layer_index = self.all_layers.index_of(before_layer)
            self.all_layers.insert(before_layer_index, new_connective)
        # Place the new connective
        new_connective.move_to(layer)
        # Animate the creation of the new connective layer
        create_new_connective = Animation()
        if not new_connective is None:
            create_new_connective = Create(new_connective)
        # Collapse the neural network to fill the empty space
        removed_width = layer.width + before_connective.width + after_connective.width - new_connective.width
        shift_right_amount = np.array([removed_width / 2, 0, 0])
        shift_left_amount = np.array([-removed_width / 2, 0, 0])
        move_before_layer = Animation()
        if not before_layer is None:
            move_before_layer = before_layer.animate.shift(shift_right_amount)
        move_after_layer = Animation()
        if not after_layer is None:
            move_after_layer = after_layer.animate.shift(shift_left_amount)
        # Make the final AnimationGroup
        fade_out_group = AnimationGroup(
            fade_out_removed, 
            fade_out_before_connective, 
            fade_out_after_connective
        )
        move_group = AnimationGroup(
            move_before_layer, 
            move_after_layer
        )
        animation_group = AnimationGroup(
            fade_out_group,
            move_group,
            create_new_connective,
            lag_ratio=1.0
        )

        return animation_group

        """
        remove_layer = list(self.all_layers)[remove_index]
        if remove_index > 0:
            connective_before = list(self.all_layers)[remove_index - 1]
        else:
            connective_before = None
        if remove_index < len(list(self.all_layers)) - 1:
            connective_after = list(self.all_layers)[remove_index + 1]
        else:
            connective_after = None
        # Collapse the surrounding layer
        layers_before = list(self.all_layers)[:remove_index]
        layers_after = list(self.all_layers)[remove_index+1:]
        before_group = Group(*layers_before)
        after_group = Group(*layers_after)
        before_shift_amount = np.array([remove_layer.width/2, 0, 0])
        after_shift_amount = np.array([-remove_layer.width/2, 0, 0])
        # Remove the layers from the neural network representation
        self.all_layers.remove(remove_layer)
        if not connective_before is None:
            self.all_layers.remove(connective_before)
        if not connective_after is None:
            self.all_layers.remove(connective_after)
        # Connect the layers before and layers after
        pre_index = remove_index - 1
        pre_layer = None
        if pre_index >= 0:
            pre_layer = list(self.all_layers)[pre_index]
        post_index = remove_index
        post_layer = None
        if post_index < len(list(self.all_layers)):
            post_layer = list(self.all_layers)[post_index]
        if not pre_layer is None and not post_layer is None:
            connective_layer = get_connective_layer(pre_layer, post_layer)
            self.all_layers = Group(
                *self.all_layers[:remove_index], 
                connective_layer, 
                *self.all_layers[remove_index:]
            )
        # Make animations
        fade_out_animation = FadeOut(remove_layer)
        shift_animations = AnimationGroup(
            before_group.animate.shift(before_shift_amount),
            after_group.animate.shift(after_shift_amount)
        )
        animation_group = AnimationGroup(
            fade_out_animation,
            shift_animations,
            lag_ratio=1.0
        )

        return animation_group
        """

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

    def make_forward_pass_animation(self, run_time=10, passing_flash=True):
        """Generates an animation for feed forward propagation"""
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
            animation = Create(layer)
            animations.append(animation)

        animation_group = AnimationGroup(*animations, lag_ratio=1.0)
        
        return animation_group

    def __repr__(self):
        """Print string representation of layers"""
        inner_string = ""
        for layer in self.all_layers:
            inner_string += f"{repr(layer)} {layer.z_index} ,\n"
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