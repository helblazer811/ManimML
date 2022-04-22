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

from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.util import get_connective_layer
from manim_ml.list_group import ListGroup

class RemoveLayer(AnimationGroup):
    """
        Animation for removing a layer from a neural network.

        Note: I needed to do something strange for creating the new connective layer.
        The issue with creating it intially is that the positions of the sides of the 
        connective layer depend upon the location of the moved layers **after** the
        move animations are performed. However, all of these animations are performed
        after the animations have been created. This means that the animation depends upon
        the state of the neural network layers after previous animations have been run. 
        To fix this issue I needed to use an UpdateFromFunc.
    """

    def __init__(self, layer, neural_network, layer_spacing=0.2):
        self.layer = layer
        self.neural_network = neural_network
        self.layer_spacing = layer_spacing
        # Get the before and after layers
        layers_tuple = self.get_connective_layers()
        self.before_layer = layers_tuple[0]
        self.after_layer = layers_tuple[1]
        self.before_connective = layers_tuple[2]
        self.after_connective = layers_tuple[3]
        # Make the animations
        remove_animations = self.make_remove_animation()
        move_animations = self.make_move_animation()
        new_connective_animation = self.make_new_connective_animation()
        # Add all of the animations to the group
        animations_list = [
            remove_animations,
            move_animations,
            new_connective_animation
        ]

        super().__init__(*animations_list, lag_ratio=1.0)

    def get_connective_layers(self):
        """Gets the connective layers before and after self.layer"""
        # Get layer index
        layer_index = self.neural_network.all_layers.index_of(self.layer)
        if layer_index == -1:
            raise Exception("Layer object not found")
        # Get the layers before and after
        before_layer = None
        after_layer = None
        before_connective = None
        after_connective = None
        if layer_index - 2 >= 0:
            before_layer = self.neural_network.all_layers[layer_index - 2]
            before_connective = self.neural_network.all_layers[layer_index - 1]
        if layer_index + 2 < len(self.neural_network.all_layers): 
            after_layer = self.neural_network.all_layers[layer_index + 2]
            after_connective = self.neural_network.all_layers[layer_index + 1]
        
        return before_layer, after_layer, before_connective, after_connective

    def make_remove_animation(self):
        """Removes layer and the surrounding connective layers"""
        remove_layer_animation = self.make_remove_layer_animation()
        remove_connective_animation = self.make_remove_connective_layers_animation()
        # Remove animations
        remove_animations = AnimationGroup(
            remove_layer_animation, 
            remove_connective_animation
        )

        return remove_animations

    def make_remove_layer_animation(self):
        """Removes the layer"""
        # Remove the layer
        self.neural_network.all_layers.remove(self.layer)
        # Fade out the removed layer
        fade_out_removed = FadeOut(self.layer)
        return fade_out_removed

    def make_remove_connective_layers_animation(self):
        """Removes the connective layers before and after layer if they exist"""
        # Fade out the removed connective layers
        fade_out_before_connective = AnimationGroup()
        if not self.before_connective is None:
            self.neural_network.all_layers.remove(self.before_connective)
            fade_out_before_connective = FadeOut(self.before_connective)
        fade_out_after_connective = AnimationGroup()
        if not self.after_connective is None:
            self.neural_network.all_layers.remove(self.after_connective)
            fade_out_after_connective = FadeOut(self.after_connective)
        # Group items
        remove_connective_group = AnimationGroup(
            fade_out_after_connective, 
            fade_out_before_connective
        )

        return remove_connective_group

    def make_move_animation(self):
        """Collapses layers"""
        # Animate the movements
        move_before_layers = AnimationGroup()
        shift_right_amount = None
        if not self.before_layer is None:
            # Compute shift amount
            layer_dist = np.abs(self.layer.get_center() - self.before_layer.get_right())[0]
            shift_right_amount = np.array([layer_dist - self.layer_spacing/2, 0, 0])
            # Shift all layers before forward
            before_layer_index = self.neural_network.all_layers.index_of(self.before_layer)
            layers_before = Group(*self.neural_network.all_layers[:before_layer_index + 1])
            move_before_layers = layers_before.animate.shift(shift_right_amount)
        move_after_layers = AnimationGroup()
        shift_left_amount = None
        if not self.after_layer is None:
            layer_dist = np.abs(self.after_layer.get_left() - self.layer.get_center())[0]
            shift_left_amount = np.array([-layer_dist + self.layer_spacing / 2, 0, 0])
            # Shift all layers after backward
            after_layer_index = self.neural_network.all_layers.index_of(self.after_layer)
            layers_after = Group(*self.neural_network.all_layers[after_layer_index:])
            move_after_layers = layers_after.animate.shift(shift_left_amount)
        # Group the move animations
        move_group = AnimationGroup(
            move_before_layers, 
            move_after_layers
        )

        return move_group

    def make_new_connective_animation(self):
        """Makes new connective layer"""
        self.anim_count = 0
        def create_new_connective(neural_network):
            """
                Creates new connective layer
            
                This is a closure that creates a new connective layer and animates it.
            """
            self.anim_count += 1
            if self.anim_count == 1:
                if not self.before_layer is None and not self.after_layer is None:
                    print(neural_network)
                    new_connective = get_connective_layer(self.before_layer, self.after_layer)
                    before_layer_index = neural_network.all_layers.index_of(self.before_layer) + 1
                    neural_network.all_layers.insert(before_layer_index, new_connective)
                    print(neural_network)

        update_func_anim = UpdateFromFunc(self.neural_network, create_new_connective)

        return update_func_anim

class InsertLayer(AnimationGroup):
    """Animation for inserting layer at given index"""

    def __init__(self, layer, index, neural_network):
        self.layer = layer
        self.index = index
        self.neural_network = neural_network
        # Layers before and after
        self.layers_before = self.neural_network.all_layers[:self.index]
        self.layers_after = self.neural_network.all_layers[self.index:]

        remove_connective_layer = self.remove_connective_layer()
        move_layers = self.make_move_layers()
        # create_layer = self.make_create_layer()
        # create_connective_layers = self.make_create_connective_layers()
        animations = [
            remove_connective_layer,
            move_layers,
        #    create_layer, 
        #    create_connective_layers
        ]

        super().__init__(*animations, lag_ratio=1.0)

    def remove_connective_layer(self):
        """Removes the connective layer before the insertion index"""
        # Check if connective layer exists
        if len(self.layers_before) > 0:
            removed_connective = self.layers_before[-1]
            self.neural_network.all_layers.remove(removed_connective)
            # Make remove animation
            remove_animation = FadeOut(removed_connective)
            return remove_animation

        return AnimationGroup()

    def make_move_layers(self):
        """Shifts layers before and after"""
        # Before layer shift
        before_shift_animation = AnimationGroup()
        if len(self.layers_before) > 0:
            before_shift = np.array([-self.layer.width/2, 0, 0])
            # Shift layers before
            before_shift_animation = Group(*self.layers_before).animate.shift(before_shift)
        # After layer shift
        after_shift_animation = AnimationGroup()
        if len(self.layers_after) > 0:
            after_shift = np.array([self.layer.width/2, 0, 0])
            # Shift layers after
            after_shift_animation = Group(*self.layers_after).animate.shift(after_shift)
        # Make animation group
        shift_animations = AnimationGroup(
            before_shift_animation, 
            after_shift_animation
        )

        return shift_animations

    def make_create_layer(self):
        """Animates the creation of the layer"""
        pass

    def make_create_connective_layers(self):
        pass

        
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

        insert_animation = Create(layer)
        animation_group = AnimationGroup(
            shift_animations, 
            insert_animation,
            lag_ratio=1.0
        )

        return animation_group
    
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

    def make_forward_pass_animation(self, run_time=10, passing_flash=True,
                                    **kwargs):
        """Generates an animation for feed forward propagation"""
        all_animations = []
        for layer_index, layer in enumerate(self.input_layers[:-1]):
            layer_forward_pass = layer.make_forward_pass_animation(**kwargs)
            all_animations.append(layer_forward_pass)
            connective_layer = self.connective_layers[layer_index]
            connective_forward_pass = connective_layer.make_forward_pass_animation(**kwargs)
            all_animations.append(connective_forward_pass)
        # Do last layer animation
        last_layer_forward_pass = self.input_layers[-1].make_forward_pass_animation(**kwargs)
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