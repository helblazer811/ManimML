"""
    Transformations for manipulating a neural network object. 
"""
from manim import *
from manim_ml.neural_network.layers.util import get_connective_layer


class RemoveLayer(AnimationGroup):
    """
    Animation for removing a layer from a neural network.

    Note: I needed to do something strange for creating the new connective layer.
    The issue with creating it initially is that the positions of the sides of the
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
        animations_list = [remove_animations, move_animations, new_connective_animation]

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
            remove_layer_animation, remove_connective_animation
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
            fade_out_after_connective, fade_out_before_connective
        )

        return remove_connective_group

    def make_move_animation(self):
        """Collapses layers"""
        # Animate the movements
        move_before_layers = AnimationGroup()
        shift_right_amount = None
        if not self.before_layer is None:
            # Compute shift amount
            layer_dist = np.abs(
                self.layer.get_center() - self.before_layer.get_right()
            )[0]
            shift_right_amount = np.array([layer_dist - self.layer_spacing / 2, 0, 0])
            # Shift all layers before forward
            before_layer_index = self.neural_network.all_layers.index_of(
                self.before_layer
            )
            layers_before = Group(
                *self.neural_network.all_layers[: before_layer_index + 1]
            )
            move_before_layers = layers_before.animate.shift(shift_right_amount)
        move_after_layers = AnimationGroup()
        shift_left_amount = None
        if not self.after_layer is None:
            layer_dist = np.abs(self.after_layer.get_left() - self.layer.get_center())[
                0
            ]
            shift_left_amount = np.array([-layer_dist + self.layer_spacing / 2, 0, 0])
            # Shift all layers after backward
            after_layer_index = self.neural_network.all_layers.index_of(
                self.after_layer
            )
            layers_after = Group(*self.neural_network.all_layers[after_layer_index:])
            move_after_layers = layers_after.animate.shift(shift_left_amount)
        # Group the move animations
        move_group = AnimationGroup(move_before_layers, move_after_layers)

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
                    new_connective_class = get_connective_layer(
                        self.before_layer, self.after_layer
                    )
                    before_layer_index = (
                        neural_network.all_layers.index_of(self.before_layer) + 1
                    )
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
        # Check valid index
        assert index < len(self.neural_network.all_layers)
        # Layers before and after
        self.layers_before = self.neural_network.all_layers[: self.index]
        self.layers_after = self.neural_network.all_layers[self.index :]
        # Get the non-connective layer before and after
        if len(self.layers_before) > 0:
            self.layer_before = self.layers_before[-2]
        if len(self.layers_after) > 0:
            self.layer_after = self.layers_after[0]
        # Move layer
        if not self.layer_after is None:
            self.layer.move_to(self.layer_after)
        # Make animations
        (
            self.old_connective_layer,
            remove_connective_layer,
        ) = self.remove_connective_layer_animation()
        move_layers = self.make_move_layers_animation()
        create_layer = self.make_create_layer_animation()
        # create_connective_layers = self.make_create_connective_layers()
        animations = [
            remove_connective_layer,
            move_layers,
            create_layer,
            #    create_connective_layers
        ]

        super().__init__(*animations, lag_ratio=1.0)

    def get_connective_layer_widths(self):
        """Gets the widths of the connective layers"""
        # Make the layers
        before_connective = None
        after_connective = None
        # Get the connective layer objects
        if len(self.layers_before) > 0:
            before_connective = get_connective_layer(self.layer_before, self.layer)
        if len(self.layers_after) > 0:
            after_connective = get_connective_layer(self.layer, self.layer_after)
        # Compute the widths
        before_connective_width = 0
        if not before_connective is None:
            before_connective_width = before_connective.width
        after_connective_width = 0
        if not after_connective is None:
            after_connective_width = after_connective.width
        return before_connective_width, after_connective_width

    def remove_connective_layer_animation(self):
        """Removes the connective layer before the insertion index"""
        # Check if connective layer before exists
        if len(self.layers_before) > 0:
            removed_connective = self.layers_before[-1]
            self.layers_before.remove(removed_connective)
            self.neural_network.all_layers.remove(removed_connective)
            # Make remove animation
            remove_animation = FadeOut(removed_connective)
            return removed_connective, remove_animation

        return None, AnimationGroup()

    def make_move_layers_animation(self):
        """Shifts layers before and after"""
        (
            before_connective_width,
            after_connective_width,
        ) = self.get_connective_layer_widths()
        old_connective_width = 0
        if not self.old_connective_layer is None:
            old_connective_width = self.old_connective_layer.width
        # Before layer shift
        before_shift_animation = AnimationGroup()
        if len(self.layers_before) > 0:
            before_shift = np.array(
                [
                    -self.layer.width / 2
                    - before_connective_width
                    + old_connective_width,
                    0,
                    0,
                ]
            )
            # Shift layers before
            before_shift_animation = Group(*self.layers_before).animate.shift(
                before_shift
            )
        # After layer shift
        after_shift_animation = AnimationGroup()
        if len(self.layers_after) > 0:
            after_shift = np.array(
                [self.layer.width / 2 + after_connective_width, 0, 0]
            )
            # Shift layers after
            after_shift_animation = Group(*self.layers_after).animate.shift(after_shift)
        # Make animation group
        shift_animations = AnimationGroup(before_shift_animation, after_shift_animation)

        return shift_animations

    def make_create_layer_animation(self):
        """Animates the creation of the layer"""
        return Create(self.layer)

    def make_create_connective_layers_animation(
        self, before_connective, after_connective
    ):
        """Create connective layers"""
        # Make the layers
        before_connective = None
        after_connective = None
        # Get the connective layer objects
        if len(self.layers_before) > 0:
            before_connective = get_connective_layer(self.layers_before[-1], self.layer)
        if len(self.layers_after) > 0:
            after_connective = get_connective_layer(self.layers_after[0], self.layer)
        # Insert the layers
        # Make the animation
        animation_group = AnimationGroup(
            Create(before_connective), Create(after_connective)
        )

        return animation_group
