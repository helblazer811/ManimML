import numpy as np

from manim import *
from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer, ThreeDLayer
from manim_ml.gridded_rectangle import GriddedRectangle

from manim.utils.space_ops import rotation_matrix


def get_rotated_shift_vectors(input_layer, normalized=False):
    """Rotates the shift vectors"""
    # Make base shift vectors
    right_shift = np.array([input_layer.cell_width, 0, 0])
    down_shift = np.array([0, -input_layer.cell_width, 0])
    # Make rotation matrix
    rot_mat = rotation_matrix(ThreeDLayer.rotation_angle, ThreeDLayer.rotation_axis)
    # Rotate the vectors
    right_shift = np.dot(right_shift, rot_mat.T)
    down_shift = np.dot(down_shift, rot_mat.T)
    # Normalize the vectors
    if normalized:
        right_shift = right_shift / np.linalg.norm(right_shift)
        down_shift = down_shift / np.linalg.norm(down_shift)

    return right_shift, down_shift


class Filters(VGroup):
    """Group for showing a collection of filters connecting two layers"""

    def __init__(
        self,
        input_layer,
        output_layer,
        line_color=ORANGE,
        cell_width=1.0,
        stroke_width=2.0,
        show_grid_lines=False,
        output_feature_map_to_connect=None,  # None means all at once
    ):
        super().__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.line_color = line_color
        self.cell_width = cell_width
        self.stroke_width = stroke_width
        self.show_grid_lines = show_grid_lines
        self.output_feature_map_to_connect = output_feature_map_to_connect
        # Make the filter
        self.input_rectangles = self.make_input_feature_map_rectangles()
        # self.input_rectangles.set_z_index(5)
        # self.add(self.input_rectangles)
        self.output_rectangles = self.make_output_feature_map_rectangles()
        # self.output_rectangles.set_z_index(5)
        # self.add(self.output_rectangles)
        self.connective_lines = self.make_connective_lines()
        # self.connective_lines.set_z_index(5)
        # self.add(self.connective_lines)

    def make_input_feature_map_rectangles(self):
        rectangles = []
        rectangle_width = (
            self.output_layer.filter_size[0] * self.output_layer.cell_width
        )
        rectangle_height = (
            self.output_layer.filter_size[1] * self.output_layer.cell_width
        )
        filter_color = self.output_layer.filter_color

        for index, feature_map in enumerate(self.input_layer.feature_maps):
            rectangle = GriddedRectangle(
                width=rectangle_width,
                height=rectangle_height,
                fill_color=filter_color,
                stroke_color=filter_color,
                fill_opacity=0.2,
                stroke_width=self.stroke_width,
                grid_xstep=self.cell_width,
                grid_ystep=self.cell_width,
                grid_stroke_width=self.stroke_width / 2,
                grid_stroke_color=filter_color,
                show_grid_lines=self.show_grid_lines,
            )
            # normal_vector = rectangle.get_normal_vector()
            rectangle.rotate(
                ThreeDLayer.rotation_angle,
                about_point=rectangle.get_center(),
                axis=ThreeDLayer.rotation_axis,
            )
            # Move the rectangle to the corner of the feature map
            rectangle.next_to(
                feature_map.get_corners_dict()["top_left"],
                submobject_to_align=rectangle.get_corners_dict()["top_left"],
                buff=0.0
                # aligned_edge=feature_map.get_corners_dict()["top_left"].get_center()
            )
            rectangle.set_z_index(5)

            rectangles.append(rectangle)

        feature_map_rectangles = VGroup(*rectangles)

        return feature_map_rectangles

    def make_output_feature_map_rectangles(self):
        rectangles = []

        rectangle_width = self.output_layer.cell_width
        rectangle_height = self.output_layer.cell_width
        filter_color = self.output_layer.filter_color

        for index, feature_map in enumerate(self.output_layer.feature_maps):
            # Make sure current feature map is the right filter
            if not self.output_feature_map_to_connect is None:
                if index != self.output_feature_map_to_connect:
                    continue
            # Make the rectangle
            rectangle = GriddedRectangle(
                width=rectangle_width,
                height=rectangle_height,
                fill_color=filter_color,
                fill_opacity=0.2,
                stroke_color=filter_color,
                stroke_width=self.stroke_width,
                grid_xstep=self.cell_width,
                grid_ystep=self.cell_width,
                grid_stroke_width=self.stroke_width / 2,
                grid_stroke_color=filter_color,
                show_grid_lines=self.show_grid_lines,
            )
            # Rotate the rectangle
            rectangle.rotate(
                ThreeDLayer.rotation_angle,
                about_point=rectangle.get_center(),
                axis=ThreeDLayer.rotation_axis,
            )
            # Move the rectangle to the corner location
            rectangle.next_to(
                feature_map.get_corners_dict()["top_left"],
                submobject_to_align=rectangle.get_corners_dict()["top_left"],
                buff=0.0
                # aligned_edge=feature_map.get_corners_dict()["top_left"].get_center()
            )
            rectangles.append(rectangle)

        feature_map_rectangles = VGroup(*rectangles)

        return feature_map_rectangles

    def make_connective_lines(self):
        """Lines connecting input filter with output node"""

        corner_names = ["top_left", "bottom_left", "top_right", "bottom_right"]

        def make_input_connective_lines():
            """Makes connective lines between the corners of the input filters"""
            first_input_rectangle = self.input_rectangles[0]
            last_input_rectangle = self.input_rectangles[-1]
            # Get the corner dots for each rectangle
            first_input_corners = first_input_rectangle.get_corners_dict()
            last_input_corners = last_input_rectangle.get_corners_dict()
            # Iterate through each corner and make the lines
            lines = []
            for corner_name in corner_names:
                line = Line(
                    first_input_corners[corner_name].get_center(),
                    last_input_corners[corner_name].get_center(),
                    color=self.line_color,
                    stroke_width=self.stroke_width,
                )
                lines.append(line)

            return VGroup(*lines)

        def make_output_connective_lines():
            """Makes connective lines between the corners of the output filters"""
            first_output_rectangle = self.output_rectangles[0]
            last_output_rectangle = self.output_rectangles[-1]
            # Get the corner dots for each rectangle
            first_output_corners = first_output_rectangle.get_corners_dict()
            last_output_corners = last_output_rectangle.get_corners_dict()
            # Iterate through each corner and make the lines
            lines = []
            for corner_name in corner_names:
                line = Line(
                    first_output_corners[corner_name].get_center(),
                    last_output_corners[corner_name].get_center(),
                    color=self.line_color,
                    stroke_width=self.stroke_width,
                )
                lines.append(line)

            return VGroup(*lines)

        def make_input_to_output_connective_lines():
            """Make connective lines between last input filter and first output filter"""
            # Choose the correct feature map to link to
            input_rectangle = self.input_rectangles[-1]
            output_rectangle = self.output_rectangles[0]
            # Get the corner dots for each rectangle
            input_corners = input_rectangle.get_corners_dict()
            output_corners = output_rectangle.get_corners_dict()
            # Iterate through each corner and make the lines
            lines = []
            for corner_name in corner_names:
                line = Line(
                    input_corners[corner_name].get_center(),
                    output_corners[corner_name].get_center(),
                    color=self.line_color,
                    stroke_width=self.stroke_width,
                )
                lines.append(line)

            return VGroup(*lines)

        input_lines = make_input_connective_lines()
        output_lines = make_output_connective_lines()
        input_output_lines = make_input_to_output_connective_lines()

        connective_lines = VGroup(*input_lines, *output_lines, *input_output_lines)

        return connective_lines

    @override_animation(Create)
    def _create_override(self, **kwargs):
        """
        NOTE This create override animation
        is a workaround to make sure that the filter
        does not show up in the scene before the create animation.

        Without this override the filters were shown at the beginning
        of the neural network forward pass animation
        instead of just when the filters were supposed to appear.
        I think this is a bug with Succession in the core
        Manim Community Library.

        TODO Fix this
        """

        def add_content(object):
            object.add(self.input_rectangles)
            object.add(self.connective_lines)
            object.add(self.output_rectangles)

            return object

        return ApplyFunction(add_content, self)
        return AnimationGroup(
            Create(self.input_rectangles),
            Create(self.connective_lines),
            Create(self.output_rectangles),
            lag_ratio=0.0,
        )

    def make_pulse_animation(self, shift_amount):
        """Make animation of the filter pulsing"""
        passing_flash = ShowPassingFlash(
            self.connective_lines.shift(shift_amount).set_stroke_width(
                self.stroke_width * 1.5
            ),
            time_width=0.2,
            color=RED,
            z_index=10,
        )

        return passing_flash


class Convolutional2DToConvolutional2D(ConnectiveLayer, ThreeDLayer):
    """Feed Forward to Embedding Layer"""

    input_class = Convolutional2DLayer
    output_class = Convolutional2DLayer

    def __init__(
        self,
        input_layer: Convolutional2DLayer,
        output_layer: Convolutional2DLayer,
        color=ORANGE,
        filter_opacity=0.3,
        line_color=ORANGE,
        pulse_color=ORANGE,
        cell_width=0.2,
        show_grid_lines=True,
        highlight_color=ORANGE,
        **kwargs,
    ):
        super().__init__(
            input_layer,
            output_layer,
            **kwargs,
        )
        self.color = color
        self.filter_color = self.output_layer.filter_color
        self.filter_size = self.output_layer.filter_size
        self.feature_map_size = self.input_layer.feature_map_size
        self.num_input_feature_maps = self.input_layer.num_feature_maps
        self.num_output_feature_maps = self.output_layer.num_feature_maps
        self.cell_width = self.output_layer.cell_width
        self.stride = self.output_layer.stride
        self.filter_opacity = filter_opacity
        self.cell_width = cell_width
        self.line_color = line_color
        self.pulse_color = pulse_color
        self.show_grid_lines = show_grid_lines
        self.highlight_color = highlight_color

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs,
    ):
        return super().construct_layer(input_layer, output_layer, **kwargs)

    def animate_filters_all_at_once(self, filters):
        """Animates each of the filters all at once"""
        animations = []
        # Make filters
        filters = Filters(
            self.input_layer,
            self.output_layer,
            line_color=self.color,
            cell_width=self.cell_width,
            show_grid_lines=self.show_grid_lines,
            output_feature_map_to_connect=None,  # None means all at once
        )
        animations.append(Create(filters))
        # Get the rotated shift vectors
        right_shift, down_shift = get_rotated_shift_vectors(self.input_layer)
        left_shift = -1 * right_shift
        # Make the animation
        num_y_moves = int(
            (self.feature_map_size[1] - self.filter_size[1]) / self.stride
        )
        num_x_moves = int(
            (self.feature_map_size[0] - self.filter_size[0]) / self.stride
        )
        for y_move in range(num_y_moves):
            # Go right num_x_moves
            for x_move in range(num_x_moves):
                # Shift right
                shift_animation = ApplyMethod(filters.shift, self.stride * right_shift)
                # shift_animation = self.animate.shift(right_shift)
                animations.append(shift_animation)
            # Go back left num_x_moves and down one
            shift_amount = (
                self.stride * num_x_moves * left_shift + self.stride * down_shift
            )
            # Make the animation
            shift_animation = ApplyMethod(filters.shift, shift_amount)
            animations.append(shift_animation)
        # Do last row move right
        for x_move in range(num_x_moves):
            # Shift right
            shift_animation = ApplyMethod(filters.shift, self.stride * right_shift)
            # shift_animation = self.animate.shift(right_shift)
            animations.append(shift_animation)
        # Remove the filters
        animations.append(FadeOut(filters))
        return Succession(*animations, lag_ratio=1.0)

    def animate_filters_one_at_a_time(self, highlight_active_feature_map=True):
        """Animates each of the filters one at a time"""
        animations = []
        output_feature_maps = self.output_layer.feature_maps
        for feature_map_index in range(len(output_feature_maps)):
            # Make filters
            filters = Filters(
                self.input_layer,
                self.output_layer,
                line_color=self.color,
                cell_width=self.cell_width,
                show_grid_lines=self.show_grid_lines,
                output_feature_map_to_connect=feature_map_index,  # None means all at once
            )
            animations.append(Create(filters))
            # Highlight the feature map
            if highlight_active_feature_map:
                feature_map = output_feature_maps[feature_map_index]
                original_feature_map_color = feature_map.color
                # Change the output feature map colors
                change_color_animations = []
                change_color_animations.append(
                    ApplyMethod(feature_map.set_color, self.highlight_color)
                )
                # Change the input feature map colors
                input_feature_maps = self.input_layer.feature_maps
                for input_feature_map in input_feature_maps:
                    change_color_animations.append(
                        ApplyMethod(input_feature_map.set_color, self.highlight_color)
                    )
                # Combine the animations
                animations.append(
                    AnimationGroup(*change_color_animations, lag_ratio=0.0)
                )
            # Get the rotated shift vectors
            right_shift, down_shift = get_rotated_shift_vectors(self.input_layer)
            left_shift = -1 * right_shift
            # Make the animation
            num_y_moves = int(
                (self.feature_map_size[1] - self.filter_size[1]) / self.stride
            )
            num_x_moves = int(
                (self.feature_map_size[0] - self.filter_size[0]) / self.stride
            )
            for y_move in range(num_y_moves):
                # Go right num_x_moves
                for x_move in range(num_x_moves):
                    # Make a pulse animation for the corners
                    """
                    pulse_animation = filters.make_pulse_animation(
                        shift_amount=shift_amount
                    )
                    animations.append(pulse_animation)
                    """
                    z_index_animation = ApplyMethod(filters.set_z_index, 5)
                    animations.append(z_index_animation)
                    # Shift right
                    shift_animation = ApplyMethod(
                        filters.shift, self.stride * right_shift
                    )
                    # shift_animation = self.animate.shift(right_shift)
                    animations.append(shift_animation)

                # Go back left num_x_moves and down one
                shift_amount = (
                    self.stride * num_x_moves * left_shift + self.stride * down_shift
                )
                # Make the animation
                shift_animation = ApplyMethod(filters.shift, shift_amount)
                animations.append(shift_animation)
            # Do last row move right
            for x_move in range(num_x_moves):
                # Shift right
                shift_animation = ApplyMethod(filters.shift, self.stride * right_shift)
                # shift_animation = self.animate.shift(right_shift)
                animations.append(shift_animation)
            # Remove the filters
            animations.append(FadeOut(filters))
            # Un-highlight the feature map
            if highlight_active_feature_map:
                feature_map = output_feature_maps[feature_map_index]
                # Change the output feature map colors
                change_color_animations = []
                change_color_animations.append(
                    ApplyMethod(feature_map.set_color, original_feature_map_color)
                )
                # Change the input feature map colors
                input_feature_maps = self.input_layer.feature_maps
                for input_feature_map in input_feature_maps:
                    change_color_animations.append(
                        ApplyMethod(
                            input_feature_map.set_color, original_feature_map_color
                        )
                    )
                # Combine the animations
                animations.append(
                    AnimationGroup(*change_color_animations, lag_ratio=0.0)
                )

        return Succession(*animations, lag_ratio=1.0)

    def make_forward_pass_animation(
        self,
        layer_args={},
        all_filters_at_once=False,
        highlight_active_feature_map=True,
        run_time=10.5,
        **kwargs,
    ):
        """Forward pass animation from conv2d to conv2d"""
        print(f"All filters at once: {all_filters_at_once}")
        # Make filter shifting animations
        if all_filters_at_once:
            return self.animate_filters_all_at_once()
        else:
            return self.animate_filters_one_at_a_time(
                highlight_active_feature_map=highlight_active_feature_map
            )

    def scale(self, scale_factor, **kwargs):
        self.cell_width *= scale_factor
        super().scale(scale_factor, **kwargs)

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return Succession()
