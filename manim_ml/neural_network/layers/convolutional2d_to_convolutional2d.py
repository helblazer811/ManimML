from cv2 import line
from manim import *
from manim_ml.neural_network.layers.convolutional2d import Convolutional2DLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer


class Convolutional2DToConvolutional2D(ConnectiveLayer):
    """2D Conv to 2d Conv"""

    input_class = Convolutional2DLayer
    output_class = Convolutional2DLayer

    def __init__(
        self,
        input_layer,
        output_layer,
        color=WHITE,
        filter_opacity=0.3,
        line_color=WHITE,
        pulse_color=ORANGE,
        **kwargs
    ):
        super().__init__(
            input_layer,
            output_layer,
            input_class=Convolutional2DLayer,
            output_class=Convolutional2DLayer,
            **kwargs
        )
        self.color = color
        self.filter_color = self.input_layer.filter_color
        self.filter_width = self.input_layer.filter_width
        self.filter_height = self.input_layer.filter_height
        self.feature_map_width = self.input_layer.feature_map_width
        self.feature_map_height = self.input_layer.feature_map_height
        self.cell_width = self.input_layer.cell_width
        self.stride = self.input_layer.stride
        self.filter_opacity = filter_opacity
        self.line_color = line_color
        self.pulse_color = pulse_color

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return AnimationGroup()

    def make_filter(self):
        """Make filter object"""
        # Make opaque rectangle
        filter = Rectangle(
            color=self.filter_color,
            fill_color=self.filter_color,
            width=self.cell_width * self.filter_width,
            height=self.cell_width * self.filter_height,
            grid_xstep=self.cell_width,
            grid_ystep=self.cell_width,
            fill_opacity=self.filter_opacity,
        )
        # Move filter to top left of feature map
        filter.move_to(
            self.input_layer.feature_map.get_corner(LEFT + UP), aligned_edge=LEFT + UP
        )

        return filter

    def make_output_node(self):
        """Put output node in top left corner of output feature map"""
        # Make opaque rectangle
        filter = Rectangle(
            color=self.filter_color,
            fill_color=self.filter_color,
            width=self.cell_width,
            height=self.cell_width,
            fill_opacity=self.filter_opacity,
        )
        # Move filter to top left of feature map
        filter.move_to(
            self.output_layer.feature_map.get_corner(LEFT + UP), aligned_edge=LEFT + UP
        )

        return filter

    def make_filter_propagation_animation(self):
        """Make filter propagation animation"""
        lines_copy = self.filter_lines.copy().set_color(ORANGE)
        animation_group = AnimationGroup(
            Create(lines_copy, lag_ratio=0.0),
            # FadeOut(self.filter_lines),
            FadeOut(lines_copy),
            lag_ratio=1.0,
        )

        return animation_group

    def make_filter_lines(self):
        """Lines connecting input filter with output node"""
        filter_lines = []
        corner_directions = [LEFT + UP, RIGHT + UP, RIGHT + DOWN, LEFT + DOWN]
        for corner_direction in corner_directions:
            filter_corner = self.filter.get_corner(corner_direction)
            output_corner = self.output_node.get_corner(corner_direction)
            line = Line(filter_corner, output_corner, stroke_color=self.line_color)
            filter_lines.append(line)

        filter_lines = VGroup(*filter_lines)
        filter_lines.set_z_index(5)
        # Make updater that links the lines to the filter and output node
        def filter_updater(filter_lines):
            for corner_index, corner_direction in enumerate(corner_directions):
                line = filter_lines[corner_index]
                filter_corner = self.filter.get_corner(corner_direction)
                output_corner = self.output_node.get_corner(corner_direction)
                # line._set_start_and_end_attrs(filter_corner, output_corner)
                # line.put_start_and_end_on(filter_corner, output_corner)
                line.set_points_by_ends(filter_corner, output_corner)
                # line._set_start_and_end_attrs(filter_corner, output_corner)
                # line.set_points([filter_corner, output_corner])

        filter_lines.add_updater(filter_updater)

        return filter_lines

    def make_assets(self):
        """Make all of the assets"""
        # Make the filter
        self.filter = self.make_filter()
        self.add(self.filter)
        # Make output node
        self.output_node = self.make_output_node()
        self.add(self.output_node)
        # Make filter lines
        self.filter_lines = self.make_filter_lines()
        self.add(self.filter_lines)

        super().set_z_index(5)

    def make_forward_pass_animation(self, layer_args={}, run_time=1.5, **kwargs):
        """Forward pass animation from conv2d to conv2d"""
        # Make assets
        self.make_assets()
        self.lines_copies = VGroup()
        self.add(self.lines_copies)
        # Make the animations
        animations = []
        # Create filter animation
        animations.append(
            AnimationGroup(
                Create(self.filter),
                Create(self.output_node),
                #         Create(self.filter_lines)
            )
        )
        # Make scan filter animation
        num_y_moves = (
            int((self.feature_map_height - self.filter_height) / self.stride) + 1
        )
        num_x_moves = int((self.feature_map_width - self.filter_width) / self.stride)
        for y_location in range(num_y_moves):
            if y_location > 0:
                # Shift filter back to start and down
                shift_animation = ApplyMethod(
                    self.filter.shift,
                    np.array(
                        [
                            -self.cell_width
                            * (self.feature_map_width - self.filter_width),
                            -self.stride * self.cell_width,
                            0,
                        ]
                    ),
                )
                # Shift output node
                shift_output_node = ApplyMethod(
                    self.output_node.shift,
                    np.array(
                        [
                            -(self.output_layer.feature_map_width - 1)
                            * self.cell_width,
                            -self.cell_width,
                            0,
                        ]
                    ),
                )
                # Make animation group
                animation_group = AnimationGroup(
                    shift_animation,
                    shift_output_node,
                )
                animations.append(animation_group)
                # Make filter passing flash
                # animation = self.make_filter_propagation_animation()
                animations.append(Create(self.filter_lines, lag_ratio=0.0))
                # animations.append(animation)

            for x_location in range(num_x_moves):
                # Shift filter right
                shift_animation = ApplyMethod(
                    self.filter.shift, np.array([self.stride * self.cell_width, 0, 0])
                )
                # Shift output node
                shift_output_node = ApplyMethod(
                    self.output_node.shift, np.array([self.cell_width, 0, 0])
                )
                # Make animation group
                animation_group = AnimationGroup(
                    shift_animation,
                    shift_output_node,
                )
                animations.append(animation_group)
                # Make filter passing flash
                old_z_index = self.filter_lines.z_index
                lines_copy = (
                    self.filter_lines.copy()
                    .set_color(ORANGE)
                    .set_z_index(old_z_index + 1)
                )
                # self.add(lines_copy)
                # self.lines_copies.add(lines_copy)
                animations.append(Create(self.filter_lines, lag_ratio=0.0))
                # animations.append(FadeOut(self.filter_lines))
                # animation = self.make_filter_propagation_animation()
                # animations.append(animation)
                # animations.append(Create(self.filter_lines, lag_ratio=1.0))
                # animations.append(FadeOut(self.filter_lines))
        # Fade out
        animations.append(
            AnimationGroup(
                FadeOut(self.filter),
                FadeOut(self.output_node),
                FadeOut(self.filter_lines),
            )
        )
        # Make animation group
        animation_group = Succession(*animations, lag_ratio=1.0)
        return animation_group

    def set_z_index(self, z_index, family=False):
        """Override set_z_index"""
        super().set_z_index(4)

    def scale(self, scale_factor, **kwargs):
        self.cell_width *= scale_factor
        super().scale(scale_factor, **kwargs)
