from manim import *
from manim_ml.neural_network.layers.convolutional3d import Convolutional3DLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer, ThreeDLayer
from manim_ml.gridded_rectangle import GriddedRectangle

from manim.utils.space_ops import rotation_matrix

class Filters(VGroup):
    """Group for showing a collection of filters connecting two layers"""

    def __init__(
            self,
            input_layer, 
            output_layer, 
            line_color=ORANGE, 
            stroke_width=2.0,
        ):
        super().__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.line_color = line_color
        self.stroke_width = stroke_width
        # Make the filter
        self.input_rectangles = self.make_input_feature_map_rectangles()
        # self.add(self.input_rectangles)
        self.output_rectangles = self.make_output_feature_map_rectangles()
        # self.add(self.output_rectangles)
        self.connective_lines = self.make_connective_lines()
        # self.add(self.connective_lines)

    def make_input_feature_map_rectangles(self):
        rectangles = []

        rectangle_width = self.input_layer.filter_width * self.input_layer.cell_width
        rectangle_height = self.input_layer.filter_height * self.input_layer.cell_width
        filter_color = self.input_layer.filter_color

        for index, feature_map in enumerate(self.input_layer.feature_maps):
            rectangle = GriddedRectangle(
                width=rectangle_width, 
                height=rectangle_height,
                fill_color=filter_color,
                stroke_color=filter_color,
                fill_opacity=0.2,
                z_index=2,
                stroke_width=self.stroke_width,
            )
            rectangle.rotate(
                ThreeDLayer.three_d_x_rotation, 
                about_point=rectangle.get_center(), 
                axis=[1, 0, 0]
            )
            rectangle.rotate(
                ThreeDLayer.three_d_y_rotation, 
                about_point=rectangle.get_center(), 
                axis=[0, 1, 0]
            )
            # Move the rectangle to the corner of the feature map
            rectangle.move_to(
                feature_map,
                aligned_edge=np.array([-1, 1, 0])
            )

            rectangles.append(rectangle)
            
        feature_map_rectangles = VGroup(*rectangles)

        return feature_map_rectangles

    def make_output_feature_map_rectangles(self):
        rectangles = []

        rectangle_width = self.output_layer.cell_width
        rectangle_height = self.output_layer.cell_width
        filter_color = self.output_layer.filter_color

        for index, feature_map in enumerate(self.output_layer.feature_maps):
            rectangle = GriddedRectangle(
                width=rectangle_width, 
                height=rectangle_height,
                fill_color=filter_color,
                stroke_color=filter_color,
                fill_opacity=0.2,
                stroke_width=self.stroke_width,
                z_index=2,
            )
            # Center on feature map
            # rectangle.move_to(feature_map.get_center())
            # Rotate the rectangle
            rectangle.rotate(
                ThreeDLayer.three_d_x_rotation, 
                about_point=rectangle.get_center(), 
                axis=[1, 0, 0]
            )
            rectangle.rotate(
                ThreeDLayer.three_d_y_rotation, 
                about_point=rectangle.get_center(), 
                axis=[0, 1, 0]
            )
            # Move the rectangle to the corner location
            rectangle.move_to(
                feature_map,
                aligned_edge=np.array([-1, 1, 0])
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
                    stroke_width=self.stroke_width
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
                    stroke_width=self.stroke_width
                )
                lines.append(line)

            return VGroup(*lines)

        def make_input_to_output_connective_lines():
            """Make connective lines between last input filter and first output filter"""
            last_input_rectangle = self.input_rectangles[-1]
            first_output_rectangle = self.output_rectangles[0]
            # Get the corner dots for each rectangle
            last_input_corners = last_input_rectangle.get_corners_dict()
            first_output_corners = first_output_rectangle.get_corners_dict()
            # Iterate through each corner and make the lines
            lines = []
            for corner_name in corner_names:
                line = Line(
                    last_input_corners[corner_name].get_center(),
                    first_output_corners[corner_name].get_center(),
                    color=self.line_color,
                    stroke_width=self.stroke_width
                )
                lines.append(line)

            return VGroup(*lines)   
            
        input_lines = make_input_connective_lines()
        output_lines = make_output_connective_lines()
        input_output_lines = make_input_to_output_connective_lines()

        connective_lines = VGroup(
            *input_lines, 
            *output_lines, 
            *input_output_lines
        )
 
        return connective_lines

    @override_animation(Create)
    def _create_override(self, **kwargs):
        """
            NOTE This create override animation
            is a workaround to make sure that the filter
            does not show up in the scene before the create animation. 

            Without this override the filters were shown at the beginning
            of the neural network forward pass animimation 
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

        return ApplyFunction(
            add_content,
            self
        )


class Convolutional3DToConvolutional3D(ConnectiveLayer, ThreeDLayer):
    """Feed Forward to Embedding Layer"""
    input_class = Convolutional3DLayer
    output_class = Convolutional3DLayer

    def __init__(self, input_layer: Convolutional3DLayer, output_layer: Convolutional3DLayer, 
                color=WHITE, filter_opacity=0.3, line_color=WHITE, 
                pulse_color=ORANGE, **kwargs):
        super().__init__(input_layer, output_layer, input_class=Convolutional3DLayer, 
                output_class=Convolutional3DLayer, **kwargs)
        self.color = color
        self.filter_color = self.input_layer.filter_color
        self.filter_width = self.input_layer.filter_width
        self.filter_height = self.input_layer.filter_height
        self.feature_map_width = self.input_layer.feature_map_width
        self.feature_map_height = self.input_layer.feature_map_height
        self.num_input_feature_maps = self.input_layer.num_feature_maps
        self.num_output_feature_maps = self.output_layer.num_feature_maps
        self.cell_width = self.input_layer.cell_width
        self.stride = self.input_layer.stride
        self.filter_opacity = filter_opacity
        self.line_color = line_color
        self.pulse_color = pulse_color

    def make_filter_propagation_animation(self):
        """Make filter propagation animation"""
        # TODO implement this
        raise NotImplementedError()
        # Deprecated code
        old_z_index = self.filter_lines.z_index
        lines_copy = self.filter_lines.copy().set_color(ORANGE).set_z_index(old_z_index + 1)
        animation_group = AnimationGroup(
            Create(lines_copy, lag_ratio=0.0),
            # FadeOut(self.filter_lines),
            FadeOut(lines_copy),
            lag_ratio=1.0
        )

        return animation_group

    def get_rotated_shift_vectors(self):
        """
            Rotates the shift vectors
        """
        x_rot_mat = rotation_matrix(
            ThreeDLayer.three_d_x_rotation, 
            [1, 0, 0]
        )
        y_rot_mat = rotation_matrix(
            ThreeDLayer.three_d_y_rotation, 
            [0, 1, 0]
        )
        # Make base shift vectors
        right_shift = np.array([self.input_layer.cell_width, 0, 0])
        down_shift = np.array([0, -self.input_layer.cell_width, 0])
        # Rotate the vectors
        right_shift = np.dot(right_shift, x_rot_mat.T)
        right_shift = np.dot(right_shift, y_rot_mat.T)
        down_shift = np.dot(down_shift, x_rot_mat.T)
        down_shift = np.dot(down_shift, y_rot_mat.T)

        return right_shift, down_shift

    def make_forward_pass_animation(self, layer_args={}, 
            all_filters_at_once=False, run_time=10.5, **kwargs):
        """Forward pass animation from conv2d to conv2d"""
        animations = []
        # Make filters
        filters = Filters(self.input_layer, self.output_layer)
        filters.set_z_index(self.input_layer.feature_maps[0].get_z_index() + 1)
        # self.add(filters)
        animations.append(
            Create(filters)
        )
        # Get shift vectors
        right_shift, down_shift = self.get_rotated_shift_vectors()
        left_shift = -1 * right_shift
        # filters.rotate(ThreeDLayer.three_d_theta, axis=[0, 0, 1])
        # filters.rotate(ThreeDLayer.three_d_phi, axis=-filters.get_center())
        # Make animations for creating the filters, output_nodes, and filter_lines
        # TODO decide if I want to create the filters at the start of a conv 
        # animation or have them there by default
        # Rotate the base shift vectors
        # Make filter shifting animations
        num_y_moves = int((self.feature_map_height - self.filter_height) / self.stride)
        num_x_moves = int((self.feature_map_width - self.filter_width) / self.stride)
        for y_move in range(num_y_moves):
            # Go right num_x_moves
            for x_move in range(num_x_moves):
                # Shift right
                shift_animation = ApplyMethod(
                    filters.shift,
                    self.stride * right_shift
                )
                # shift_animation = self.animate.shift(right_shift)
                animations.append(shift_animation)
            
            # Go back left num_x_moves and down one
            shift_amount = self.stride * num_x_moves * left_shift + self.stride * down_shift
            # Make the animation
            shift_animation = ApplyMethod(
                filters.shift,
                shift_amount  
            )
            animations.append(shift_animation)
        # Do last row move right
        for x_move in range(num_x_moves):
            # Shift right
            shift_animation = ApplyMethod(
                filters.shift,
                self.stride * right_shift
            )
            # shift_animation = self.animate.shift(right_shift)
            animations.append(shift_animation)
        # Remove the filters
        animations.append(
            FadeOut(filters)
        )
        # Remove filters
        return Succession(
            *animations,
            lag_ratio=1.0
        )

    def set_z_index(self, z_index, family=False):
        """Override set_z_index"""
        super().set_z_index(4)

    def scale(self, scale_factor, **kwargs):
        self.cell_width *= scale_factor
        super().scale(scale_factor, **kwargs)

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return Succession()
