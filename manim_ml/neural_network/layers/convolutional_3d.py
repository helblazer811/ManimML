from manim import *
from manim_ml.neural_network.layers.parent_layers import VGroupNeuralNetworkLayer

class Convolutional3DLayer(VGroupNeuralNetworkLayer):
    """Handles rendering a convolutional layer for a nn"""

    def __init__(self, num_filters, filter_width, filter_height, filter_spacing=0.1, color=BLUE, 
                pulse_color=ORANGE, **kwargs):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_spacing = filter_spacing
        self.color = color
        self.pulse_color = pulse_color

        self._construct_layer(num_filters=self.num_filters, filter_width=self.filter_width, filter_height=self.filter_height)

    def _construct_layer(self, num_filters=5, filter_width=4, filter_height=4):
        """Creates the neural network layer"""
        # Make axes, but hide the lines
        axes = ThreeDAxes(
            tips=False,
            x_length=1,
            y_length=1,
            x_axis_config={
                "include_ticks": False,
                "stroke_width": 0.0
            },
            y_axis_config={
                "include_ticks": False,
                "stroke_width": 0.0
            },
            z_axis_config={
                "include_ticks": False,
                "stroke_width": 0.0 
            }
        )
        self.add(axes)
        # Set the camera angle so that the 
        # self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        # Draw rectangles that are filled in with opacity
        self.rectangles = VGroup()
        for filter_index in range(num_filters):
            rectangle = Rectangle(
                color=self.color, 
                height=filter_height,
                width=filter_width,
                fill_color=self.color,
                fill_opacity=0.2, 
                stroke_color=WHITE,
            )
            rectangle.rotate_about_origin((80 - filter_index*0.5) * DEGREES, np.array([0, 1, 0])) # Rotate about z axis
            rectangle.rotate_about_origin(15 * DEGREES, np.array([1, 0, 0])) # Rotate about x axis
            rectangle.shift(np.array([filter_index*self.filter_spacing, filter_height*0.5, -3]))

            self.rectangles.add(rectangle)
        
        self.add(self.rectangles)

        self.corner_lines = self.make_filter_corner_lines()
        self.add(self.corner_lines)

    def make_filter_corner_lines(self):
        """Make filter corner lines"""
        corner_lines = VGroup()

        first_rectangle = self.rectangles[0]
        last_rectangle = self.rectangles[-1]
        first_vertices = first_rectangle.get_vertices()
        last_vertices = last_rectangle.get_vertices()
        for vertex_index in range(len(first_vertices)):
            # Make a line 
            line = Line(
                start=first_vertices[vertex_index],
                end=last_vertices[vertex_index],
                color=WHITE,
                stroke_opacity=0.0
            )
            corner_lines.add(line)

        return corner_lines

    def make_forward_pass_animation(self, run_time=5, layer_args={}, **kwargs):
        """Convolution forward pass animation"""
        passing_flashes = []
        for line in self.corner_lines:
            pulse = ShowPassingFlash(
                line.copy()
                    .set_color(self.pulse_color)
                    .set_stroke(opacity=1.0), 
                time_width=0.5,
                run_time=run_time,
                rate_func=rate_functions.linear
            )
            passing_flashes.append(pulse)

        per_filter_run_time = run_time / len(self.rectangles)
        filter_flashes = []
        for filter in self.rectangles:
            single_flash = Succession(
                ApplyMethod(filter.set_color, self.pulse_color, run_time=per_filter_run_time/4),
                Wait(per_filter_run_time/2),
                ApplyMethod(filter.set_color, self.color, run_time=per_filter_run_time/4),
                ApplyMethod(filter.set_stroke_color, WHITE, run_time=0.0)
            )
            filter_flashes.append(single_flash)

        filter_flashes = Succession(
            *filter_flashes,
        )
        # Make animation group
        animation_group = AnimationGroup(
            *passing_flashes,
            filter_flashes
        )

        return animation_group

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return FadeIn(self.rectangles)
