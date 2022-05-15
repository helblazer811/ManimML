from cProfile import run
from manim import *
from manim_ml.neural_network.layers.convolutional_3d import Convolutional3DLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer

class Convolutional3DToConvolutional3D(ConnectiveLayer):
    """Feed Forward to Embedding Layer"""
    input_class = Convolutional3DLayer
    output_class = Convolutional3DLayer

    def __init__(self, input_layer, output_layer, color=WHITE, pulse_color=RED,
                **kwargs):
        super().__init__(input_layer, output_layer, input_class=Convolutional3DLayer, output_class=Convolutional3DLayer,
                        **kwargs)
        self.color = color
        self.pulse_color = pulse_color

        self.lines = self.make_lines()
        self.add(self.lines)

    def make_lines(self):
        """Make lines connecting the input and output layers"""
        lines = VGroup()
        # Get the first and last rectangle
        input_rectangle = self.input_layer.rectangles[-1]
        output_rectangle = self.output_layer.rectangles[0]
        input_vertices = input_rectangle.get_vertices()
        output_vertices = output_rectangle.get_vertices()
        # Go through each vertex
        for vertex_index in range(len(input_vertices)):
            # Make a line 
            line = Line(
                start=input_vertices[vertex_index],
                end=output_vertices[vertex_index],
                color=self.color,
                stroke_opacity=0.0
            )
            lines.add(line)

        return lines

    def make_forward_pass_animation(self, layer_args={}, run_time=1.5, **kwargs):
        """Forward pass animation from conv to conv"""
        animations = []
        # Go thorugh the lines
        for line in self.lines:
            pulse = ShowPassingFlash(
                line.copy()
                    .set_color(self.pulse_color)
                    .set_stroke(opacity=1.0), 
                    time_width=0.5,
                    run_time=run_time
                )
            animations.append(pulse)
        # Make animation group
        animation_group = AnimationGroup(
            *animations,
            run_time=run_time
        )

        return animation_group

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return AnimationGroup()

