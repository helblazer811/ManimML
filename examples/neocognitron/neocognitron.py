from manim import *

from manim_ml.neural_network import NeuralNetwork
from manim_ml.neural_network.layers.parent_layers import NeuralNetworkLayer, ConnectiveLayer, ThreeDLayer
import manim_ml

config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 10.5
config.frame_width = 10.5

class NeocognitronFilter(VGroup):
    """
        Filter connecting the S and C Cells of a neocognitron layer.
    """

    def __init__(
        self,
        input_cell, 
        output_cell, 
        receptive_field_radius, 
        outline_color=BLUE,
        active_color=ORANGE, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.outline_color = outline_color
        self.active_color = active_color
        self.input_cell = input_cell
        self.output_cell = output_cell
        # Draw the receptive field
        # Make the points of a equilateral triangle in the plane of the 
        # cell about a random center
        # center_point = input_cell.get_center()
        shift_amount_x = np.random.uniform(
            -(input_cell.get_height()/2 - receptive_field_radius - 0.01),
            input_cell.get_height()/2 - receptive_field_radius - 0.01,
        )
        shift_amount_y = np.random.uniform(
            -(input_cell.get_height()/2 - receptive_field_radius - 0.01),
            input_cell.get_height()/2 - receptive_field_radius - 0.01,
        )
        # center_point += np.array([shift_amount_x, shift_amount_y, 0])
        # # Make the triangle points
        # side_length = np.sqrt(3) * receptive_field_radius
        # normal_vector = np.cross(
        #     input_cell.get_left() - input_cell.get_center(),
        #     input_cell.get_top() - input_cell.get_center(),
        # )
        # Get vector in the plane of the cell
        # vector_in_plane = input_cell.get_left() - input_cell.get_center()
        # point_a = center_point + vector_in_plane * receptive_field_radius
        # # rotate the vector 120 degrees about the vector normal to the cell
        # vector_in_plane = rotate_vector(vector_in_plane, PI/3, normal_vector)
        # point_b = center_point + vector_in_plane * receptive_field_radius
        # # rotate the vector 120 degrees about the vector normal to the cell
        # vector_in_plane = rotate_vector(vector_in_plane, PI/3, normal_vector)
        # point_c = center_point + vector_in_plane * receptive_field_radius
        # self.receptive_field = Circle.from_three_points(
        #     point_a, 
        #     point_b,
        #     point_c,
        #     color=outline_color,
        #     stroke_width=2.0,
        # )
        self.receptive_field = Circle(
            radius=receptive_field_radius,
            color=outline_color,
            stroke_width=3.0,
        )
        self.add(self.receptive_field)
        # Move receptive field to a random point within the cell
        # minus the radius of the receptive field
        self.receptive_field.move_to(
            input_cell
        )
        # Shift a random amount in the x and y direction within 
        self.receptive_field.shift(
            np.array([shift_amount_x, shift_amount_y, 0])
        )
        # Choose a random point on the c_cell
        shift_amount_x = np.random.uniform(
            -(output_cell.get_height()/2 - receptive_field_radius - 0.01),
            output_cell.get_height()/2 - receptive_field_radius - 0.01,
        )
        shift_amount_y = np.random.uniform(
            -(output_cell.get_height()/2 - receptive_field_radius - 0.01),
            output_cell.get_height()/2 - receptive_field_radius - 0.01,
        )
        self.dot = Dot(
            color=outline_color,
            radius=0.04
        )
        self.dot.move_to(output_cell.get_center() + np.array([shift_amount_x, shift_amount_y, 0]))
        self.add(self.dot)
        # Make lines connecting receptive field to the dot
        self.lines = VGroup()
        self.lines.add(
            Line(
                self.receptive_field.get_center() + np.array([0, receptive_field_radius, 0]),
                self.dot,
                color=outline_color,
                stroke_width=3.0,
            )
        )
        self.lines.add(
            Line(
                self.receptive_field.get_center() - np.array([0, receptive_field_radius, 0]),
                self.dot,
                color=outline_color,
                stroke_width=3.0,
            )
        )
        self.add(self.lines)

    def make_filter_pulse_animation(self, **kwargs):
        succession = Succession(
            ApplyMethod(
                self.receptive_field.set_color, 
                self.active_color, 
                run_time=0.25
            ),
            ApplyMethod(
                self.receptive_field.set_color, 
                self.outline_color, 
                run_time=0.25
            ),
            ShowPassingFlash(
                self.lines.copy().set_color(self.active_color),
                time_width=0.5,
            ),
            ApplyMethod(
                self.dot.set_color, 
                self.active_color, 
                run_time=0.25
            ),
            ApplyMethod(
                self.dot.set_color, 
                self.outline_color, 
                run_time=0.25
            ),
        )

        return succession

class NeocognitronLayer(NeuralNetworkLayer, ThreeDLayer):
    
    def __init__(
        self, 
        num_cells, 
        cell_height,
        receptive_field_radius,
        layer_name,
        active_color=manim_ml.config.color_scheme.active_color,
        cell_color=manim_ml.config.color_scheme.secondary_color,
        outline_color=manim_ml.config.color_scheme.primary_color,
        show_outline=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_cells = num_cells
        self.cell_height = cell_height
        self.receptive_field_radius = receptive_field_radius
        self.layer_name = layer_name
        self.active_color = active_color
        self.cell_color = cell_color
        self.outline_color = outline_color
        self.show_outline = show_outline

        self.plane_label = Text(layer_name).scale(0.4)

    def make_cell_plane(self, layer_name, cell_buffer=0.1, stroke_width=2.0):
        """Makes a plane of cells. 
        """
        cell_plane = VGroup()
        for cell_index in range(self.num_cells):
            # Draw the cell box
            cell_box = Rectangle(
                width=self.cell_height,
                height=self.cell_height,
                color=self.cell_color,
                fill_color=self.cell_color,
                fill_opacity=0.3,
                stroke_width=stroke_width,
            )
            if cell_index > 0:
                cell_box.next_to(cell_plane[-1], DOWN, buff=cell_buffer)
            cell_plane.add(cell_box)
        # Draw the outline
        if self.show_outline:
            self.plane_outline = SurroundingRectangle(
                cell_plane,
                color=self.cell_color,
                buff=cell_buffer,
                stroke_width=stroke_width,
            )
            cell_plane.add(
                self.plane_outline
            )
        # Draw a label above the container box
        self.plane_label.next_to(cell_plane, UP, buff=0.2)
        cell_plane.add(self.plane_label)

        return cell_plane

    def construct_layer(self, input_layer, output_layer, **kwargs):
        # Make the Cell layer
        self.cell_plane = self.make_cell_plane(self.layer_name)
        self.add(self.cell_plane)
        
    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        """Forward pass for query"""
        # # Pulse and un pulse the cell plane rectangle
        flash_outline_animations = []
        for cell in self.cell_plane:
            flash_outline_animations.append(
                Succession(
                    ApplyMethod(
                        cell.set_stroke_color,
                        self.active_color,
                        run_time=0.25
                    ),
                    ApplyMethod(
                        cell.set_stroke_color,
                        self.outline_color,
                        run_time=0.25
                    )
                )
            )
        
        return AnimationGroup(
            *flash_outline_animations,
            lag_ratio=0.0
        )

class NeocognitronToNeocognitronLayer(ConnectiveLayer):
    input_class = NeocognitronLayer
    output_class = NeocognitronLayer

    def __init__(self, input_layer, output_layer, **kwargs):
        super().__init__(input_layer, output_layer, **kwargs)

    def construct_layer(self, input_layer, output_layer, **kwargs):
        self.filters = []
        for cell_index in range(input_layer.num_cells):
            input_cell = input_layer.cell_plane[cell_index]
            # Randomly choose a cell from the output layer
            output_cell = output_layer.cell_plane[
                np.random.randint(0, output_layer.num_cells)
            ]
            # Make the filter
            filter = NeocognitronFilter(
                input_cell=input_cell,
                output_cell=output_cell,
                receptive_field_radius=input_layer.receptive_field_radius,
                outline_color=self.input_layer.outline_color
            )
            # filter = NeocognitronFilter(
            #     outline_color=input_layer.outline_color
            # )
            self.filters.append(filter)

        self.add(VGroup(*self.filters))

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        """Forward pass for query"""
        filter_pulses = []
        for filter in self.filters:
            filter_pulses.append(
                filter.make_filter_pulse_animation()
            )
        return AnimationGroup(
            *filter_pulses
        )

manim_ml.neural_network.layers.util.register_custom_connective_layer(
    NeocognitronToNeocognitronLayer,
)

class Neocognitron(NeuralNetwork):

    def __init__(
        self, 
        camera,
        cells_per_layer=[4, 5, 4, 4, 3, 3, 5, 5],
        cell_heights=[0.8, 0.8, 0.8*0.75, 0.8*0.75, 0.8*0.75**2, 0.8*0.75**2, 0.8*0.75**3, 0.02],
        layer_names=["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4"],
        receptive_field_sizes=[0.2, 0.2, 0.2*0.75, 0.2*0.75, 0.2*0.75**2, 0.2*0.75**2, 0.2*0.75**3, 0.0],
    ):
        self.cells_per_layer = cells_per_layer
        self.cell_heights = cell_heights
        self.layer_names = layer_names
        self.receptive_field_sizes = receptive_field_sizes
        # Make the neo-cognitron input layer
        input_layers = []
        input_layers.append(
            NeocognitronLayer(
                1,
                1.5,
                0.2,
                "U0",
                show_outline=False,
            )
        )
        # Make the neo-cognitron layers
        for i in range(len(cells_per_layer)):
            layer = NeocognitronLayer(
                cells_per_layer[i],
                cell_heights[i],
                receptive_field_sizes[i],
                layer_names[i]
            )
            input_layers.append(layer)
        
        # Make all of the layer labels fixed in frame
        for layer in input_layers:
            if isinstance(layer, NeocognitronLayer):
                # camera.add_fixed_orientation_mobjects(layer.plane_label)
                pass

        all_layers = []
        for layer_index in range(len(input_layers) - 1):
            input_layer = input_layers[layer_index]
            all_layers.append(input_layer)
            output_layer = input_layers[layer_index + 1]
            connective_layer = NeocognitronToNeocognitronLayer(
                input_layer,
                output_layer
            )
            all_layers.append(connective_layer)
        all_layers.append(input_layers[-1])

        def custom_layout_func(neural_network):
            # Insert the connective layers
            # Pass the layers to neural network super class
            # Rotate pairs of layers
            layer_index = 1
            while layer_index < len(input_layers):
                prev_layer = input_layers[layer_index - 1]
                s_layer = input_layers[layer_index]
                s_layer.next_to(prev_layer, RIGHT, buff=0.0)
                s_layer.shift(RIGHT * 0.4)
                c_layer = input_layers[layer_index + 1]
                c_layer.next_to(s_layer, RIGHT, buff=0.0)
                c_layer.shift(RIGHT * 0.2)
                # Rotate the pair of layers
                group = Group(s_layer, c_layer)
                group.move_to(np.array([group.get_center()[0], 0, 0]))

                # group.shift(layer_index * 3 * np.array([0, 0, 1.0]))
                # group.rotate(40 * DEGREES, axis=UP, about_point=group.get_center())
                # c_layer.rotate(40*DEGREES, axis=UP, about_point=group.get_center())
                # s_layer.shift(
                #     layer_index // 2 * np.array([0, 0, 0.3])
                # )
                # c_layer.shift(
                #     layer_index // 2 * np.array([0, 0, 0.3])
                # )
                layer_index += 2
            neural_network.move_to(ORIGIN)

        super().__init__(
            all_layers,
            layer_spacing=0.5,
            custom_layout_func=custom_layout_func
        )

class Scene(ThreeDScene):

    def construct(self):
        neocognitron = Neocognitron(self.camera)
        neocognitron.shift(DOWN*0.5)
        self.add(neocognitron)
        title = Text("Neocognitron").scale(1)
        self.add_fixed_in_frame_mobjects(title)
        title.next_to(neocognitron, UP, buff=0.4)
        self.add(title)
        """
        self.play(
            neocognitron.make_forward_pass_animation()
        )
        """
        print(self.camera.gamma)
        print(self.camera.theta)
        print(self.camera.phi)
        neocognitron.rotate(90*DEGREES, axis=RIGHT)
        neocognitron.shift(np.array([0, 0, -0.2]))
        # title.rotate(90*DEGREES, axis=RIGHT)
        # self.set_camera_orientation(phi=-15*DEGREES)
        # vec = np.array([0.0, 0.2, 0.0])
        # vec /= np.linalg.norm(vec)
        # x, y, z = vec[0], vec[1], vec[2]
        # theta = np.arccos(z)
        # phi = np.arctan(y / x)
        self.set_camera_orientation(phi=90 * DEGREES, theta=-70*DEGREES, gamma=0*DEGREES)
        # self.set_camera_orientation(theta=theta, phi=phi)

        forward_pass = neocognitron.make_forward_pass_animation()
        self.play(forward_pass)

