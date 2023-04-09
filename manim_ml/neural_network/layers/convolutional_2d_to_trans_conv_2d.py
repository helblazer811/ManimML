import numpy as np

from manim import *
from manim_ml.utils.mobjects.gridded_rectangle import GriddedRectangle
from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer, FeatureMap
from manim_ml.neural_network.layers.trans_conv_2d import TransposeConvolution2DLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer, ThreeDLayer
from manim_ml.neural_network.layers.convolutional_2d_to_convolutional_2d import Convolutional2DToConvolutional2D


class Convolutional2DToTransConv2D(Convolutional2DToConvolutional2D,ConnectiveLayer, ThreeDLayer):
    input_class = Convolutional2DLayer
    output_class = TransposeConvolution2DLayer

    def __init__(
            self,
            input_layer: Convolutional2DLayer,
            output_layer: TransposeConvolution2DLayer,
            # color=ORANGE,
            # filter_opacity=0.3,
            # line_color=ORANGE,
            active_color=ORANGE,
            # cell_width=0.2,
            # show_grid_lines=True,
            # highlight_color=ORANGE,
            **kwargs,
    ):
        super().__init__(input_layer, output_layer, **kwargs)
        self.active_color = active_color

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs,
    ):
        return super().construct_layer(input_layer, output_layer, **kwargs)
    
    def make_forward_pass_animation(self, layer_args={}, run_time=1.5, **kwargs):
        """Forward pass animation from conv2d to transposed conv2d"""
        
        new_input_layer_feature_map_size= (
            self.input_layer.feature_map_size[0] * (self.output_layer.in_padding[0]+1) + self.output_layer.in_padding[0],
            self.input_layer.feature_map_size[1] * (self.output_layer.in_padding[1]+1) + self.output_layer.in_padding[1],
        )

        #Replace previous mobject with something

        grid_stroke_width = 1
        cell_width = self.input_layer.cell_width

        new_feature_animation = []

        for feature_map_index, feature_map in enumerate(self.input_layer.feature_maps):
            gridded_rectangle = GriddedRectangle(
                color=self.active_color,
                    height = new_input_layer_feature_map_size[1]*cell_width,
                    width = new_input_layer_feature_map_size[0]*cell_width,
                    grid_xstep=cell_width,
                    grid_ystep=cell_width,
                    grid_stroke_width=grid_stroke_width,
                    grid_stroke_color=self.active_color,
                    show_grid_lines=True,
            )

            gridded_rectangle.set_z_index(1)
            # 2. Randomly highlight one of the cells in the kernel.
            highlighted_cells = []

            for kernel_x in range(self.output_layer.in_padding[0], self.input_layer.feature_map_size[0]*(self.output_layer.in_padding[0]+1), self.output_layer.in_padding[0]+1):
                    for kernel_y in range(self.output_layer.in_padding[1], self.input_layer.feature_map_size[1]*(self.output_layer.in_padding[1]+1), self.output_layer.in_padding[1]+1):
                        cell_rectangle = GriddedRectangle(
                            color=self.active_color,
                            height=cell_width,
                            width=cell_width,
                            fill_opacity=0.7,
                            stroke_width=0.0,
                            z_index=10,
                        )

                        # Move to the correct location
                        kernel_shift_vector = [
                            cell_width * kernel_x,
                            -1 * cell_width * kernel_y,
                            0,
                        ]

                        cell_rectangle.next_to(
                            gridded_rectangle.get_corners_dict()["top_left"],
                            submobject_to_align=cell_rectangle.get_corners_dict()[
                                "top_left"
                            ],
                            buff=0.0,
                        )
                        cell_rectangle.shift(kernel_shift_vector)
                        highlighted_cells.append(cell_rectangle)


            qqq = gridded_rectangle.copy()
            qqq.set_opacity(0)
            self.input_layer.feature_maps[feature_map_index] = qqq.rotate(
                ThreeDLayer.rotation_angle,
                about_point=qqq.get_center(),
                axis=ThreeDLayer.rotation_axis,
            ).move_to(feature_map.get_center())

            # Rotate the gridded rectangles so they match the angle
            # of the conv maps
            gridded_rectangle_group = VGroup(gridded_rectangle, *highlighted_cells)
            gridded_rectangle_group.rotate(
                ThreeDLayer.rotation_angle,
                about_point=gridded_rectangle.get_center(),
                axis=ThreeDLayer.rotation_axis,
            )

            gridded_rectangle_group.move_to(
                feature_map.get_center(),
            )

            # new_feature_maps.append(gridded_rectangle_group)
            new_feature_animation.append(ReplacementTransform(
                feature_map,
                gridded_rectangle_group,
            ))

        
        # Change the feature map size of the input layer to account for padding
        self.input_layer.feature_map_size = new_input_layer_feature_map_size
        
        self.feature_map_size = new_input_layer_feature_map_size
        
        # Call on the super class to make the animation for the conv2d to conv2d now that you
        # have the padded feature maps
        conv2d = super().make_forward_pass_animation(layer_args, run_time, **kwargs)
        

        return Succession(AnimationGroup(*new_feature_animation), Wait(1), conv2d)
    
