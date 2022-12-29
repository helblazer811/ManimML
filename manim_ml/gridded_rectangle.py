from manim import *
import numpy as np

class GriddedRectangle(VGroup):
    """Rectangle object with grid lines"""

    def __init__(self, color=ORANGE, height=2.0, width=4.0, 
                mark_paths_closed=True, close_new_points=True,
                grid_xstep=None, grid_ystep=None, grid_stroke_width=0.0, #DEFAULT_STROKE_WIDTH/2, 
                grid_stroke_color=None, grid_stroke_opacity=None,
                stroke_width=2.0, fill_opacity=0.2, **kwargs):
        super().__init__()
        # Fields
        self.mark_paths_closed = mark_paths_closed
        self.close_new_points = close_new_points
        self.grid_xstep = grid_xstep
        self.grid_ystep = grid_ystep
        self.grid_stroke_width = grid_stroke_width
        self.grid_stroke_color = grid_stroke_color
        self.grid_stroke_opacity = grid_stroke_opacity
        self.stroke_width = stroke_width
        self.rotation_angles = [0, 0, 0]
        # Make rectangle
        self.rectangle = Rectangle(
            width=width, 
            height=height, 
            color=color,
            stroke_width=stroke_width,
            fill_color=color,
            fill_opacity=fill_opacity
        )
        self.add(self.rectangle)
        
    def get_corners_dict(self):
        """Returns a dictionary of the corners"""
        # Sort points through clockwise rotation of a vector in the xy plane
        return{
            "top_right": Dot(self.rectangle.get_corner([1, 1, 0])),
            "top_left": Dot(self.rectangle.get_corner([-1, 1, 0])),
            "bottom_left": Dot(self.rectangle.get_corner([-1, -1, 0])),
            "bottom_right": Dot(self.rectangle.get_corner([1, -1, 0])),
        }

    def make_grid_lines(self):
        """Make grid lines in rectangle"""
        grid_lines = VGroup()
        width = self.width
        height = self.width

        v = self.inner_rectangle.get_vertices()
        if self.grid_xstep is not None:
            grid_xstep = abs(self.grid_xstep)
            count = int(width / grid_xstep)
            grid = VGroup(
                *(
                    Line(
                        v[1] + i * grid_xstep * RIGHT,
                        v[1] + i * grid_xstep * RIGHT + height * DOWN,
                        color=self.color,
                        stroke_width=self.grid_stroke_width
                    )
                    for i in range(1, count)
                )
            )
            grid_lines.add(grid)
    
        if self.grid_ystep is not None:
            grid_ystep = abs(self.grid_ystep)
            count = int(height / grid_ystep)
            grid = VGroup(
                *(
                    Line(
                        v[1] + i * grid_ystep * DOWN,
                        v[1] + i * grid_ystep * DOWN + width * RIGHT,
                        color=self.color,
                        stroke_width = self.grid_stroke_width
                    )
                    for i in range(1, count)
                )
            ) 
            grid_lines.add(grid)

        return grid_lines

    def get_center(self):
        return self.rectangle.get_center()
