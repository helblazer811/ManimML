from manim import *
import numpy as np


class GriddedRectangle(VGroup):
    """Rectangle object with grid lines"""

    def __init__(
        self,
        color=ORANGE,
        height=2.0,
        width=4.0,
        mark_paths_closed=True,
        close_new_points=True,
        grid_xstep=None,
        grid_ystep=None,
        grid_stroke_width=0.0,  # DEFAULT_STROKE_WIDTH/2,
        grid_stroke_color=ORANGE,
        grid_stroke_opacity=1.0,
        stroke_width=2.0,
        fill_opacity=0.2,
        show_grid_lines=False,
        **kwargs
    ):
        super().__init__()
        # Fields
        self.color = color
        self.mark_paths_closed = mark_paths_closed
        self.close_new_points = close_new_points
        self.grid_xstep = grid_xstep
        self.grid_ystep = grid_ystep
        self.grid_stroke_width = grid_stroke_width
        self.grid_stroke_color = grid_stroke_color
        self.grid_stroke_opacity = grid_stroke_opacity if show_grid_lines else 0.0
        self.stroke_width = stroke_width
        self.rotation_angles = [0, 0, 0]
        self.show_grid_lines = show_grid_lines
        self.untransformed_width = width
        self.untransformed_height = height
        # Make rectangle
        self.rectangle = Rectangle(
            width=width,
            height=height,
            color=color,
            stroke_width=stroke_width,
            fill_color=color,
            fill_opacity=fill_opacity,
            shade_in_3d=True
        )
        self.add(self.rectangle)
        # Make grid lines
        grid_lines = self.make_grid_lines()
        self.add(grid_lines)
        # Make corner rectangles
        self.corners_dict = self.make_corners_dict()
        self.add(*self.corners_dict.values())

    def make_corners_dict(self):
        """Make corners dictionary"""
        corners_dict = {
            "top_right": Dot(
                self.rectangle.get_corner([1, 1, 0]), fill_opacity=0.0, radius=0.0
            ),
            "top_left": Dot(
                self.rectangle.get_corner([-1, 1, 0]), fill_opacity=0.0, radius=0.0
            ),
            "bottom_left": Dot(
                self.rectangle.get_corner([-1, -1, 0]), fill_opacity=0.0, radius=0.0
            ),
            "bottom_right": Dot(
                self.rectangle.get_corner([1, -1, 0]), fill_opacity=0.0, radius=0.0
            ),
        }

        return corners_dict

    def get_corners_dict(self):
        """Returns a dictionary of the corners"""
        # Sort points through clockwise rotation of a vector in the xy plane
        return self.corners_dict

    def make_grid_lines(self):
        """Make grid lines in rectangle"""
        grid_lines = VGroup()

        v = self.rectangle.get_vertices()
        if self.grid_xstep is not None:
            grid_xstep = abs(self.grid_xstep)
            count = int(self.width / grid_xstep)
            grid = VGroup(
                *(
                    Line(
                        v[1] + i * grid_xstep * RIGHT,
                        v[1] + i * grid_xstep * RIGHT + self.height * DOWN,
                        stroke_color=self.grid_stroke_color,
                        stroke_width=self.grid_stroke_width,
                        stroke_opacity=self.grid_stroke_opacity,
                        shade_in_3d=True
                    )
                    for i in range(1, count)
                )
            )
            grid_lines.add(grid)

        if self.grid_ystep is not None:
            grid_ystep = abs(self.grid_ystep)
            count = int(self.height / grid_ystep)
            grid = VGroup(
                *(
                    Line(
                        v[1] + i * grid_ystep * DOWN,
                        v[1] + i * grid_ystep * DOWN + self.width * RIGHT,
                        stroke_color=self.grid_stroke_color,
                        stroke_width=self.grid_stroke_width,
                        stroke_opacity=self.grid_stroke_opacity,
                    )
                    for i in range(1, count)
                )
            )
            grid_lines.add(grid)

        return grid_lines

    def get_center(self):
        return self.rectangle.get_center()

    def get_normal_vector(self):
        vertex_1 = self.rectangle.get_vertices()[0]
        vertex_2 = self.rectangle.get_vertices()[1]
        vertex_3 = self.rectangle.get_vertices()[2]
        # First vector
        normal_vector = np.cross((vertex_1 - vertex_2), (vertex_1 - vertex_3))

        return normal_vector

    def set_color(self, color):
        """Sets the color of the gridded rectangle"""
        self.color = color
        self.rectangle.set_color(color)
        self.rectangle.set_stroke_color(color)
