from manim import *
import numpy as np

class CornersRectangle(Rectangle):
    """Rectangle with functionality for getting the corner coordinates"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corners = VGroup(
            *[Dot(corner_point) for corner_point in self.get_vertices()]
        )
        self.corners.set_fill(opacity=0.0)
        self.add(self.corners)

    def get_corners_dict(self):
        """Returns a dictionary of the corners"""
        return {
            "top_left": self.corners[3],
            "top_right": self.corners[0],
            "bottom_left": self.corners[2],
            "bottom_right": self.corners[1], 
        }

class GriddedRectangle(VGroup):
    """Rectangle object with grid lines"""

    def __init__(self, center, color=WHITE, height=2.0, width=4.0, 
                mark_paths_closed=True, close_new_points=True,
                grid_xstep=None, grid_ystep=None, grid_stroke_width=0.0, #DEFAULT_STROKE_WIDTH/2, 
                grid_stroke_color=None, grid_stroke_opacity=None, **kwargs):
        super().__init__()
        # Fields
        self.center = center
        self.mark_paths_closed = mark_paths_closed
        self.close_new_points = close_new_points
        self.grid_xstep = grid_xstep
        self.grid_ystep = grid_ystep
        self.grid_stroke_width = grid_stroke_width
        self.grid_stroke_color = grid_stroke_color
        self.grid_stroke_opacity = grid_stroke_opacity
        self.rotation_angles = [0, 0, 0]
        # Make inner_rectangle
        self.inner_rectangle = Rectangle(
            width=width, 
            height=height, 
            stroke_opacity=0.0,
            stroke_width=0.0
        )
        print(self.inner_rectangle.get_vertices())
        # self.inner_rectangle = Polygon(
        """
        points = [
            self.center + np.array([width / 2, height / 2, 0]),
            self.center + np.array([width / 2, -1*(height / 2), 0]),
            self.center + np.array([-1 * (width / 2), -1 * (height / 2), 0]),
            self.center + np.array([-1 * (width / 2), height / 2, 0]),
        ]
        self.inner_rectangle = Polygram(
            points,
            stroke_opacity=0.0,
            stroke_width=0.0
        )
        """
        self.add(self.inner_rectangle)
        # Make outline rectangle
        self.outline_rectangle = SurroundingRectangle(
            self.inner_rectangle, 
            color=color,
            buff=0.0,
            **kwargs
        )
        self.add(self.outline_rectangle)
        # Move to center
        self.move_to(self.center)
        # Setup Object
        # TODO re-implement gridded rectangle
        # self.grid_lines = self.make_grid_lines()
        # self.add(self.grid_lines)
        # Make dots for the corners
        # Make outer corner dots
        self.outer_corners = VGroup(
            *[Dot(corner_point) for corner_point in self.outline_rectangle.get_vertices()]
        )

        self.outer_corners.set_fill(opacity=0.0)
        self.add(self.outer_corners)
        # Make inner corner dots
        self.inner_corners = VGroup(
            *[Dot(corner_point) for corner_point in self.inner_rectangle.get_vertices()]
        )
        self.inner_corners.set_fill(opacity=0.0)
        self.add(self.inner_corners)
    
    def get_corners_dict(self, inner_rectangle=False):
        """Returns a dictionary of the corners"""
        if inner_rectangle:
            return {
                "top_left": self.inner_corners[3],
                "top_right": self.inner_corners[0],
                "bottom_left": self.inner_corners[2],
                "bottom_right": self.inner_corners[1], 
            }
        else:
            return {
                "top_left": self.outer_corners[3],
                "top_right": self.outer_corners[0],
                "bottom_left": self.outer_corners[2],
                "bottom_right": self.outer_corners[1], 
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

    def rotate_about_origin(self, angle, axis=OUT, axes=[]):
        self.rotation_angles[np.nonzero(axis)[0][0]] = angle
        return super().rotate_about_origin(angle, axis, axes)

    def get_normal_vector(self):
        """Gets the vector normal to main rectangle face"""
        # Get three corner points
        corner_1 = self.rectangle.get_top()
        corner_2 = self.rectangle.get_left()
        corner_3 = self.rectangle.get_right()
        # Make vectors from them
        a = corner_1 - corner_3
        b = corner_1 - corner_2
        # Compute cross product
        normal_vector = np.cross(b, a)
        normal_vector /= np.linalg.norm(normal_vector)

        return normal_vector

    def get_rotation_axis_and_angle(self):
        """Gets the angle of rotation necessary to rotate something from the default z-axis to the rectangle"""
        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::

                    >>> angle_between((1, 0, 0), (0, 1, 0))
                    1.5707963267948966
                    >>> angle_between((1, 0, 0), (1, 0, 0))
                    0.0
                    >>> angle_between((1, 0, 0), (-1, 0, 0))
                    3.141592653589793
            """
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        normal_vector = self.get_normal_vector()
        z_axis = Z_AXIS
        # Get angle between normal vector and z axis
        axis = np.cross(normal_vector, z_axis)
        angle = angle_between(normal_vector, z_axis)

        return axis, angle