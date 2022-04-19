from manim import *
import numpy as np
import math

class GaussianDistribution(VGroup):
    """Object for drawing a Gaussian distribution"""

    def __init__(self, axes, mean=None, cov=None, **kwargs):
        super(VGroup, self).__init__(**kwargs)
        self.axes = axes
        self.mean = mean
        self.cov = cov
        if mean is None:
            self.mean = np.array([0.0, 0.0])
        if cov is None:
            self.cov = np.array([[3, 0], [0, 3]])
        # Make the Gaussian
        self.ellipses = self.construct_gaussian_distribution(self.mean, self.cov)
        
    @override_animation(Create)
    def _create_gaussian_distribution(self):
        return Create(self.ellipses)

    def compute_covariance_rotation_and_scale(self, covariance):
        # Get the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        y, x = eigenvectors[0, 1], eigenvectors[0, 0]
        center_location = np.array([y, x, 0])
        center_location = self.axes.coords_to_point(*center_location)
        angle = math.atan(x / y) # x over y to denote the angle between y axis and vector
        # Calculate the width and height
        height = np.abs(eigenvalues[0])
        width = np.abs(eigenvalues[1])
        shape_coord = np.array([width, height, 0])
        shape_coord = self.axes.coords_to_point(*shape_coord)
        width = shape_coord[0]
        height = shape_coord[1]
        
        return angle, width, height

    def construct_gaussian_distribution(self, mean, covariance, color=ORANGE, 
                                        num_ellipses=4):
        """Returns a 2d Gaussian distribution object with given mean and covariance"""
        # map mean and covariance to frame coordinates
        mean = self.axes.coords_to_point(*mean)
        # Figure out the scale and angle of rotation
        rotation, width, height = self.compute_covariance_rotation_and_scale(covariance)
        # Make covariance ellipses
        opacity = 0.0
        ellipses = VGroup()
        for ellipse_number in range(num_ellipses):
            opacity += 1.0 / num_ellipses
            ellipse_width = width * (1 - opacity)
            ellipse_height = height * (1 - opacity)
            ellipse = Ellipse(
                width=ellipse_width, 
                height=ellipse_height, 
                color=color, 
                fill_opacity=opacity, 
                stroke_width=0.0
            )
            ellipse.move_to(mean)
            ellipse.rotate(rotation)
            ellipses.add(ellipse)

        return ellipses

