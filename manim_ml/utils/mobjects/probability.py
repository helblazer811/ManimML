from manim import *
import numpy as np
import math


class GaussianDistribution(VGroup):
    """Object for drawing a Gaussian distribution"""

    def __init__(
        self, axes, mean=None, cov=None, dist_theme="gaussian", color=ORANGE, **kwargs
    ):
        super(VGroup, self).__init__(**kwargs)
        self.axes = axes
        self.mean = mean
        self.cov = cov
        self.dist_theme = dist_theme
        self.color = color
        if mean is None:
            self.mean = np.array([0.0, 0.0])
        if cov is None:
            self.cov = np.array([[1, 0], [0, 1]])
        # Make the Gaussian
        if self.dist_theme is "gaussian":
            self.ellipses = self.construct_gaussian_distribution(
                self.mean, self.cov, color=self.color
            )
            self.add(self.ellipses)
        elif self.dist_theme is "ellipse":
            self.ellipses = self.construct_simple_gaussian_ellipse(
                self.mean, self.cov, color=self.color
            )
            self.add(self.ellipses)
        else:
            raise Exception(f"Uncrecognized distribution theme: {self.dist_theme}")

    """  
    @override_animation(Create)
    def _create_gaussian_distribution(self):
        return Create(self)
    """

    def compute_covariance_rotation_and_scale(self, covariance):
        def eigsorted(cov):
            """
            Eigenvalues and eigenvectors of the covariance matrix.
            """
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        def cov_ellipse(cov, nstd):
            """
            Source: http://stackoverflow.com/a/12321306/1391441
            """

            vals, vecs = eigsorted(cov)
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            # Width and height are "full" widths, not radius
            width, height = 2 * nstd * np.sqrt(vals)

            return width, height, theta

        width, height, angle = cov_ellipse(covariance, 1)
        scale_factor = (
            np.abs(self.axes.x_range[0] - self.axes.x_range[1]) / self.axes.x_length
        )
        width /= scale_factor
        height /= scale_factor
        return angle, width, height

    def construct_gaussian_distribution(
        self, mean, covariance, color=ORANGE, num_ellipses=4
    ):
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
                stroke_width=2.0,
            )
            ellipse.move_to(mean)
            ellipse.rotate(rotation)
            ellipses.add(ellipse)

        return ellipses

    def construct_simple_gaussian_ellipse(self, mean, covariance, color=ORANGE):
        """Returns a 2d Gaussian distribution object with given mean and covariance"""
        # Map mean and covariance to frame coordinates
        mean = self.axes.coords_to_point(*mean)
        angle, width, height = self.compute_covariance_rotation_and_scale(covariance)
        # Make covariance ellipses
        ellipses = VGroup()
        opacity = 0.4
        ellipse = Ellipse(
            width=width,
            height=height,
            color=color,
            fill_opacity=opacity,
            stroke_width=1.0,
        )
        ellipse.move_to(mean)
        ellipse.rotate(angle)
        ellipses.add(ellipse)
        ellipses.set_z_index(3)

        return ellipses
