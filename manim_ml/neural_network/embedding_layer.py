from manim import *
from manim_ml.neural_network.layers import NeuralNetworkLayer
import numpy as np
import math

class NeuralNetworkEmbedding(NeuralNetworkLayer, Axes):
    """NeuralNetwork embedding object that can show probability distributions"""

    def __init__(self):
        super().__init__(NeuralNetworkEmbedding, self)

    def compute_covariance_rotation_and_scale(self, covariance):
        # Get the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        y, x = eigenvectors[0, 1], eigenvectors[0, 0]
        print(eigenvectors[0])
        angle = math.atan(x / y) # x over y to denote the angle between y axis and vector
        # Calculate the width and height
        height = np.abs(eigenvalues[0])
        width = np.abs(eigenvalues[1])
        return angle, width, height

    def construct_gaussian_distribution(self, mean, covariance, color=ORANGE, 
                                        dot_radius=0.05, num_ellipses=4):
        """Returns a 2d Gaussian distribution object with given mean and covariance"""
        # map mean and covariance to frame coordinates
        mean = self.coords_to_point(*mean)
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

    def construct_gaussian_point_cloud(self, mean, covariance, color=BLUE):
        """Plots points sampled from a Gaussian with the given mean and covariance"""
        embedding = VGroup()
        # Sample points from a Gaussian
        num_points = 200
        standard_deviation = [0.9, 0.9]
        mean = [0, 0]
        points = np.random.normal(mean, standard_deviation, size=(num_points, 2))
        # Make an axes
        embedding.axes = Axes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            x_length=2.2,
            y_length=2.2,
            tips=False,
        )
        # Add each point to the axes
        self.point_dots = VGroup()
        for point in points:
            point_location = embedding.axes.coords_to_point(*point)
            dot = Dot(point_location, color=self.point_color, radius=self.dot_radius/2) 
            self.point_dots.add(dot)

        embedding.add(self.point_dots)

        return embedding

    def make_forward_pass_animation(self):

        pass

class NeuralNetworkEmbeddingTestScene(Scene):

    def construct(self):
        nne = NeuralNetworkEmbedding()
        mean = np.array([0, 0])
        cov = np.array([[0.1, 0.8], [0.0, 0.8]])

        point_cloud = nne.construct_gaussian_point_cloud(mean, cov)
        self.add(point_cloud)
        gaussian = nne.construct_gaussian_distribution(mean, cov)
        gaussian.scale(3)

        self.add(gaussian)