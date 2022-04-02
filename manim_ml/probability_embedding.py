from manim import *

from manim_ml.neural_network import NeuralNetwork
import numpy as np
import math

class NeuralNetworkEmbedding(Axes):
    """NeuralNetwork embedding object that can show probability distributions"""

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

    def construct_gaussian_point_cloud(self, mean, covariance, color=BLUE):
        """Plots points sampled from a Gaussian with the given mean and covariance"""
        pass

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

class NeuralNetworkEmbeddingTestScene(Scene):

    def construct(self):
        nne = NeuralNetworkEmbedding()
        mean = np.array([0, 0])
        cov = np.array([[0.1, 0.8], [0.0, 0.8]])
        
        gaussian = nne.construct_gaussian_distribution(mean, cov)
        gaussian.scale(3)

        self.add(gaussian)