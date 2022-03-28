from manim import *

class NeuralNetworkEmbedding(Axes):
    """NeuralNetwork embedding object that can show probability distributions"""

    def construct_gaussian_distribution(self, mean, covariance, color=ORANGE, dot_radius=0.05, ellipse_stroke_width=0.3):
        """Returns a 2d Gaussian distribution object with given mean and covariance"""
        # map mean and covariance to frame coordinates
        mean = self.coords_to_point(*mean)
        covariance = self.coords_to_point(*covariance)
        # Make a covariance ellipse centered at mean 
        center_dot = Dot(mean, radius=dot_radius, color=color)
        ellipse = Ellipse(width=covariance[0], height=covariance[1], color=color, fill_opacity=0.3, stroke_width=ellipse_stroke_width)
        ellipse.move_to(mean)
        gaussian_distribution = VGroup(
            center_dot, 
            ellipse
        )

        return gaussian_distribution