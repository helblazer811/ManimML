from manim import *
from manim_ml.neural_network.layers import ConnectiveLayer, VGroupNeuralNetworkLayer
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
        self.ellipses.set_z_index(2)

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

class EmbeddingLayer(VGroupNeuralNetworkLayer):
    """NeuralNetwork embedding object that can show probability distributions"""

    def __init__(self, point_radius=0.02):
        super(EmbeddingLayer, self).__init__()
        self.point_radius = point_radius
        self.axes = Axes(
            tips=False,
            x_length=1,
            y_length=1
        )
        self.add(self.axes)
        # Make point cloud
        mean = np.array([0, 0])
        covariance = np.array([[1.5, 0], [0, 1.5]])
        self.point_cloud = self.construct_gaussian_point_cloud(mean, covariance)
        self.add(self.point_cloud)
        # Make latent distribution
        self.latent_distribution = GaussianDistribution(self.axes, mean=mean, cov=covariance) # Use defaults

    def sample_point_location_from_distribution(self):
        """Samples from the current latent distribution"""
        mean = self.latent_distribution.mean
        cov = self.latent_distribution.cov
        point = np.random.multivariate_normal(mean, cov)
        # Make dot at correct location
        location = self.axes.coords_to_point(point[0], point[1])
        
        return location

    def get_distribution_location(self):
        """Returns mean of latent distribution in axes frame"""
        return self.axes.coords_to_point(self.latent_distribution.mean)

    def construct_gaussian_point_cloud(self, mean, covariance, point_color=BLUE,
                                    num_points=200):
        """Plots points sampled from a Gaussian with the given mean and covariance"""
        # Sample points from a Gaussian
        points = np.random.multivariate_normal(mean, covariance, num_points)
        # Add each point to the axes
        point_dots = VGroup()
        for point in points:
            point_location = self.axes.coords_to_point(*point)
            dot = Dot(point_location, color=point_color, radius=self.point_radius/2) 
            point_dots.add(dot)

        return point_dots

    def make_forward_pass_animation(self):
        """Forward pass animation"""
        # Make ellipse object corresponding to the latent distribution
        self.latent_distribution = GaussianDistribution(self.axes) # Use defaults
        # Create animation
        animations = []
        
        #create_distribution = Create(self.latent_distribution.construct_gaussian_distribution(self.latent_distribution.mean, self.latent_distribution.cov)) #Create(self.latent_distribution)
        create_distribution = Create(self.latent_distribution.ellipses) 
        animations.append(create_distribution)

        animation_group = AnimationGroup(*animations)

        return animation_group

    @override_animation(Create)
    def _create_embedding_layer(self, **kwargs):
        # Plot each point at once
        point_animations = []
        for point in self.point_cloud:
            point_animations.append(GrowFromCenter(point))

        point_animation = AnimationGroup(*point_animations, lag_ratio=1.0, run_time=2.5)

        return point_animation

class FeedForwardToEmbedding(ConnectiveLayer):
    """Feed Forward to Embedding Layer"""

    def __init__(self, input_layer, output_layer, animation_dot_color=RED, dot_radius=0.03):
        super().__init__(input_layer, output_layer)
        self.feed_forward_layer = input_layer
        self.embedding_layer = output_layer
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius

    def make_forward_pass_animation(self, run_time=1.5):
        """Makes dots converge on a specific location"""
        # Find point to converge on by sampling from gaussian distribution
        location = self.embedding_layer.sample_point_location_from_distribution()
        # Set the embedding layer latent distribution
        # Move to location
        animations = []
        # Move the dots to the centers of each of the nodes in the FeedForwardLayer
        dots = []
        for node in self.feed_forward_layer.node_group:
            new_dot = Dot(node.get_center(), radius=self.dot_radius, color=self.animation_dot_color)
            per_node_succession = Succession(
                Create(new_dot),
                new_dot.animate.move_to(location),
            )
            animations.append(per_node_succession)
            dots.append(new_dot)
        self.dots = VGroup(*dots)
        self.add(self.dots)
        # Follow up with remove animations
        remove_animations = []
        for dot in dots:
            remove_animations.append(FadeOut(dot))
        self.remove(self.dots)
        remove_animations = AnimationGroup(*remove_animations, run_time=0.2)
        animations = AnimationGroup(*animations)
        animation_group = Succession(animations, remove_animations, lag_ratio=1.0)

        return animation_group

    @override_animation(Create)
    def _create_embedding_layer(self, **kwargs):
        return AnimationGroup()

class EmbeddingToFeedForward(ConnectiveLayer):
    """Feed Forward to Embedding Layer"""

    def __init__(self, input_layer, output_layer, animation_dot_color=RED, dot_radius=0.03):
        super().__init__(input_layer, output_layer)
        self.feed_forward_layer = output_layer
        self.embedding_layer = input_layer
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius

    def make_forward_pass_animation(self, run_time=1.5):
        """Makes dots diverge from the given location and move the decoder"""
        # Find point to converge on by sampling from gaussian distribution
        location = self.embedding_layer.sample_point_location_from_distribution()
        # Move to location
        animations = []
        # Move the dots to the centers of each of the nodes in the FeedForwardLayer
        dots = []
        for node in self.feed_forward_layer.node_group:
            new_dot = Dot(location, radius=self.dot_radius, color=self.animation_dot_color)
            per_node_succession = Succession(
                Create(new_dot),
                new_dot.animate.move_to(node.get_center()),
            )
            animations.append(per_node_succession)
            dots.append(new_dot)
        # Follow up with remove animations
        remove_animations = []
        for dot in dots:
            remove_animations.append(FadeOut(dot))
        remove_animations = AnimationGroup(*remove_animations, run_time=0.2)
        animations = AnimationGroup(*animations)
        animation_group = Succession(animations, remove_animations, lag_ratio=1.0)

        return animation_group

    @override_animation(Create)
    def _create_embedding_layer(self, **kwargs):
        return AnimationGroup()

class NeuralNetworkEmbeddingTestScene(Scene):

    def construct(self):
        nne = EmbeddingLayer()
        mean = np.array([0, 0])
        cov = np.array([[5.0, 1.0], [0.0, 1.0]])

        point_cloud = nne.construct_gaussian_point_cloud(mean, cov)
        nne.add(point_cloud)

        gaussian = nne.construct_gaussian_distribution(mean, cov)
        nne.add(gaussian)

        self.add(nne)