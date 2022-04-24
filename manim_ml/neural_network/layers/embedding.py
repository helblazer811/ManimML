from manim import *
from manim_ml.probability import GaussianDistribution
from manim_ml.neural_network.layers.parent_layers import VGroupNeuralNetworkLayer

class EmbeddingLayer(VGroupNeuralNetworkLayer):
    """NeuralNetwork embedding object that can show probability distributions"""

    def __init__(self, point_radius=0.02, mean = np.array([0, 0]), 
                covariance=np.array([[1.5, 0], [0, 1.5]]), dist_theme="gaussian", **kwargs):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        self.point_radius = point_radius
        self.dist_theme = dist_theme
        self.axes = Axes(
            tips=False,
            x_length=0.8,
            y_length=0.8
        )
        self.add(self.axes)
        # Make point cloud
        self.point_cloud = self.construct_gaussian_point_cloud(mean, covariance)
        self.add(self.point_cloud)
        # Make latent distribution
        self.latent_distribution = GaussianDistribution(self.axes, mean=mean, cov=covariance,
                                                        dist_theme=self.dist_theme) # Use defaults

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

    def make_forward_pass_animation(self, **kwargs):
        """Forward pass animation"""
        # Make ellipse object corresponding to the latent distribution
        self.latent_distribution = GaussianDistribution(
            self.axes, 
            dist_theme=self.dist_theme, 
            cov=np.array([[0.8, 0], [0.0, 0.8]])
        ) # Use defaults
        # Create animation
        animations = []
        #create_distribution = Create(self.latent_distribution.construct_gaussian_distribution(self.latent_distribution.mean, self.latent_distribution.cov)) #Create(self.latent_distribution)
        create_distribution = Create(self.latent_distribution.ellipses) 
        animations.append(create_distribution)

        animation_group = AnimationGroup(*animations)

        return animation_group

    @override_animation(Create)
    def _create_override(self, **kwargs):
        # Plot each point at once
        point_animations = []
        for point in self.point_cloud:
            point_animations.append(GrowFromCenter(point))

        point_animation = AnimationGroup(*point_animations, lag_ratio=1.0, run_time=2.5)

        return point_animation

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