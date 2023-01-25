from manim import *
from manim_ml.probability import GaussianDistribution
from manim_ml.neural_network.layers.parent_layers import VGroupNeuralNetworkLayer


class EmbeddingLayer(VGroupNeuralNetworkLayer):
    """NeuralNetwork embedding object that can show probability distributions"""

    def __init__(
        self,
        point_radius=0.02,
        mean=np.array([0, 0]),
        covariance=np.array([[1.0, 0], [0, 1.0]]),
        dist_theme="gaussian",
        paired_query_mode=False,
        **kwargs
    ):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        self.mean = mean
        self.covariance = covariance
        self.gaussian_distributions = VGroup()
        self.add(self.gaussian_distributions)
        self.point_radius = point_radius
        self.dist_theme = dist_theme
        self.paired_query_mode = paired_query_mode

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        self.axes = Axes(
            tips=False,
            x_length=0.8,
            y_length=0.8,
            x_range=(-1.4, 1.4),
            y_range=(-1.8, 1.8),
            x_axis_config={"include_ticks": False, "stroke_width": 0.0},
            y_axis_config={"include_ticks": False, "stroke_width": 0.0},
        )
        self.add(self.axes)
        self.axes.move_to(self.get_center())
        # Make point cloud
        self.point_cloud = self.construct_gaussian_point_cloud(
            self.mean, self.covariance
        )
        self.add(self.point_cloud)
        # Make latent distribution
        self.latent_distribution = GaussianDistribution(
            self.axes, mean=self.mean, cov=self.covariance
        )  # Use defaults

    def add_gaussian_distribution(self, gaussian_distribution):
        """Adds given GaussianDistribution to the list"""
        self.gaussian_distributions.add(gaussian_distribution)

        return Create(gaussian_distribution)

    def remove_gaussian_distribution(self, gaussian_distribution):
        """Removes the given gaussian distribution from the embedding"""
        for gaussian in self.gaussian_distributions:
            if gaussian == gaussian_distribution:
                self.gaussian_distributions.remove(gaussian_distribution)
                return FadeOut(gaussian)

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

    def construct_gaussian_point_cloud(
        self, mean, covariance, point_color=WHITE, num_points=400
    ):
        """Plots points sampled from a Gaussian with the given mean and covariance"""
        # Sample points from a Gaussian
        np.random.seed(5)
        points = np.random.multivariate_normal(mean, covariance, num_points)
        # Add each point to the axes
        point_dots = VGroup()
        for point in points:
            point_location = self.axes.coords_to_point(*point)
            dot = Dot(point_location, color=point_color, radius=self.point_radius / 2)
            dot.set_z_index(-1)
            point_dots.add(dot)

        return point_dots

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        """Forward pass animation"""
        animations = []
        if "triplet_args" in layer_args:
            triplet_args = layer_args["triplet_args"]
            positive_dist_args = triplet_args["positive_dist"]
            negative_dist_args = triplet_args["negative_dist"]
            anchor_dist_args = triplet_args["anchor_dist"]
            # Create each dist
            anchor_dist = GaussianDistribution(self.axes, **anchor_dist_args)
            animations.append(Create(anchor_dist))

            positive_dist = GaussianDistribution(self.axes, **positive_dist_args)
            animations.append(Create(positive_dist))

            negative_dist = GaussianDistribution(self.axes, **negative_dist_args)
            animations.append(Create(negative_dist))
            # Draw edges in between anchor and positive, anchor and negative
            anchor_positive = Line(
                anchor_dist.get_center(),
                positive_dist.get_center(),
                color=GOLD,
                stroke_width=DEFAULT_STROKE_WIDTH / 2,
            )
            anchor_positive.set_z_index(3)
            animations.append(Create(anchor_positive))

            anchor_negative = Line(
                anchor_dist.get_center(),
                negative_dist.get_center(),
                color=GOLD,
                stroke_width=DEFAULT_STROKE_WIDTH / 2,
            )
            anchor_negative.set_z_index(3)

            animations.append(Create(anchor_negative))
        elif not self.paired_query_mode:
            # Normal embedding mode
            if "dist_args" in layer_args:
                scale_factor = 1.0
                if "scale_factor" in layer_args:
                    scale_factor = layer_args["scale_factor"]
                self.latent_distribution = GaussianDistribution(
                    self.axes, **layer_args["dist_args"]
                ).scale(scale_factor)
            else:
                # Make ellipse object corresponding to the latent distribution
                # self.latent_distribution = GaussianDistribution(
                #     self.axes,
                #     dist_theme=self.dist_theme,
                #     cov=np.array([[0.8, 0], [0.0, 0.8]])
                # )
                pass
            # Create animation
            create_distribution = Create(self.latent_distribution)
            animations.append(create_distribution)
        else:
            # Paired Query Mode
            assert "positive_dist_args" in layer_args
            assert "negative_dist_args" in layer_args
            positive_dist_args = layer_args["positive_dist_args"]
            negative_dist_args = layer_args["negative_dist_args"]
            # Handle logic for embedding a paired query into the embedding layer
            positive_dist = GaussianDistribution(self.axes, **positive_dist_args)
            self.gaussian_distributions.add(positive_dist)
            negative_dist = GaussianDistribution(self.axes, **negative_dist_args)
            self.gaussian_distributions.add(negative_dist)

            animations.append(Create(positive_dist))
            animations.append(Create(negative_dist))

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
