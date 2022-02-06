"""Autoencoder Manim Visualizations

In this module I define Manim visualizations for Variational Autoencoders
and Traditional Autoencoders.

"""
from configparser import Interpolation
from typing_extensions import runtime
from manim import *
import pickle
import numpy as np
import neural_network

class VariationalAutoencoder(Group):
    """Variational Autoencoder Manim Visualization"""
    
    def __init__(
        self, encoder_nodes_per_layer=[5, 3], decoder_nodes_per_layer=[3, 5], point_color=BLUE, 
        dot_radius=0.05, ellipse_stroke_width=2.0
    ):
        super(Group, self).__init__()
        self.encoder_nodes_per_layer = encoder_nodes_per_layer
        self.decoder_nodes_per_layer = decoder_nodes_per_layer
        self.point_color = point_color
        self.dot_radius = dot_radius
        self.ellipse_stroke_width = ellipse_stroke_width
        # Make the VMobjects
        self.encoder, self.decoder = self._construct_encoder_and_decoder()
        self.embedding = self._construct_embedding()
        # Setup the relative locations
        self.embedding.move_to(self.encoder)
        self.embedding.shift([1.1 * self.encoder.width, 0, 0])
        self.decoder.move_to(self.embedding)
        self.decoder.shift([self.decoder.width * 1.1, 0, 0])
        # Add the objects to the VAE object
        self.add(self.encoder)
        self.add(self.decoder)
        self.add(self.embedding)

    def _construct_encoder_and_decoder(self):
        """Makes the VAE encoder and decoder"""
        # Make the encoder
        layer_node_count = self.encoder_nodes_per_layer
        encoder = neural_network.NeuralNetwork(layer_node_count, dot_radius=self.dot_radius)
        encoder.scale(1.2)
        # Make the decoder
        layer_node_count = self.decoder_nodes_per_layer
        decoder = neural_network.NeuralNetwork(layer_node_count, dot_radius=self.dot_radius)
        decoder.scale(1.2)

        return encoder, decoder

    def _construct_embedding(self):
        """Makes a Gaussian-like embedding"""
        embedding = VGroup()
        # Sample points from a Gaussian
        num_points = 200
        standard_deviation = [0.7, 0.7]
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

    def _construct_input_output_images(self, image_pair):
        """Places the input and output images for the AE"""
        # Takes the image pair
        # image_pair is assumed to be [2, x, y]
        input_image = image_pair[0][None, :, :]
        recon_image = image_pair[1][None, :, :]
        # Convert images to rgb
        input_image = np.repeat(input_image, 3, axis=0)
        input_image = np.rollaxis(input_image, 0, start=3)
        recon_image = np.repeat(recon_image, 3, axis=0)
        recon_image = np.rollaxis(recon_image, 0, start=3)
        # Make an image objects
        input_image_object = ImageMobject(input_image, image_mode="RGB")
        input_image_object.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        input_image_object.height = 2
        recon_image_object = ImageMobject(recon_image, image_mode="RGB")
        recon_image_object.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        recon_image_object.height = 2

        return input_image_object, recon_image_object

    def make_dot_convergence_animation(self, location, run_time=1.5):
        """Makes dots converge on a specific location"""
        # Move to location
        animations = []
        for dot in self.encoder.dots:
            coords = self.embedding.axes.coords_to_point(*location)
            animations.append(dot.animate.move_to(coords))
        move_animations = AnimationGroup(*animations, run_time=1.5)
        # Follow up with remove animations
        remove_animations = []
        for dot in self.encoder.dots:
            remove_animations.append(FadeOut(dot))
        remove_animations = AnimationGroup(*remove_animations, run_time=0.2)

        animation_group = Succession(move_animations, remove_animations, lag_ratio=1.0)

        return animation_group

    def make_dot_divergence_animation(self, location, run_time=3.0):
        """Makes dots diverge from the given location and move the decoder"""
        animations = []
        for node in self.decoder.layers[0].node_group:
            new_dot = Dot(location, radius=self.dot_radius, color=RED)
            per_node_succession = Succession(
                Create(new_dot),
                new_dot.animate.move_to(node.get_center()),
            )
            animations.append(per_node_succession)

        animation_group = AnimationGroup(*animations)
        return animation_group

    def make_forward_pass_animation(self, image_pair, run_time=1.5):
        """Overriden forward pass animation specific to a VAE"""
        per_unit_runtime = run_time
        # Setup images
        self.input_image, self.output_image = self._construct_input_output_images(image_pair)
        self.input_image.move_to(self.encoder.get_left())
        self.input_image.shift(LEFT)
        self.output_image.move_to(self.decoder.get_right())
        self.output_image.shift(RIGHT * 1.2)
        # Make encoder forward pass
        encoder_forward_pass = self.encoder.make_forward_propagation_animation(run_time=per_unit_runtime)
        # Make red dot in embedding
        mean = [1.0, 1.5]
        mean_point = self.embedding.axes.coords_to_point(*mean)
        std = [0.8, 1.2]
        # Make the dot convergence animation
        dot_convergence_animation = self.make_dot_convergence_animation(mean, run_time=per_unit_runtime)
        encoding_succesion = Succession(
            encoder_forward_pass, 
            dot_convergence_animation
        )
        # Make an ellipse centered at mean_point witAnimationGraph std outline
        center_dot = Dot(mean_point, radius=self.dot_radius, color=GREEN)
        ellipse = Ellipse(width=std[0], height=std[1], color=RED, fill_opacity=0.5, stroke_width=self.ellipse_stroke_width)
        ellipse.move_to(mean_point)
        ellipse_animation = AnimationGroup(
            GrowFromCenter(center_dot), 
            GrowFromCenter(ellipse),
        )
        # Make the dot divergence animation
        dot_divergence_animation = self.make_dot_divergence_animation(mean_point, run_time=per_unit_runtime)
        # Make decoder foward pass
        decoder_forward_pass = self.decoder.make_forward_propagation_animation(run_time=per_unit_runtime)
        # Add the animations to the group
        animation_group = AnimationGroup(
            FadeIn(self.input_image),
            encoding_succesion,
            ellipse_animation,
            dot_divergence_animation,
            decoder_forward_pass,
            FadeIn(self.output_image),
            lag_ratio=1,
        )

        return animation_group


class MNISTImageHandler():
    """Deals with loading serialized VAE mnist images from "autoencoder_models" """

    def __init__(
        self, 
        image_pairs_path="src/autoencoder_models/image_pairs.pkl", 
        interpolations_path="src/autoencoder_models/interpolations.pkl"
    ):
        self.image_pairs_path = image_pairs_path
        self.interpolations_path = interpolations_path

        self.image_pairs = []
        self.interpolations = []

        self.load_serialized_data()

    def load_serialized_data(self):
        with open(self.image_pairs_path, "rb") as f:
            self.image_pairs = pickle.load(f)

        with open(self.interpolations_path, "rb") as f:
            self.interpolations_path = pickle.load(f)

"""
    The VAE Scene for the twitter video. 
"""

config.pixel_height = 720 
config.pixel_width = 1280 
config.frame_height = 10.0
config.frame_width = 10.0
# Set random seed so point distribution is constant
np.random.seed(1)

class VAEScene(Scene):
    """Scene object for a Variational Autoencoder and Autoencoder"""

    def construct(self):
        # Set Scene config
        vae = VariationalAutoencoder()
        mnist_image_handler = MNISTImageHandler()
        image_pair = mnist_image_handler.image_pairs[2]
        vae.move_to(ORIGIN)
        vae.scale(1.2)
        self.add(vae)
        forward_pass_animation = vae.make_forward_pass_animation(image_pair)
        self.play(forward_pass_animation)
        """
        autoencoder = Autoencoder()
        autoencoder.move_to(ORIGIN)
        # Make a forward pass animation
        self.add(autoencoder)
        forward_pass_animation = autoencoder.make_forward_pass_animation(run_time=1.5)
        self.play(forward_pass_animation)
        """