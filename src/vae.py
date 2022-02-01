"""Autoencoder Manim Visualizations

In this module I define Manim visualizations for Variational Autoencoders
and Traditional Autoencoders.

"""
from manim import *
import numpy as np
import neural_network

class Autoencoder(VGroup):
    """Traditional Autoencoder Manim Visualization"""

    def __init__(self, encoder_nodes_per_layer=[6, 4], decoder_nodes_per_layer=[4, 6], point_color=BLUE):
        super(VGroup, self).__init__()
        self.encoder_nodes_per_layer = encoder_nodes_per_layer
        self.decoder_nodes_per_layer = decoder_nodes_per_layer
        self.point_color = point_color
        # Make the VMobjects
        self.encoder, self.decoder = self._construct_encoder_and_decoder()
        self.embedding = self._construct_embedding()
        # self.input_image, self.output_image = self._construct_input_output_images()
        # Setup the relative locations
        self.embedding.move_to(self.encoder)
        self.embedding.shift([0.9 * self.embedding.width, 0, 0])
        self.decoder.move_to(self.embedding)
        self.decoder.shift([self.embedding.width * 0.9, 0, 0])
        # self.embedding.shift(self.encoder.width * 1.5)
        # self.decoder.move_to(self.embedding.get_center())
        # Add the objects to the VAE object
        self.add(self.encoder)
        self.add(self.decoder)
        self.add(self.embedding)
        # self.add(self.input_image)
        # self.add(self.output_image)

    def _construct_encoder_and_decoder(self):
        """Makes the VAE encoder and decoder"""
        # Make the encoder
        layer_node_count = self.encoder_nodes_per_layer
        encoder = neural_network.NeuralNetwork(layer_node_count)
        # Make the decoder
        layer_node_count = self.decoder_nodes_per_layer
        decoder = neural_network.NeuralNetwork(layer_node_count)

        return encoder, decoder

    def _construct_embedding(self):
        """Makes a Gaussian-like embedding"""
        embedding = VGroup()
        # Sample points from a Gaussian
        num_points = 200
        standard_deviation = [1, 1]
        mean = [0, 0]
        points = np.random.normal(mean, standard_deviation, size=(num_points, 2))
        # Make an axes
        embedding.axes = Axes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            x_length = 3,
            y_length = 3,
            tips=False,
        )
        # Add each point to the axes
        point_dots = VGroup()
        for point in points:
            point_location = embedding.axes.coords_to_point(*point)
            dot = Dot(point_location, color=self.point_color) 
            point_dots.add(dot)

        embedding.add(point_dots)
        return embedding

    def _construct_input_output_images(self):
        pass

    def make_embedding_generation_animation(self):
        """Animates the embedding getting created"""
        pass

    def make_forward_pass_animation(self, run_time=2):
        """Makes an animation of a forward pass throgh the VAE"""
        per_unit_runtime = run_time // 3
        # Make encoder forward pass
        encoder_forward_pass = self.encoder.make_forward_propagation_animation(run_time=per_unit_runtime)
        # Make red dot in embedding
        location = np.random.normal(0, 1, (2))
        location_point = self.embedding.axes.coords_to_point(*location)
        dot = Dot(location_point, color=RED)
        create_dot_animation = Create(dot, run_time=per_unit_runtime)
        # Make decoder foward pass
        decoder_forward_pass = self.decoder.make_forward_propagation_animation(run_time=per_unit_runtime)
        # Add the animations to the group
        animation_group = AnimationGroup(
            encoder_forward_pass,
            create_dot_animation,
            decoder_forward_pass,
            lag_ratio=1
        )

        return animation_group

    def make_interpolation_animation(self):
        """Makes animation of interpolating in the latent space"""
        pass

class VariationalAutoencoder(Autoencoder):
    """Variational Autoencoder Manim Visualization"""
    
    def __init__(self):
        super(self, Autoencoder).__init__()

    def make_forward_pass_animation(self):
        """Overriden forward pass animation specific to a VAE"""
        return super().make_forward_pass_animation()

class VAEScene(Scene):
    """Scene object for a Variational Autoencoder and Autoencoder"""

    def construct(self):
        autoencoder = Autoencoder()
        autoencoder.move_to(ORIGIN)
        # Make a forward pass animation
        self.add(autoencoder)
        forward_pass_animation = autoencoder.make_forward_pass_animation(run_time=1.5)
        self.play(forward_pass_animation)