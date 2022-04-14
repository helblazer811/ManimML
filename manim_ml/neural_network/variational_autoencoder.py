"""Variational Autoencoder Manim Visualizations

In this module I define Manim visualizations for Variational Autoencoders
and Traditional Autoencoders.

"""
from manim import *
import numpy as np
from PIL import Image
import os
from manim_ml.neural_network.feed_forward import FeedForwardLayer
from manim_ml.neural_network.image import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork
from manim_ml.neural_network.embedding import EmbeddingLayer

class VariationalAutoencoder(VGroup):
    """Variational Autoencoder Manim Visualization"""
    
    def __init__(self, encoder_nodes_per_layer=[5, 3], decoder_nodes_per_layer=[3, 5], 
                point_color=BLUE, dot_radius=0.05, ellipse_stroke_width=1.0, 
                layer_spacing=0.5):
        super(VGroup, self).__init__()
        self.encoder_nodes_per_layer = encoder_nodes_per_layer
        self.decoder_nodes_per_layer = decoder_nodes_per_layer
        self.point_color = point_color
        self.dot_radius = dot_radius
        self.layer_spacing = layer_spacing
        self.ellipse_stroke_width = ellipse_stroke_width
        # Make the VMobjects
        self.neural_network, self.embedding_layer = self._construct_neural_network()

    def _construct_neural_network(self):
        """Makes the VAE encoder, embedding layer, and decoder"""
        embedding_layer = EmbeddingLayer()

        neural_network = NeuralNetwork([
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            embedding_layer,
            FeedForwardLayer(3),
            FeedForwardLayer(5)
        ])

        return neural_network, embedding_layer

    @override_animation(Create)
    def _create_vae(self):
        return Create(self.neural_network)

    def make_triplet_forward_pass(self, triplet):
        pass
        
    def make_image_forward_pass(self, input_image, output_image, run_time=1.5):
        """Override forward pass animation specific to a VAE"""
        # Make a wrapper NN with images
        wrapper_neural_network = NeuralNetwork([
            ImageLayer(input_image),
            self.neural_network,
            ImageLayer(output_image)
        ])
        # Make animation 
        animation_group = AnimationGroup(
            Create(wrapper_neural_network),
            wrapper_neural_network.make_forward_pass_animation(),
            lag_ratio=1.0
        )

        return animation_group

        """
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
        center_dot = Dot(mean_point, radius=self.dot_radius, color=RED)
        ellipse = Ellipse(width=std[0], height=std[1], color=RED, fill_opacity=0.3, stroke_width=self.ellipse_stroke_width)
        ellipse.move_to(mean_point)
        self.distribution_objects = VGroup(
            center_dot, 
            ellipse
        )
        # Make ellipse animation
        ellipse_animation = AnimationGroup(
            GrowFromCenter(center_dot), 
            GrowFromCenter(ellipse),
        )
        # Make the dot divergence animation
        sampled_point = [0.51, 1.0]
        divergence_point = self.embedding.axes.coords_to_point(*sampled_point)
        dot_divergence_animation = self.make_dot_divergence_animation(divergence_point, run_time=per_unit_runtime)
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
        """