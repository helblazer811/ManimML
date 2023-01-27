"""Variational Autoencoder Manim Visualizations

In this module I define Manim visualizations for Variational Autoencoders
and Traditional Autoencoders.

"""
from manim import *
import numpy as np
from PIL import Image
from manim_ml.neural_network.layers import FeedForwardLayer, EmbeddingLayer, ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork


class VariationalAutoencoder(VGroup):
    """Variational Autoencoder Manim Visualization"""

    def __init__(
        self,
        encoder_nodes_per_layer=[5, 3],
        decoder_nodes_per_layer=[3, 5],
        point_color=BLUE,
        dot_radius=0.05,
        ellipse_stroke_width=1.0,
        layer_spacing=0.5,
    ):
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

        neural_network = NeuralNetwork(
            [
                FeedForwardLayer(5),
                FeedForwardLayer(3),
                embedding_layer,
                FeedForwardLayer(3),
                FeedForwardLayer(5),
            ]
        )

        return neural_network, embedding_layer

    @override_animation(Create)
    def _create_vae(self):
        return Create(self.neural_network)

    def make_triplet_forward_pass(self, triplet):
        pass

    def make_image_forward_pass(self, input_image, output_image, run_time=1.5):
        """Override forward pass animation specific to a VAE"""
        # Make a wrapper NN with images
        wrapper_neural_network = NeuralNetwork(
            [ImageLayer(input_image), self.neural_network, ImageLayer(output_image)]
        )
        # Make animation
        animation_group = AnimationGroup(
            Create(wrapper_neural_network),
            wrapper_neural_network.make_forward_pass_animation(),
            lag_ratio=1.0,
        )

        return animation_group
