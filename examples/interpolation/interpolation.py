
"""Visualization of VAE Interpolation"""
from pathlib import Path

from manim import *
import numpy as np
from PIL import Image
from manim_ml.neural_network.layers import EmbeddingLayer
from manim_ml.neural_network.layers import FeedForwardLayer
from manim_ml.neural_network.layers import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

ROOT_DIR = Path(__file__).parents[2]


"""
    The VAE Scene for the twitter video. 
"""
config.pixel_height = 720 
config.pixel_width = 1280 
config.frame_height = 6.0
config.frame_width = 6.0
# Set random seed so point distribution is constant
np.random.seed(1)

class InterpolationScene(MovingCameraScene):
    """Scene object for a Variational Autoencoder and Autoencoder"""

    def construct(self):
        # Set Scene config
        numpy_image = np.asarray(Image.open(ROOT_DIR / 'assets/mnist/digit.jpeg'))
        vae = NeuralNetwork([
            ImageLayer(numpy_image, height=1.4),
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            EmbeddingLayer(dist_theme="ellipse").scale(2),
            FeedForwardLayer(3),
            FeedForwardLayer(5),
            ImageLayer(numpy_image, height=1.4),
        ])

        vae.move_to(ORIGIN)
        vae.encoder.shift(LEFT*0.5)
        vae.decoder.shift(RIGHT*0.5)
        mnist_image_handler = variational_autoencoder.MNISTImageHandler()
        image_pair = mnist_image_handler.image_pairs[3]
        # Make forward pass animation and DO NOT run it
        forward_pass_animation = vae.make_forward_pass_animation(image_pair)
        # Make the interpolation animation
        interpolation_images = mnist_image_handler.interpolation_images
        interpolation_animation = vae.make_interpolation_animation(interpolation_images)
        embedding_zoom_animation = self.camera.auto_zoom([
            vae.embedding, 
            vae.decoder, 
            vae.output_image
        ], margin=0.5)
        # Make animations
        forward_pass_animations = []
        for i in range(7):
            anim = vae.decoder.make_forward_propagation_animation(run_time=0.5)
            forward_pass_animations.append(anim)
        forward_pass_animation_group = AnimationGroup(*forward_pass_animations, lag_ratio=1.0)
        # Make forward pass animations
        self.play(Create(vae), run_time=1.5)
        self.play(FadeOut(vae.encoder), run_time=1.0)
        self.play(embedding_zoom_animation, run_time=1.5)
        interpolation_animation = AnimationGroup(
            forward_pass_animation_group, 
            interpolation_animation
        )
        self.play(interpolation_animation, run_time=9.0)
