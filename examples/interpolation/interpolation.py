"""Visualization of VAE Interpolation"""
import sys
import os

sys.path.append(os.environ["PROJECT_ROOT"])
from manim import *
import pickle
import numpy as np
import manim_ml.neural_network as neural_network
import examples.variational_autoencoder.variational_autoencoder as variational_autoencoder

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
        vae = variational_autoencoder.VariationalAutoencoder(
            dot_radius=0.035, layer_spacing=0.5
        )
        vae.move_to(ORIGIN)
        vae.encoder.shift(LEFT * 0.5)
        vae.decoder.shift(RIGHT * 0.5)
        mnist_image_handler = variational_autoencoder.MNISTImageHandler()
        image_pair = mnist_image_handler.image_pairs[3]
        # Make forward pass animation and DO NOT run it
        forward_pass_animation = vae.make_forward_pass_animation(image_pair)
        # Make the interpolation animation
        interpolation_images = mnist_image_handler.interpolation_images
        interpolation_animation = vae.make_interpolation_animation(interpolation_images)
        embedding_zoom_animation = self.camera.auto_zoom(
            [vae.embedding, vae.decoder, vae.output_image], margin=0.5
        )
        # Make animations
        forward_pass_animations = []
        for i in range(7):
            anim = vae.decoder.make_forward_propagation_animation(run_time=0.5)
            forward_pass_animations.append(anim)
        forward_pass_animation_group = AnimationGroup(
            *forward_pass_animations, lag_ratio=1.0
        )
        # Make forward pass animations
        self.play(Create(vae), run_time=1.5)
        self.play(FadeOut(vae.encoder), run_time=1.0)
        self.play(embedding_zoom_animation, run_time=1.5)
        interpolation_animation = AnimationGroup(
            forward_pass_animation_group, interpolation_animation
        )
        self.play(interpolation_animation, run_time=9.0)
