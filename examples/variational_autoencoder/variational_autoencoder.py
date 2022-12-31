"""Autoencoder Manim Visualizations

In this module I define Manim visualizations for Variational Autoencoders
and Traditional Autoencoders.

"""
from pathlib import Path

from manim import *
import numpy as np
from PIL import Image
from manim_ml.neural_network.layers import EmbeddingLayer
from manim_ml.neural_network.layers import FeedForwardLayer
from manim_ml.neural_network.layers import ImageLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

ROOT_DIR = Path(__file__).parents[2]


<<<<<<< HEAD
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

    def make_reset_vae_animation(self):
        """Resets the VAE to just the neural network"""
        animation_group = AnimationGroup(
            FadeOut(self.input_image),
            FadeOut(self.output_image),
            FadeOut(self.distribution_objects),
            run_time=1.0
        )

        return animation_group
        
    def make_forward_pass_animation(self, image_pair, run_time=1.5, **kwargs):
        """Overriden forward pass animation specific to a VAE"""
        per_unit_runtime = run_time
        # Setup images
        self.input_image, self.output_image = self._construct_input_output_images(image_pair)
        self.input_image.move_to(self.encoder.get_left())
        self.input_image.shift(LEFT)
        self.output_image.move_to(self.decoder.get_right())
        self.output_image.shift(RIGHT*1.3)
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

        return animation_group

    def make_interpolation_animation(self, interpolation_images, frame_rate=5):
        """Makes an animation interpolation"""
        num_images = len(interpolation_images)
        # Make madeup path
        interpolation_latent_path = np.linspace([-0.7, -1.2], [1.2, 1.5], num=num_images)
        # Make the path animation
        first_dot_location = self.embedding.axes.coords_to_point(*interpolation_latent_path[0])
        last_dot_location = self.embedding.axes.coords_to_point(*interpolation_latent_path[-1])
        moving_dot = Dot(first_dot_location, radius=self.dot_radius, color=RED)
        self.add(moving_dot)
        animation_list = [Create(Line(first_dot_location, last_dot_location, color=RED), run_time=0.1*num_images)]
        for image_index in range(num_images - 1):
            next_index = image_index + 1
            # Get path
            next_point = interpolation_latent_path[next_index]
            next_position = self.embedding.axes.coords_to_point(*next_point)
            # Draw path from current point to next point
            move_animation = moving_dot.animate(run_time=0.1*num_images).move_to(next_position)
            animation_list.append(move_animation)

        interpolation_animation = AnimationGroup(*animation_list)
        # Make the images animation
        animation_list = [Wait(0.5)]
        for numpy_image in interpolation_images:
            numpy_image = numpy_image[None, :, :]
            manim_image = self._construct_image_mobject(numpy_image)
            # Move the image to the correct location
            manim_image.move_to(self.output_image)
            # Add the image
            animation_list.append(FadeIn(manim_image, run_time=0.1))
            # Wait
            # animation_list.append(Wait(1 / frame_rate))
            # Remove the image
            # animation_list.append(FadeOut(manim_image, run_time=0.1))
        images_animation = AnimationGroup(*animation_list, lag_ratio=1.0)
        # Combine the two into an AnimationGroup
        animation_group = AnimationGroup(
            interpolation_animation,
            images_animation
        )

        return animation_group
=======
class VAEScene(Scene):
    """Scene object for a Variational Autoencoder and Autoencoder"""
>>>>>>> 0bc3ad561ba224f3d33e9f843665c1d50d64a68b

    def construct(self):

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

        vae.scale(1.3)

        self.play(Create(vae))
        self.play(vae.make_forward_pass_animation(run_time=15))
