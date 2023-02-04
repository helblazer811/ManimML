from manim import *

import scipy.stats
from manim_ml.diffusion.mcmc import MCMCAxes
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

# Make the specific scene
config.pixel_height = 720
config.pixel_width = 720
config.frame_height = 7.0
config.frame_width = 7.0

class MCMCWarmupScene(Scene):

    def construct(self):
        # Define the Gaussian Mixture likelihood
        def gaussian_mm_logpdf(x):
            """Gaussian Mixture Model Log PDF"""
            # Choose two arbitrary Gaussians
            # Big Gaussian
            big_gaussian_pdf = scipy.stats.multivariate_normal(
                mean=[-0.5, -0.5],
                cov=[1.0, 1.0]
            ).pdf(x)
            # Little Gaussian
            little_gaussian_pdf = scipy.stats.multivariate_normal(
                mean=[2.3, 1.9],
                cov=[0.3, 0.3]
            ).pdf(x)
            # Sum their likelihoods and take the log
            logpdf = np.log(big_gaussian_pdf + little_gaussian_pdf)

            return logpdf

        # Generate a bunch of true samples
        true_samples = []
        # Generate samples for little gaussian
        little_gaussian_samples = np.random.multivariate_normal(
            mean=[2.3, 1.9],
            cov=[[0.3, 0.0], [0.0, 0.3]],
            size=(10000)
        )
        big_gaussian_samples = np.random.multivariate_normal(
            mean=[-0.5, -0.5],
            cov=[[1.0, 0.0], [0.0, 1.0]],
            size=(10000)
        )
        true_samples = np.concatenate((little_gaussian_samples, big_gaussian_samples))
        # Make the MCMC axes
        axes = MCMCAxes(
            x_range=[-5, 5],
            y_range=[-5, 5],
            x_length=7.0,
            y_length=7.0
        )
        axes.move_to(ORIGIN)
        self.play(
            Create(axes)
        )
        # Make the chain sampling animation 
        chain_sampling_animation = axes.visualize_metropolis_hastings_chain_sampling(
            log_prob_fn=gaussian_mm_logpdf, 
            true_samples=true_samples,
            sampling_kwargs={
                "iterations": 2000,
                "warm_up": 50,
                "initial_location": np.array([-3.5, 3.5]),
                "sampling_seed": 4
            },
        )
        self.play(chain_sampling_animation)
        self.wait(3)
