from manim import *
from manim_ml.diffusion.mcmc import (
    MCMCAxes,
    MultidimensionalGaussianPosterior,
    metropolis_hastings_sampler,
)
from manim_ml.utils.mobjects.plotting import convert_matplotlib_figure_to_image_mobject

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
plt.style.use('dark_background')

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1200
config.frame_height = 7.0
config.frame_width = 7.0

def test_metropolis_hastings_sampler(iterations=100):
    samples, _, candidates = metropolis_hastings_sampler(iterations=iterations)
    assert samples.shape == (iterations, 2)

def plot_hexbin_gaussian_on_image_mobject(
    sample_func, 
    xlim=(-4, 4),
    ylim=(-4, 4)
):
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    n = 100_000
    samples = []
    for i in range(n):
        samples.append(sample_func())
    samples = np.array(samples)

    x = samples[:, 0]
    y = samples[:, 1]
    
    fig, ax0 = plt.subplots(1, figsize=(5, 5))

    hb = ax0.hexbin(x, y, gridsize=50, cmap='gist_heat')

    ax0.set(xlim=xlim, ylim=ylim)

    return convert_matplotlib_figure_to_image_mobject(fig)

class MCMCTest(Scene):

    def construct(
        self, 
        mu=np.array([0.0, 0.0]), 
        var=np.array([[1.0, 1.0]])
    ):

        def gaussian_sample_func():
            vals = np.random.multivariate_normal(
                mu, 
                np.eye(2) * var, 
                1
            )[0]

            return vals

        image_mobject = plot_hexbin_gaussian_on_image_mobject(
            gaussian_sample_func
        )
        self.add(image_mobject)
        self.play(FadeOut(image_mobject))

        axes = MCMCAxes(
            x_range=[-4, 4],
            y_range=[-4, 4],
        )
        self.play(
            Create(axes)
        )

        gaussian_posterior = MultidimensionalGaussianPosterior(
            mu=np.array([0.0, 0.0]), 
            var=np.array([1.0, 1.0])
        )

        chain_sampling_animation, lines = axes.visualize_metropolis_hastings_chain_sampling(
            log_prob_fn=gaussian_posterior, 
            sampling_kwargs={"iterations": 500},
        )

        self.play(
            chain_sampling_animation,
            run_time=3.5
        )
        self.play(
            FadeOut(lines)
        )
        self.wait(1)
        self.play(
            FadeIn(image_mobject)
        )


