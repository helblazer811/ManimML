from manim import *
from manim_ml.diffusion.mcmc import MCMCAxes, MultidimensionalGaussianPosterior, metropolis_hastings_sampler
# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1200
config.frame_height = 12.0
config.frame_width = 12.0

def test_metropolis_hastings_sampler(iterations=100):
    samples, _, candidates = metropolis_hastings_sampler(iterations=iterations)
    assert samples.shape == (iterations, 2)

class MCMCTest(Scene):

    def construct(self):
        axes = MCMCAxes()
        self.play(Create(axes))
        gaussian_posterior = MultidimensionalGaussianPosterior(
            mu=np.array([0.0, 0.0]),
            var=np.array([4.0, 2.0])
        )
        show_gaussian_animation = axes.show_ground_truth_gaussian(
            gaussian_posterior
        )
        self.play(show_gaussian_animation)
        chain_sampling_animation = axes.visualize_metropolis_hastings_chain_sampling(
            log_prob_fn=gaussian_posterior,
            sampling_kwargs={"iterations": 1000}
        )

        self.play(chain_sampling_animation)
