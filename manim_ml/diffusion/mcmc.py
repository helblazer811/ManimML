"""
    Tool for animating Markov Chain Monte Carlo simulations in 2D. 
"""
from manim import *
import matplotlib
import matplotlib.pyplot as plt
from manim_ml.utils.mobjects.plotting import convert_matplotlib_figure_to_image_mobject
import numpy as np
import scipy
import scipy.stats
from tqdm import tqdm
import seaborn as sns

from manim_ml.utils.mobjects.probability import GaussianDistribution

######################## MCMC Algorithms #########################

def gaussian_proposal(x, sigma=0.3):
    """
    Gaussian proposal distribution.

    Draw new parameters from Gaussian distribution with
    mean at current position and standard deviation sigma.

    Since the mean is the current position and the standard
    deviation is fixed. This proposal is symmetric so the ratio
    of proposal densities is 1.

    Parameters
    ----------
    x : np.ndarray or list
        point to center proposal around
    sigma : float, optional
        standard deviation of gaussian for proposal, by default 0.1

    Returns
    -------
    np.ndarray
        propossed point
    """
    # Draw x_star
    x_star = x + np.random.randn(len(x)) * sigma
    # proposal ratio factor is 1 since jump is symmetric
    qxx = 1

    return (x_star, qxx)


class MultidimensionalGaussianPosterior:
    """
    N-Dimensional Gaussian distribution with

    mu ~ Normal(0, 10)
    var ~ LogNormal(0, 1.5)

    Prior on mean is U(-500, 500)
    """

    def __init__(self, ndim=2, seed=12345, scale=3, mu=None, var=None):
        """_summary_

        Parameters
        ----------
        ndim : int, optional
            _description_, by default 2
        seed : int, optional
            _description_, by default 12345
        scale : int, optional
            _description_, by default 10
        """
        np.random.seed(seed)
        self.scale = scale

        if var is None:
            self.var = 10 ** (np.random.randn(ndim) * 1.5)
        else:
            self.var = var

        if mu is None:
            self.mu = scipy.stats.norm(loc=0, scale=self.scale).rvs(ndim)
        else:
            self.mu = mu

    def __call__(self, x):
        """
        Call multivariate normal posterior.
        """

        if np.all(x < 500) and np.all(x > -500):
            return scipy.stats.multivariate_normal(mean=self.mu, cov=self.var).logpdf(x)
        else:
            return -1e6

def metropolis_hastings_sampler(
    log_prob_fn=MultidimensionalGaussianPosterior(),
    prop_fn=gaussian_proposal,
    initial_location: np.ndarray = np.array([0, 0]),
    iterations=25,
    warm_up=0,
    ndim=2,
    sampling_seed=1
):
    """Samples using a Metropolis-Hastings sampler.

    Parameters
    ----------
    log_prob_fn : function, optional
        Function to compute log-posterior, by default MultidimensionalGaussianPosterior
    prop_fn : function, optional
        Function to compute proposal location, by default gaussian_proposal
    initial_location : np.ndarray, optional
        initial location for the chain
    iterations : int, optional
        number of iterations of the markov chain, by default 100
    warm_up : int, optional,
        number of warm up iterations

    Returns
    -------
    samples : np.ndarray
        numpy array of 2D samples of length `iterations`
    warm_up_samples : np.ndarray
        numpy array of 2D warm up samples  of length `warm_up`
    candidate_samples: np.ndarray
        numpy array of the candidate samples for each time step
    """
    np.random.seed(sampling_seed)
    # initialize chain, acceptance rate and lnprob
    chain = np.zeros((iterations, ndim))
    proposals = np.zeros((iterations, ndim))
    lnprob = np.zeros(iterations)
    accept_rate = np.zeros(iterations)
    # first samples
    chain[0] = initial_location
    proposals[0] = initial_location
    lnprob0 = log_prob_fn(initial_location)
    lnprob[0] = lnprob0
    # start loop
    x0 = initial_location
    naccept = 0
    for ii in range(1, iterations):
        # propose
        x_star, factor = prop_fn(x0)
        # draw random uniform number
        u = np.random.uniform(0, 1)
        # compute hastings ratio
        lnprob_star = log_prob_fn(x_star)
        H = np.exp(lnprob_star - lnprob0) * factor
        # accept/reject step (update acceptance counter)
        if u < H:
            x0 = x_star
            lnprob0 = lnprob_star
            naccept += 1
        # update chain
        chain[ii] = x0
        proposals[ii] = x_star
        lnprob[ii] = lnprob0
        accept_rate[ii] = naccept / ii

    return chain, np.array([]), proposals

#################### MCMC Visualization Tools ######################

def make_dist_image_mobject_from_samples(samples, ylim, xlim):
    # Make the plot
    matplotlib.use('Agg')
    plt.figure(figsize=(10,10), dpi=100)
    print(np.shape(samples[:, 0]))
    displot = sns.displot(
        x=samples[:, 0], 
        y=samples[:, 1], 
        cmap="Reds", 
        kind="kde",
        norm=matplotlib.colors.LogNorm()
    )
    plt.ylim(ylim[0], ylim[1])
    plt.xlim(xlim[0], xlim[1])
    plt.axis('off')
    fig = displot.fig
    image_mobject = convert_matplotlib_figure_to_image_mobject(fig)

    return image_mobject

class Uncreate(Create):
    def __init__(
        self,
        mobject,
        reverse_rate_function: bool = True,
        introducer: bool = True,
        remover: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            mobject,
            reverse_rate_function=reverse_rate_function,
            introducer=introducer,
            remover=remover,
            **kwargs,
        )

class MCMCAxes(Group):
    """Container object for visualizing MCMC on a 2D axis"""

    def __init__(
        self,
        dot_color=BLUE,
        dot_radius=0.02,
        accept_line_color=GREEN,
        reject_line_color=RED,
        line_color=BLUE,
        line_stroke_width=2,
        x_range=[-3, 3],
        y_range=[-3, 3],
        x_length=5,
        y_length=5
    ):
        super().__init__()
        self.dot_color = dot_color
        self.dot_radius = dot_radius
        self.accept_line_color = accept_line_color
        self.reject_line_color = reject_line_color
        self.line_color = line_color
        self.line_stroke_width = line_stroke_width
        # Make the axes
        self.x_length = x_length
        self.y_length = y_length
        self.x_range = x_range
        self.y_range = y_range
        self.axes = Axes(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
            x_axis_config={"stroke_opacity": 0.0},
            y_axis_config={"stroke_opacity": 0.0},
            tips=False,
        )
        self.add(self.axes)

    @override_animation(Create)
    def _create_override(self, **kwargs):
        """Overrides Create animation"""
        return AnimationGroup(Create(self.axes))

    def visualize_gaussian_proposal_about_point(self, mean, cov=None) -> AnimationGroup:
        """Creates a Gaussian distribution about a certain point

        Parameters
        ----------
        mean : np.ndarray
            mean of proposal distribution
        cov : np.ndarray
            covariance matrix of proposal distribution

        Returns
        -------
        AnimationGroup
            animation of creating the proposal Gaussian distribution
        """
        gaussian = GaussianDistribution(
            axes=self.axes, mean=mean, cov=cov, dist_theme="gaussian"
        )

        create_guassian = Create(gaussian)
        return create_guassian

    def make_transition_animation(
        self, 
        start_point, 
        end_point, 
        candidate_point, 
        show_dots=True,
        run_time=0.1
    ) -> AnimationGroup:
        """Makes an transition animation for a single point on a Markov Chain

        Parameters
        ----------
        start_point: Dot
            Start point of the transition
        end_point : Dot
            End point of the transition
        show_dots: boolean, optional
            Whether or not to show the dots

        Returns
        -------
        AnimationGroup
            Animation of the transition from start to end
        """
        start_location = self.axes.point_to_coords(start_point.get_center())
        end_location = self.axes.point_to_coords(end_point.get_center())
        candidate_location = self.axes.point_to_coords(candidate_point.get_center())
        # Figure out if a point is accepted or rejected
        # point_is_rejected = not candidate_location == end_location
        point_is_rejected = False
        if point_is_rejected:
            return AnimationGroup(), Dot().set_opacity(0.0)
        else:
            create_end_point = Create(end_point)
            line = Line(
                start_point,
                end_point,
                color=self.line_color,
                stroke_width=self.line_stroke_width,
                buff=-0.1
            )

            create_line = Create(line)

            if show_dots:
                return AnimationGroup(
                    create_end_point, 
                    create_line, 
                    lag_ratio=1.0, 
                    run_time=run_time
                ), line
            else:
                return AnimationGroup(
                    create_line, 
                    lag_ratio=1.0, 
                    run_time=run_time
                ), line

    def show_ground_truth_gaussian(self, distribution):
        """ """
        mean = distribution.mu
        var = np.eye(2) * distribution.var
        distribution_drawing = GaussianDistribution(
            self.axes, mean, var, dist_theme="gaussian"
        ).set_opacity(0.2)
        return AnimationGroup(Create(distribution_drawing))

    def visualize_metropolis_hastings_chain_sampling(
        self,
        log_prob_fn=MultidimensionalGaussianPosterior(),
        prop_fn=gaussian_proposal,
        show_dots=False,
        true_samples=None,
        sampling_kwargs={},
    ):
        """
        Makes an animation for visualizing a 2D markov chain using
        metropolis hastings samplings

        Parameters
        ----------
        axes : manim.mobject.graphing.coordinate_systems.Axes
            Manim 2D axes to plot the chain on
        log_prob_fn : function, optional
            Function to compute log-posterior, by default MultidmensionalGaussianPosterior
        prop_fn : function, optional
            Function to compute proposal location, by default gaussian_proposal
        initial_location : list, optional
            initial location for the markov chain, by default None
        show_dots : bool, optional
            whether or not to show the dots on the screen, by default False
        iterations : int, optional
            number of iterations of the markov chain, by default 100

        Returns
        -------
        animation : AnimationGroup
            animation for creating the markov chain
        """
        # Compute the chain samples using a Metropolis Hastings Sampler
        mcmc_samples, warm_up_samples, candidate_samples = metropolis_hastings_sampler(
            log_prob_fn=log_prob_fn, 
            prop_fn=prop_fn,
            **sampling_kwargs
        )
        # print(f"MCMC samples: {mcmc_samples}")
        # print(f"Candidate samples: {candidate_samples}")
        # Make the animation for visualizing the chain
        transition_animations = []
        # Place the initial point
        current_point = mcmc_samples[0]
        current_point = Dot(
            self.axes.coords_to_point(current_point[0], current_point[1]),
            color=self.dot_color,
            radius=self.dot_radius,
        )
        create_initial_point = Create(current_point)
        transition_animations.append(create_initial_point)
        # Show the initial point's proposal distribution
        # NOTE: visualize the warm up and the iterations
        lines = []
        warmup_points = [] 
        num_iterations = len(mcmc_samples) + len(warm_up_samples)
        for iteration in tqdm(range(1, num_iterations)):
            next_sample = mcmc_samples[iteration]
            # print(f"Next sample: {next_sample}")
            candidate_sample = candidate_samples[iteration - 1]
            # Make the next point
            next_point = Dot(
                self.axes.coords_to_point(
                    next_sample[0], 
                    next_sample[1]
                ),
                color=self.dot_color,
                radius=self.dot_radius,
            )
            candidate_point = Dot(
                self.axes.coords_to_point(
                    candidate_sample[0], 
                    candidate_sample[1]
                ),
                color=self.dot_color,
                radius=self.dot_radius,
            )
            # Make a transition animation
            transition_animation, line = self.make_transition_animation(
                current_point, next_point, candidate_point
            )
            # Save assets
            lines.append(line)
            if iteration < len(warm_up_samples):
                warmup_points.append(candidate_point)

            # Add the transition animation
            transition_animations.append(transition_animation)
            # Setup for next iteration
            current_point = next_point
        # Overall MCMC animation
        # 1. Fade in the distribution
        image_mobject = make_dist_image_mobject_from_samples(
            true_samples,
            xlim=(self.x_range[0], self.x_range[1]),
            ylim=(self.y_range[0], self.y_range[1])
        )
        image_mobject.scale_to_fit_height(
            self.y_length     
        )
        image_mobject.move_to(self.axes)
        fade_in_distribution = FadeIn(
            image_mobject,
            run_time=0.5
        )
        # 2. Start sampling the chain
        chain_sampling_animation = AnimationGroup(
            *transition_animations, 
            lag_ratio=1.0,
            run_time=5.0
        )
        # 3. Convert the chain to points, excluding the warmup
        lines = VGroup(*lines)
        warm_up_points = VGroup(*warmup_points)
        fade_out_lines_and_warmup = AnimationGroup(
            Uncreate(lines), 
            Uncreate(warm_up_points),
            lag_ratio=0.0
        )
        # Make the final animation
        animation_group = Succession(
            fade_in_distribution,
            chain_sampling_animation,
            fade_out_lines_and_warmup,
            lag_ratio=1.0
        )

        return animation_group
