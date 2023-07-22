"""
    Shows video of diffusion process. 
"""
import manim_ml
from manim import *
from PIL import Image
import os
from diffusers import StableDiffusionPipeline
import numpy as np
import scipy

from manim_ml.diffusion.mcmc import metropolis_hastings_sampler, gaussian_proposal
from manim_ml.diffusion.random_walk import RandomWalk
from manim_ml.utils.mobjects.probability import make_dist_image_mobject_from_samples

def generate_stable_diffusion_images(prompt, num_inference_steps=30):
    """Generates a list of progressively denoised images using Stable Diffusion

    Parameters
    ----------
    num_inference_steps : int, optional
        _description_, by default 30
    """
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
    image_list = [] 
    # Save image callback
    def save_image_callback(step, timestep, latents):
        print("Saving image callback")
        # Decode the latents
        image = pipeline.vae.decode(latents)
        image_list.append(image)

    # Generate the image
    pipeline(prompt, num_inference_steps=num_inference_steps, callback=save_image_callback)

    return image_list

def make_time_schedule_bar(num_time_steps=30):
    """Makes a bar that gets moved back and forth according to the diffusion model time"""
    # Draw time bar with initial value
    time_bar = NumberLine(
        [0, num_time_steps], length=25, stroke_width=10, include_ticks=False, include_numbers=False,
        color=manim_ml.config.color_scheme.secondary_color

    )
    time_bar.shift(4.5 * DOWN)
    current_time = ValueTracker(0.3)
    time_point = time_bar.number_to_point(current_time.get_value())
    time_dot = Dot(time_point, color=manim_ml.config.color_scheme.secondary_color, radius=0.2)
    label_location = time_bar.number_to_point(1.0)
    label_location -= DOWN * 0.1
    label_text = MathTex("t", color=manim_ml.config.color_scheme.text_color).scale(1.5)
    # label_text = Text("time")
    label_text.move_to(time_bar.get_center())
    label_text.shift(DOWN * 0.5)
    # Make an updater for the dot
    def dot_updater(time_dot):
        # Get location on time_bar
        point_loc = time_bar.number_to_point(current_time.get_value())
        time_dot.move_to(point_loc)

    time_dot.add_updater(dot_updater)

    return time_bar, time_dot, label_text, current_time

def make_2d_diffusion_space():
    """Makes a 2D axis where the diffusion random walk happens in. 
    There is also a distribution of points representing the true distribution
    of images. 
    """
    axes_group = Group()
    x_range = [-8, 8]
    y_range = [-8, 8]
    y_length = 13
    x_length = 13
    # Make an axis
    axes = Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=x_length,
        y_length=y_length,
        # x_axis_config={"stroke_opacity": 1.0, "stroke_color": WHITE},
        # y_axis_config={"stroke_opacity": 1.0, "stroke_color": WHITE},
        # tips=True,
    )
    # Make the distribution for the true images as some gaussian mixture in 
    # the bottom right
    gaussian_a = np.random.multivariate_normal(
        mean=[3.0, -3.2],
        cov=[[1.0, 0.0], [0.0, 1.0]],
        size=(100)
    )
    gaussian_b = np.random.multivariate_normal(
        mean=[3.0, 3.0],
        cov=[[1.0, 0.0], [0.0, 1.0]],
        size=(100)
    )
    gaussian_c = np.random.multivariate_normal(
        mean=[0.0, -1.6],
        cov=[[1.0, 0.0], [0.0, 1.0]],
        size=(200)
    )
    all_gaussian_samples = np.concatenate([gaussian_a, gaussian_b, gaussian_c], axis=0)
    print(f"Shape of all gaussian samples: {all_gaussian_samples.shape}")

    image_mobject = make_dist_image_mobject_from_samples(
        all_gaussian_samples,
        xlim=(x_range[0], x_range[1]),
        ylim=(y_range[0], y_range[1])
    )
    image_mobject.scale_to_fit_height(
        axes.get_height()     
    )
    image_mobject.move_to(axes)

    axes_group.add(axes)
    axes_group.add(image_mobject)

    return axes_group

def generate_forward_and_reverse_chains(start_point, target_point, num_inference_steps=30):
    """Here basically we want to generate a forward and reverse chain
    for the diffusion model where the reverse chain starts at the end of the forward
    chain, and the end of the reverse chain ends somewhere within a certain radius of the 
    reverse chain.

    This can be done in a sortof hacky way by doing metropolis hastings from a start point
    to a distribution centered about the start point and vica versa.

    Parameters
    ----------
    start_point : _type_
        _description_
    """
    def start_dist_log_prob_fn(x):
        """Log probability of the start distribution"""
        # Make it a gaussian in top left of the 2D space
        gaussian_pdf = scipy.stats.multivariate_normal(
            mean=target_point,
            cov=[1.0, 1.0]
        ).pdf(x)

        return np.log(gaussian_pdf)
    
    def end_dist_log_prob_fn(x):
        gaussian_pdf = scipy.stats.multivariate_normal(
            mean=start_point,
            cov=[1.0, 1.0]
        ).pdf(x)

        return np.log(gaussian_pdf)

    forward_chain, _, _ = metropolis_hastings_sampler(
        log_prob_fn=start_dist_log_prob_fn, 
        prop_fn=gaussian_proposal,
        iterations=num_inference_steps,
        initial_location=start_point
    )

    end_point = forward_chain[-1]

    reverse_chain, _, _ = metropolis_hastings_sampler(
        log_prob_fn=end_dist_log_prob_fn,
        prop_fn=gaussian_proposal,
        iterations=num_inference_steps,
        initial_location=end_point
    )

    return forward_chain, reverse_chain

# Make the scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 30.0
config.frame_width = 30.0

class DiffusionProcess(Scene):

    def construct(self):
        # Parameters
        num_inference_steps = 50
        image_save_dir = "diffusion_images"
        prompt = "An oil painting of a dragon."
        start_time = 0
        # Compute the stable diffusion images
        if len(os.listdir(image_save_dir)) < num_inference_steps:
            stable_diffusion_images = generate_stable_diffusion_images(
                prompt,
                num_inference_steps=num_inference_steps
            )
            for index, image in enumerate(stable_diffusion_images):
                image.save(f"{image_save_dir}/{index}.png")
        else:
            stable_diffusion_images = [Image.open(f"{image_save_dir}/{i}.png") for i in range(num_inference_steps)]
        # Reverse order of list to be in line with theory timesteps
        stable_diffusion_images = stable_diffusion_images[::-1]
        # Add the initial location of the first image
        start_image = stable_diffusion_images[start_time]
        image_mobject = ImageMobject(start_image)
        image_mobject.scale(0.55)
        image_mobject.shift(LEFT * 7.5)
        self.add(image_mobject)
        # Make the time schedule bar
        time_bar, time_dot, time_label, current_time = make_time_schedule_bar(num_time_steps=num_inference_steps)
        # Place the bar at the bottom of the screen
        time_bar.move_to(image_mobject.get_bottom() + DOWN * 2)
        time_bar.set_x(0)
        self.add(time_bar)
        # Place the time label below the time bar
        self.add(time_label)
        time_label.next_to(time_bar, DOWN, buff=0.5)
        # Add 0 and T labels above the bar
        zero_label = Text("0")
        zero_label.next_to(time_bar.get_left(), UP, buff=0.5)
        self.add(zero_label)
        t_label = Text("T")
        t_label.next_to(time_bar.get_right(), UP, buff=0.5)
        self.add(t_label)
        # Move the time dot to zero
        time_dot.set_value(0)
        time_dot.move_to(time_bar.number_to_point(0))
        self.add(time_dot)
        # Add the prompt above the image
        paragraph_prompt = '"An oil painting of\n a flaming dragon."'
        text_prompt = Paragraph(paragraph_prompt, alignment="center", font_size=64)
        text_prompt.next_to(image_mobject, UP, buff=0.6)
        self.add(text_prompt)
        # Generate the chain data
        forward_chain, reverse_chain = generate_forward_and_reverse_chains(
            start_point=[3.0, -3.0],
            target_point=[-7.0, -4.0],
            num_inference_steps=num_inference_steps
        )
        # Make the axes that the distribution and chains go on
        axes, axes_group = make_2d_diffusion_space()
        # axes_group.match_height(image_mobject)
        axes_group.shift(RIGHT * 7.5)
        axes.shift(RIGHT * 7.5)
        self.add(axes_group)
        # Add title below distribution
        title = Text("Distribution of Real Images", font_size=48, color=RED)
        title.move_to(axes_group)
        title.shift(DOWN * 6)
        line = Line(
            title.get_top(),
            axes_group.get_bottom() - 4 * DOWN,
        )
        self.add(title)
        self.add(line)
        # Add a title above the plot
        title = Text("Reverse Diffusion Process", font_size=64)
        title.move_to(axes)
        title.match_y(text_prompt)
        self.add(title)
        # First do the forward noise process of adding noise to an image
        forward_random_walk = RandomWalk(forward_chain, axes=axes)
        # Sync up forward random walk
        synced_animations = []
        forward_random_walk_animation = forward_random_walk.animate_random_walk()
        print(len(forward_random_walk_animation.animations))
        for timestep, transition_animation in enumerate(forward_random_walk_animation.animations):
            # Make an animation moving the time bar
            # time_bar_animation = current_time.animate.set_value(timestep)
            time_bar_animation = current_time.animate.set_value(timestep)
            # Make an animation replacing the current image with the timestep image
            new_image = ImageMobject(stable_diffusion_images[timestep])
            new_image = new_image.set_height(image_mobject.get_height())
            new_image.move_to(image_mobject)
            image_mobject = new_image
            replace_image_animation = AnimationGroup(
                FadeOut(image_mobject),
                FadeIn(new_image),
                lag_ratio=1.0
            )
            # Sync them together
            # synced_animations.append(
            self.play(
                Succession(
                    transition_animation, 
                    time_bar_animation,
                    replace_image_animation,
                    lag_ratio=0.0
                )
            )

        # Fade out the random walk
        self.play(forward_random_walk.fade_out_random_walk())
        # self.play(
        # Succession(
        # *synced_animations,
        # )
        # )
        new_title = Text("Forward Diffusion Process", font_size=64)
        new_title.move_to(title)
        self.play(ReplacementTransform(title, new_title))
        # Second do the reverse noise process of removing noise from the image
        backward_random_walk = RandomWalk(reverse_chain, axes=axes)
        # Sync up forward random walk
        synced_animations = []
        backward_random_walk_animation = backward_random_walk.animate_random_walk()
        for timestep, transition_animation in enumerate(backward_random_walk_animation.animations):
            timestep = num_inference_steps - timestep - 1
            if timestep == num_inference_steps - 1:
                continue
            # Make an animation moving the time bar
            # time_bar_animation = time_dot.animate.set_value(timestep)
            time_bar_animation = current_time.animate.set_value(timestep)
            # Make an animation replacing the current image with the timestep image
            new_image = ImageMobject(stable_diffusion_images[timestep])
            new_image = new_image.set_height(image_mobject.get_height())
            new_image.move_to(image_mobject)
            image_mobject = new_image
            replace_image_animation = AnimationGroup(
                FadeOut(image_mobject),
                FadeIn(new_image),
                lag_ratio=1.0
            )
            # Sync them together
            # synced_animations.append(
            self.play(
                Succession(
                    transition_animation, 
                    time_bar_animation,
                    replace_image_animation,
                    lag_ratio=0.0
                )
            )
        # self.play(
        # Succession(
        # *synced_animations,
        # )
        # )
