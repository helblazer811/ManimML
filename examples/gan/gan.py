import random
from pathlib import Path

from PIL import Image
from manim import *
from manim_ml.neural_network.layers.embedding import EmbeddingLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.layers.vector import VectorLayer

from manim_ml.neural_network.neural_network import NeuralNetwork

ROOT_DIR = Path(__file__).parents[2]

config.pixel_height = 1080
config.pixel_width = 1080
config.frame_height = 8.3
config.frame_width = 8.3


class GAN(Mobject):
    """Generative Adversarial Network"""

    def __init__(self):
        super().__init__()
        self.make_entities()
        self.place_entities()
        self.titles = self.make_titles()

    def make_entities(self, image_height=1.2):
        """Makes all of the network entities"""
        # Make the fake image layer
        default_image = Image.open(ROOT_DIR / "assets/gan/fake_image.png")
        numpy_image = np.asarray(default_image)
        self.fake_image_layer = ImageLayer(
            numpy_image, height=image_height, show_image_on_create=False
        )
        # Make the Generator Network
        self.generator = NeuralNetwork(
            [
                EmbeddingLayer(covariance=np.array([[3.0, 0], [0, 3.0]])).scale(1.3),
                FeedForwardLayer(3),
                FeedForwardLayer(5),
                self.fake_image_layer,
            ],
            layer_spacing=0.1,
        )

        self.add(self.generator)
        # Make the Discriminator
        self.discriminator = NeuralNetwork(
            [
                FeedForwardLayer(5),
                FeedForwardLayer(1),
                VectorLayer(1, value_func=lambda: random.uniform(0, 1)),
            ],
            layer_spacing=0.1,
        )
        self.add(self.discriminator)
        # Make Ground Truth Dataset
        default_image = Image.open(ROOT_DIR / "assets/gan/real_image.jpg")
        numpy_image = np.asarray(default_image)
        self.ground_truth_layer = ImageLayer(numpy_image, height=image_height)
        self.add(self.ground_truth_layer)

        self.scale(1)

    def place_entities(self):
        """Positions entities in correct places"""
        # Place relative to generator
        # Place the ground_truth image layer
        self.ground_truth_layer.next_to(self.fake_image_layer, DOWN, 0.8)
        # Group the images
        image_group = Group(self.ground_truth_layer, self.fake_image_layer)
        # Move the discriminator to the right of thee generator
        self.discriminator.next_to(self.generator, RIGHT, 0.2)
        self.discriminator.match_y(image_group)
        # Move the discriminator to the height of the center of the image_group
        # self.discriminator.match_y(image_group)
        # self.ground_truth_layer.next_to(self.fake_image_layer, DOWN, 0.5)

    def make_titles(self):
        """Makes titles for the different entities"""
        titles = VGroup()

        self.ground_truth_layer_title = Text("Real Image").scale(0.3)
        self.ground_truth_layer_title.next_to(self.ground_truth_layer, UP, 0.1)
        self.add(self.ground_truth_layer_title)
        titles.add(self.ground_truth_layer_title)
        self.fake_image_layer_title = Text("Fake Image").scale(0.3)
        self.fake_image_layer_title.next_to(self.fake_image_layer, UP, 0.1)
        self.add(self.fake_image_layer_title)
        titles.add(self.fake_image_layer_title)
        # Overhead title
        overhead_title = Text("Generative Adversarial Network").scale(0.75)
        overhead_title.shift(np.array([0, 3.5, 0]))
        titles.add(overhead_title)
        # Probability title
        self.probability_title = Text("Probability").scale(0.5)
        self.probability_title.move_to(self.discriminator.input_layers[-2])
        self.probability_title.shift(UP)
        self.probability_title.shift(RIGHT * 1.05)
        titles.add(self.probability_title)

        return titles

    def make_highlight_generator_rectangle(self):
        """Returns animation that highlights the generators contents"""
        group = VGroup()

        generator_surrounding_group = Group(self.generator, self.fake_image_layer_title)

        generator_surrounding_rectangle = SurroundingRectangle(
            generator_surrounding_group, buff=0.1, stroke_width=4.0, color="#0FFF50"
        )
        group.add(generator_surrounding_rectangle)
        title = Text("Generator").scale(0.5)
        title.next_to(generator_surrounding_rectangle, UP, 0.2)
        group.add(title)

        return group

    def make_highlight_discriminator_rectangle(self):
        """Makes a rectangle for highlighting the discriminator"""
        discriminator_group = Group(
            self.discriminator,
            self.fake_image_layer,
            self.ground_truth_layer,
            self.fake_image_layer_title,
            self.probability_title,
        )

        group = VGroup()

        discriminator_surrounding_rectangle = SurroundingRectangle(
            discriminator_group, buff=0.05, stroke_width=4.0, color="#0FFF50"
        )
        group.add(discriminator_surrounding_rectangle)
        title = Text("Discriminator").scale(0.5)
        title.next_to(discriminator_surrounding_rectangle, UP, 0.2)
        group.add(title)

        return group

    def make_generator_forward_pass(self):
        """Makes forward pass of the generator"""

        forward_pass = self.generator.make_forward_pass_animation(dist_theme="ellipse")

        return forward_pass

    def make_discriminator_forward_pass(self):
        """Makes forward pass of the discriminator"""

        disc_forward = self.discriminator.make_forward_pass_animation()

        return disc_forward

    @override_animation(Create)
    def _create_override(self):
        """Overrides create"""
        animation_group = AnimationGroup(
            Create(self.generator),
            Create(self.discriminator),
            Create(self.ground_truth_layer),
            Create(self.titles),
        )
        return animation_group


class GANScene(Scene):
    """GAN Scene"""

    def construct(self):
        gan = GAN().scale(1.70)
        gan.move_to(ORIGIN)
        gan.shift(DOWN * 0.35)
        gan.shift(LEFT * 0.1)
        self.play(Create(gan), run_time=3)
        # Highlight generator
        highlight_generator_rectangle = gan.make_highlight_generator_rectangle()
        self.play(Create(highlight_generator_rectangle), run_time=1)
        # Generator forward pass
        gen_forward_pass = gan.make_generator_forward_pass()
        self.play(gen_forward_pass, run_time=5)
        # Fade out generator highlight
        self.play(Uncreate(highlight_generator_rectangle), run_time=1)
        # Highlight discriminator
        highlight_discriminator_rectangle = gan.make_highlight_discriminator_rectangle()
        self.play(Create(highlight_discriminator_rectangle), run_time=1)
        # Discriminator forward pass
        discriminator_forward_pass = gan.make_discriminator_forward_pass()
        self.play(discriminator_forward_pass, run_time=5)
        # Unhighlight discriminator
        self.play(Uncreate(highlight_discriminator_rectangle), run_time=1)
