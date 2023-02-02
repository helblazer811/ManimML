"""
    Here is a animated explanatory figure for the "Oracle Guided Image Synthesis with Relative Queries" paper. 
"""
from pathlib import Path

from manim import *
from manim_ml.neural_network.layers import triplet
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.layers.paired_query import PairedQueryLayer
from manim_ml.neural_network.layers.triplet import TripletLayer
from manim_ml.neural_network.neural_network import NeuralNetwork
from manim_ml.neural_network.layers import FeedForwardLayer, EmbeddingLayer
from manim_ml.neural_network.layers.util import get_connective_layer
import os

from manim_ml.utils.mobjects.probability import GaussianDistribution

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 6.0
config.frame_width = 6.0

ROOT_DIR = Path(__file__).parents[3]


class Localizer:
    """
    Holds the localizer object, which contains the queries, images, etc.
    needed to represent a localization run.
    """

    def __init__(self, axes):
        # Set dummy values for these
        self.index = -1
        self.axes = axes
        self.num_queries = 3
        self.assets_path = ROOT_DIR / "assets/oracle_guidance"
        self.ground_truth_image_path = self.assets_path / "ground_truth.jpg"
        self.ground_truth_location = np.array([2, 3])
        # Prior distribution
        print("initial gaussian")
        self.prior_distribution = GaussianDistribution(
            self.axes,
            mean=np.array([0.0, 0.0]),
            cov=np.array([[3, 0], [0, 3]]),
            dist_theme="ellipse",
            color=GREEN,
        )
        # Define the query images and embedded locations
        # Contains image paths [(positive_path, negative_path), ...]
        self.query_image_paths = [
            (
                os.path.join(self.assets_path, "positive_1.jpg"),
                os.path.join(self.assets_path, "negative_1.jpg"),
            ),
            (
                os.path.join(self.assets_path, "positive_2.jpg"),
                os.path.join(self.assets_path, "negative_2.jpg"),
            ),
            (
                os.path.join(self.assets_path, "positive_3.jpg"),
                os.path.join(self.assets_path, "negative_3.jpg"),
            ),
        ]
        # Contains 2D locations for each image [([2, 3], [2, 4]), ...]
        self.query_locations = [
            (np.array([-1, -1]), np.array([1, 1])),
            (np.array([1, -1]), np.array([-1, 1])),
            (np.array([0.3, -0.6]), np.array([-0.5, 0.7])),
        ]
        # Make the covariances for each query
        self.query_covariances = [
            (np.array([[0.3, 0], [0.0, 0.2]]), np.array([[0.2, 0], [0.0, 0.2]])),
            (np.array([[0.2, 0], [0.0, 0.2]]), np.array([[0.2, 0], [0.0, 0.2]])),
            (np.array([[0.2, 0], [0.0, 0.2]]), np.array([[0.2, 0], [0.0, 0.2]])),
        ]
        # Posterior distributions over time GaussianDistribution objects
        self.posterior_distributions = [
            GaussianDistribution(
                self.axes,
                dist_theme="ellipse",
                color=GREEN,
                mean=np.array([-0.3, -0.3]),
                cov=np.array([[5, -4], [-4, 6]]),
            ).scale(0.6),
            GaussianDistribution(
                self.axes,
                dist_theme="ellipse",
                color=GREEN,
                mean=np.array([0.25, -0.25]),
                cov=np.array([[3, -2], [-2, 4]]),
            ).scale(0.35),
            GaussianDistribution(
                self.axes,
                dist_theme="ellipse",
                color=GREEN,
                mean=np.array([0.4, -0.35]),
                cov=np.array([[1, 0], [0, 1]]),
            ).scale(0.3),
        ]
        # Some assumptions
        assert len(self.query_locations) == len(self.query_image_paths)
        assert len(self.query_locations) == len(self.posterior_distributions)

    def __iter__(self):
        return self

    def __next__(self):
        """Steps through each localization time instance"""
        if self.index < len(self.query_image_paths):
            self.index += 1
        else:
            raise StopIteration

        # Return query_paths, query_locations, posterior
        out_tuple = (
            self.query_image_paths[self.index],
            self.query_locations[self.index],
            self.posterior_distributions[self.index],
            self.query_covariances[self.index],
        )

        return out_tuple


class OracleGuidanceVisualization(Scene):
    def __init__(self):
        super().__init__()
        self.neural_network, self.embedding_layer = self.make_vae()
        self.localizer = iter(Localizer(self.embedding_layer.axes))
        self.subtitle = None
        self.title = None
        # Set image paths
        # VAE embedding animation image paths
        self.assets_path = ROOT_DIR / "assets/oracle_guidance"
        self.input_embed_image_path = os.path.join(self.assets_path, "input_image.jpg")
        self.output_embed_image_path = os.path.join(
            self.assets_path, "output_image.jpg"
        )

    def make_vae(self):
        """Makes a simple VAE architecture"""
        embedding_layer = EmbeddingLayer(dist_theme="ellipse")
        self.encoder = NeuralNetwork(
            [
                FeedForwardLayer(5),
                FeedForwardLayer(3),
                embedding_layer,
            ]
        )

        self.decoder = NeuralNetwork(
            [
                FeedForwardLayer(3),
                FeedForwardLayer(5),
            ]
        )

        neural_network = NeuralNetwork([self.encoder, self.decoder])

        neural_network.shift(DOWN * 0.4)
        return neural_network, embedding_layer

    @override_animation(Create)
    def _create_animation(self):
        animation_group = AnimationGroup(Create(self.neural_network))

        return animation_group

    def insert_at_start(self, layer, create=True):
        """Inserts a layer at the beggining of the network"""
        # Note: Does not move the rest of the network
        current_first_layer = self.encoder.all_layers[0]
        # Get connective layer
        connective_layer = get_connective_layer(layer, current_first_layer)
        # Insert both layers
        self.encoder.all_layers.insert(0, layer)
        self.encoder.all_layers.insert(1, connective_layer)
        # Move layers to the correct location
        # TODO: Fix this cause its hacky
        layer.shift(DOWN * 0.4)
        layer.shift(LEFT * 2.35)
        # Make insert animation
        if not create:
            animation_group = AnimationGroup(Create(connective_layer))
        else:
            animation_group = AnimationGroup(Create(layer), Create(connective_layer))
        self.play(animation_group)

    def remove_start_layer(self):
        """Removes the first layer of the network"""
        first_layer = self.encoder.all_layers.remove_at_index(0)
        first_connective = self.encoder.all_layers.remove_at_index(0)
        # Make remove animations
        animation_group = AnimationGroup(
            FadeOut(first_layer), FadeOut(first_connective)
        )

        self.play(animation_group)

    def insert_at_end(self, layer):
        """Inserts the given layer at the end of the network"""
        current_last_layer = self.decoder.all_layers[-1]
        # Get connective layer
        connective_layer = get_connective_layer(current_last_layer, layer)
        # Insert both layers
        self.decoder.all_layers.add(connective_layer)
        self.decoder.all_layers.add(layer)
        # Move layers to the correct location
        # TODO: Fix this cause its hacky
        layer.shift(DOWN * 0.4)
        layer.shift(RIGHT * 2.35)
        # Make insert animation
        animation_group = AnimationGroup(Create(layer), Create(connective_layer))
        self.play(animation_group)

    def remove_end_layer(self):
        """Removes the lsat layer of the network"""
        first_layer = self.decoder.all_layers.remove_at_index(-1)
        first_connective = self.decoder.all_layers.remove_at_index(-1)
        # Make remove animations
        animation_group = AnimationGroup(
            FadeOut(first_layer), FadeOut(first_connective)
        )

        self.play(animation_group)

    def change_title(self, text, title_location=np.array([0, 1.25, 0]), font_size=24):
        """Changes title to the given text"""
        if self.title is None:
            self.title = Text(text, font_size=font_size)
            self.title.move_to(title_location)
            self.add(self.title)
            self.play(Write(self.title), run_time=1)
            self.wait(1)
            return

        self.play(Unwrite(self.title))
        new_title = Text(text, font_size=font_size)
        new_title.move_to(self.title)
        self.title = new_title
        self.wait(0.1)
        self.play(Write(self.title))

    def change_subtitle(self, text, title_location=np.array([0, 0.9, 0]), font_size=20):
        """Changes subtitle to the next algorithm step"""
        if self.subtitle is None:
            self.subtitle = Text(text, font_size=font_size)
            self.subtitle.move_to(title_location)
            self.play(Write(self.subtitle))
            return

        self.play(Unwrite(self.subtitle))
        new_title = Text(text, font_size=font_size)
        new_title.move_to(title_location)
        self.subtitle = new_title
        self.wait(0.1)
        self.play(Write(self.subtitle))

    def make_embed_input_image_animation(self, input_image_path, output_image_path):
        """Makes embed input image animation"""
        # insert the input image at the begginging
        input_image_layer = ImageLayer.from_path(input_image_path)
        input_image_layer.scale(0.6)
        current_first_layer = self.encoder.all_layers[0]
        # Get connective layer
        connective_layer = get_connective_layer(input_image_layer, current_first_layer)
        # Insert both layers
        self.encoder.all_layers.insert(0, input_image_layer)
        self.encoder.all_layers.insert(1, connective_layer)
        # Move layers to the correct location
        # TODO: Fix this cause its hacky
        input_image_layer.shift(DOWN * 0.4)
        input_image_layer.shift(LEFT * 2.35)
        # Play full forward pass
        forward_pass = self.neural_network.make_forward_pass_animation(
            layer_args={
                self.encoder: {
                    self.embedding_layer: {
                        "dist_args": {
                            "cov": np.array([[1.5, 0], [0, 1.5]]),
                            "mean": np.array([0.5, 0.5]),
                            "dist_theme": "ellipse",
                            "color": ORANGE,
                        }
                    }
                }
            }
        )
        self.play(forward_pass)
        # insert the output image at the end
        output_image_layer = ImageLayer.from_path(output_image_path)
        output_image_layer.scale(0.6)
        self.insert_at_end(output_image_layer)
        # Remove the input and output layers
        self.remove_start_layer()
        self.remove_end_layer()
        # Remove the latent distribution
        self.play(FadeOut(self.embedding_layer.latent_distribution))

    def make_localization_time_step(self, old_posterior):
        """
        Performs one query update for the localization procedure

        Procedure:
        a. Embed query input images
        b. Oracle is asked a query
        c. Query is embedded
        d. Show posterior update
        e. Show current recomendation
        """
        # Helper functions
        def embed_query_to_latent_space(query_locations, query_covariance):
            """Makes animation for a paired query"""
            # Assumes first layer of neural network is a PairedQueryLayer
            # Make the embedding animation
            # Wait
            self.play(Wait(1))
            # Embed the query to latent space
            self.change_subtitle("3. Embed the Query in Latent Space")
            # Make forward pass animation
            self.embedding_layer.paired_query_mode = True
            # Make embedding embed query animation
            embed_query_animation = self.encoder.make_forward_pass_animation(
                run_time=5,
                layer_args={
                    self.embedding_layer: {
                        "positive_dist_args": {
                            "cov": query_covariance[0],
                            "mean": query_locations[0],
                            "dist_theme": "ellipse",
                            "color": BLUE,
                        },
                        "negative_dist_args": {
                            "cov": query_covariance[1],
                            "mean": query_locations[1],
                            "dist_theme": "ellipse",
                            "color": RED,
                        },
                    }
                },
            )
            self.play(embed_query_animation)

        # Access localizer information
        query_paths, query_locations, posterior_distribution, query_covariances = next(
            self.localizer
        )
        positive_path, negative_path = query_paths
        # Make subtitle for present user with query
        self.change_subtitle("2. Present User with Query")
        # Insert the layer into the encoder
        query_layer = PairedQueryLayer.from_paths(
            positive_path, negative_path, grayscale=False
        )
        query_layer.scale(0.5)
        self.insert_at_start(query_layer)
        # Embed query to latent space
        query_to_latent_space_animation = embed_query_to_latent_space(
            query_locations, query_covariances
        )
        # Wait
        self.play(Wait(1))
        # Update the posterior
        self.change_subtitle("4. Update the Posterior")
        # Remove the old posterior
        self.play(ReplacementTransform(old_posterior, posterior_distribution))
        """
        self.play(
            self.embedding_layer.remove_gaussian_distribution(self.localizer.posterior_distribution)
        )
        """
        # self.embedding_layer.add_gaussian_distribution(posterior_distribution)
        # self.localizer.posterior_distribution = posterior_distribution
        # Remove query layer
        self.remove_start_layer()
        # Remove query ellipses

        fade_outs = []
        for dist in self.embedding_layer.gaussian_distributions:
            self.embedding_layer.gaussian_distributions.remove(dist)
            fade_outs.append(FadeOut(dist))

        if not len(fade_outs) == 0:
            fade_outs = AnimationGroup(*fade_outs)
            self.play(fade_outs)

        return posterior_distribution

    def make_generate_estimate_animation(self, estimate_image_path):
        """Makes the generate estimate animation"""
        # Change embedding layer mode
        self.embedding_layer.paired_query_mode = False
        # Sample from posterior distribution
        # self.embedding_layer.latent_distribution = self.localizer.posterior_distribution
        emb_to_ff_ind = self.neural_network.all_layers.index_of(self.encoder)
        embedding_to_ff = self.neural_network.all_layers[emb_to_ff_ind + 1]
        self.play(embedding_to_ff.make_forward_pass_animation())
        # Pass through decoder
        self.play(self.decoder.make_forward_pass_animation(), run_time=1)
        # Create Image layer after the decoder
        output_image_layer = ImageLayer.from_path(estimate_image_path)
        output_image_layer.scale(0.5)
        self.insert_at_end(output_image_layer)
        # Wait
        self.wait(1)
        # Remove the image at the end
        print(self.neural_network)
        self.remove_end_layer()

    def make_triplet_forward_animation(self):
        """Make triplet forward animation"""
        # Make triplet layer
        anchor_path = os.path.join(self.assets_path, "anchor.jpg")
        positive_path = os.path.join(self.assets_path, "positive.jpg")
        negative_path = os.path.join(self.assets_path, "negative.jpg")
        triplet_layer = TripletLayer.from_paths(
            anchor_path,
            positive_path,
            negative_path,
            grayscale=False,
            font_size=100,
            buff=1.05,
        )
        triplet_layer.scale(0.10)
        self.insert_at_start(triplet_layer)
        # Make latent triplet animation
        self.play(
            self.encoder.make_forward_pass_animation(
                layer_args={
                    self.embedding_layer: {
                        "triplet_args": {
                            "anchor_dist": {
                                "cov": np.array([[0.3, 0], [0, 0.3]]),
                                "mean": np.array([0.7, 1.4]),
                                "dist_theme": "ellipse",
                                "color": BLUE,
                            },
                            "positive_dist": {
                                "cov": np.array([[0.2, 0], [0, 0.2]]),
                                "mean": np.array([0.8, -0.4]),
                                "dist_theme": "ellipse",
                                "color": GREEN,
                            },
                            "negative_dist": {
                                "cov": np.array([[0.4, 0], [0, 0.25]]),
                                "mean": np.array([-1, -1.2]),
                                "dist_theme": "ellipse",
                                "color": RED,
                            },
                        }
                    }
                },
                run_time=3,
            )
        )

    def construct(self):
        """
        Makes the whole visualization.

        1. Create the Architecture
            a. Create the traditional VAE architecture with images
        2. The Localization Procedure
        3. The Training Procedure
        """
        # 1. Create the Architecture
        self.neural_network.scale(1.2)
        create_vae = Create(self.neural_network)
        self.play(create_vae, run_time=3)
        # Make changing title
        self.change_title("Oracle Guided Image Synthesis\n      with Relative Queries")
        # 2. The Localization Procedure
        self.change_title("The Localization Procedure")
        # Make algorithm subtitle
        self.change_subtitle("Algorithm Steps")
        # Wait
        self.play(Wait(1))
        # Make prior distribution subtitle
        self.change_subtitle("1. Calculate Prior Distribution")
        # Draw the prior distribution
        self.play(Create(self.localizer.prior_distribution))
        old_posterior = self.localizer.prior_distribution
        # For N queries update the posterior
        for query_index in range(self.localizer.num_queries):
            # Make localization iteration
            old_posterior = self.make_localization_time_step(old_posterior)
            self.play(Wait(1))
            if not query_index == self.localizer.num_queries - 1:
                # Repeat
                self.change_subtitle("5. Repeat")
                # Wait a second
                self.play(Wait(1))
        # Generate final estimate
        self.change_subtitle("5. Generate Estimate Image")
        # Generate an estimate image
        estimate_image_path = os.path.join(self.assets_path, "estimate_image.jpg")
        self.make_generate_estimate_animation(estimate_image_path)
        self.wait(1)
        # Remove old posterior
        self.play(FadeOut(old_posterior))
        # 3. The Training Procedure
        self.change_title("The Training Procedure")
        # Make training animation
        # Do an Image forward pass
        self.change_subtitle("1. Unsupervised Image Reconstruction")
        self.make_embed_input_image_animation(
            self.input_embed_image_path, self.output_embed_image_path
        )
        self.wait(1)
        # Do triplet forward pass
        self.change_subtitle("2. Triplet Loss in Latent Space")
        self.make_triplet_forward_animation()
        self.wait(1)
