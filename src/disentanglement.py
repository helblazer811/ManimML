"""This module is dedicated to visualizing VAE disentanglement"""
from manim import *
from neural_network import NeuralNetwork
import util
import pickle

class VAEDecoder(VGroup):
    """Just shows the VAE encoder"""

    def __init__(self):
        super(VGroup, self).__init__()
        # Setup the Neural Network
        node_counts = [3, 5]
        self.neural_network = NeuralNetwork(node_counts, layer_spacing=0.55)
        self.add(self.neural_network)

    def make_encoding_animation(self):
        pass     

class DisentanglementVisualization(VGroup):

    def __init__(self, model_path="autoencoder_models/saved_models/model_dim2.pth", image_height=0.35):
        self.model_path = model_path
        self.image_height = image_height
        # Load disentanglement image objects
        with open("autoencoder_models/disentanglement.pkl", "rb") as f:
            self.image_handler = pickle.load(f)

    def make_disentanglement_generation_animation(self):
        animation_list = []
        for image_index, image in enumerate(self.image_handler["images"]):
            image_mobject = util.construct_image_mobject(image, height=self.image_height)
            r, c = self.image_handler["bin_indices"][image_index]
            # Move the image to the correct location
            r_offset = -1.2
            c_offset = 0.25
            image_location = [c_offset + c*self.image_height, r_offset + r*self.image_height, 0]
            image_mobject.move_to(image_location)
            animation_list.append(FadeIn(image_mobject))

        generation_animation = AnimationGroup(*animation_list[::-1], lag_ratio=1.0)
        return generation_animation

config.pixel_height = 720
config.pixel_width = 1280
config.frame_height = 5.0
config.frame_width = 5.0

class DisentanglementScene(Scene):
    """Disentanglement Scene Object"""

    def _construct_embedding(self, point_color=BLUE, dot_radius=0.05):
        """Makes a Gaussian-like embedding"""
        embedding = VGroup()
        # Sample points from a Gaussian
        num_points = 200
        standard_deviation = [0.6, 0.8]
        mean = [0, 0]
        points = np.random.normal(mean, standard_deviation, size=(num_points, 2))
        # Make an axes
        embedding.axes = Axes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            x_length=2.2,
            y_length=2.2,
            tips=False,
        )
        # Add each point to the axes
        self.point_dots = VGroup()
        for point in points:
            point_location = embedding.axes.coords_to_point(*point)
            dot = Dot(point_location, color=point_color, radius=dot_radius/2) 
            self.point_dots.add(dot)

        embedding.add(self.point_dots)
        return embedding

    def construct(self):
        # Make the VAE decoder
        vae_decoder = VAEDecoder()
        vae_decoder.shift([-0.55, 0, 0])
        self.play(Create(vae_decoder), run_time=1)
        # Make the embedding
        embedding = self._construct_embedding()
        embedding.scale(0.9)
        embedding.move_to(vae_decoder.get_left())
        embedding.shift([-0.85, 0, 0])
        self.play(Create(embedding))
        # Make disentanglment visualization
        disentanglement = DisentanglementVisualization()
        disentanglement_animation = disentanglement.make_disentanglement_generation_animation()
        self.play(disentanglement_animation, run_time=3)
        self.play(Wait(2))