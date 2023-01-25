from manim import *
from manim_ml.neural_network.layers import NeuralNetworkLayer
from manim_ml.image import GrayscaleImageMobject, LabeledColorImage
import numpy as np


class TripletLayer(NeuralNetworkLayer):
    """Shows triplet images"""

    def __init__(
        self,
        anchor,
        positive,
        negative,
        stroke_width=5,
        font_size=22,
        buff=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
        self.buff = buff

        self.stroke_width = stroke_width
        self.font_size = font_size

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        # Make the assets
        self.assets = self.make_assets()
        self.add(self.assets)

    @classmethod
    def from_paths(
        cls,
        anchor_path,
        positive_path,
        negative_path,
        grayscale=True,
        font_size=22,
        buff=0.2,
    ):
        """Creates a triplet using the anchor paths"""
        # Load images from path
        if grayscale:
            anchor = GrayscaleImageMobject.from_path(anchor_path)
            positive = GrayscaleImageMobject.from_path(positive_path)
            negative = GrayscaleImageMobject.from_path(negative_path)
        else:
            anchor = ImageMobject(anchor_path)
            positive = ImageMobject(positive_path)
            negative = ImageMobject(negative_path)
        # Make the layer
        triplet_layer = cls(anchor, positive, negative, font_size=font_size, buff=buff)

        return triplet_layer

    def make_assets(self):
        """
        Constructs the assets needed for a triplet layer
        """
        # Handle anchor
        anchor_group = LabeledColorImage(
            self.anchor,
            color=WHITE,
            label="Anchor",
            stroke_width=self.stroke_width,
            font_size=self.font_size,
            buff=self.buff,
        )
        # Handle positive
        positive_group = LabeledColorImage(
            self.positive,
            color=GREEN,
            label="Positive",
            stroke_width=self.stroke_width,
            font_size=self.font_size,
            buff=self.buff,
        )
        # Handle negative
        negative_group = LabeledColorImage(
            self.negative,
            color=RED,
            label="Negative",
            stroke_width=self.stroke_width,
            font_size=self.font_size,
            buff=self.buff,
        )
        # Distribute the groups uniformly vertically
        assets = Group(anchor_group, positive_group, negative_group)
        assets.arrange(DOWN, buff=1.5)

        return assets

    @override_animation(Create)
    def _create_override(self):
        # TODO make Create animation that is custom
        return FadeIn(self.assets)

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        """Forward pass for triplet"""
        return AnimationGroup()
