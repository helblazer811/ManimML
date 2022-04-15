from manim import *
from manim_ml.neural_network.layers import NeuralNetworkLayer
from manim_ml.image import GrayscaleImageMobject
import numpy as np

class TripletLayer(NeuralNetworkLayer):
    """Shows triplet images"""

    def __init__(self, anchor, positive, negative, stroke_width=5):
        super().__init__()
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

        self.stroke_width = stroke_width
        # Make the assets
        self.assets = self.make_assets()
        self.add(self.assets)
    
    @classmethod
    def from_paths(cls, anchor_path, positive_path, negative_path, grayscale=True):
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
        triplet_layer = cls(anchor, positive, negative)

        return triplet_layer

    def make_assets(self):
        """
            Constructs the assets needed for a triplet layer
        """
        # Handle anchor
        anchor_text = Text("Anchor").scale(2)
        anchor_text.next_to(self.anchor, UP, buff=1.0)
        anchor_rectangle = SurroundingRectangle(
            self.anchor, 
            color=WHITE,
            buff=0.0,
            stroke_width=self.stroke_width
        )
        anchor_group = Group(
            anchor_text,
            anchor_rectangle,
            self.anchor,
        )
        # Handle positive
        positive_text = Text("Positive").scale(2)
        positive_text.next_to(self.positive, UP, buff=1.0)
        positive_rectangle = SurroundingRectangle(
            self.positive, 
            color=GREEN,
            buff=0.0,
            stroke_width=self.stroke_width
        )
        positive_group = Group(
            positive_text,
            positive_rectangle,
            self.positive
        )
        # Handle negative
        negative_text = Text("Negative").scale(2)
        negative_text.next_to(self.negative, UP, buff=1.0)
        negative_rectangle = SurroundingRectangle(
            self.negative, 
            color=RED,
            buff=0.0,
            stroke_width=self.stroke_width
        )
        negative_group = Group(
            negative_text,
            negative_rectangle,
            self.negative
        )
        # Distribute the groups uniformly vertically
        assets = Group(anchor_group, positive_group, negative_group)
        assets.arrange(DOWN, buff=1.5)

        return assets

    @override_animation(Create)
    def _create_layer(self):
        # TODO make Create animation that is custom
        return FadeIn(self.assets)

    def make_forward_pass_animation(self):
        """Forward pass for triplet"""
        return AnimationGroup()
