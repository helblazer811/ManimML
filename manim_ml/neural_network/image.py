
from manim import *
from manim_ml.image import GrayscaleImageMobject
from manim_ml.neural_network.layers import ConnectiveLayer, NeuralNetworkLayer

class ImageLayer(NeuralNetworkLayer):
    """Single Image Layer for Neural Network"""

    def __init__(self, numpy_image, height=1.5):
        super().__init__()
        self.set_z_index(1)
        self.numpy_image = numpy_image
        if len(np.shape(self.numpy_image)) == 2:
            # Assumed Grayscale
            self.image_mobject = GrayscaleImageMobject(self.numpy_image, height=height)
        elif len(np.shape(self.numpy_image)) == 3:
            # Assumed RGB
            self.image_mobject = ImageMobject(self.numpy_image)
        self.add(self.image_mobject)
        """
        # Make an invisible box of the same width as the image object so that
        # methods like get_right() work correctly. 
        self.invisible_rectangle = SurroundingRectangle(self.image_mobject, color=WHITE)
        self.invisible_rectangle.set_fill(WHITE, opacity=0.0)
        # self.invisible_rectangle.set_stroke(WHITE, opacity=0.0)
        self.invisible_rectangle.move_to(self.image_mobject.get_center())
        self.add(self.invisible_rectangle)
        """

    @override_animation(Create)
    def _create_animation(self, **kwargs):
        return FadeIn(self.image_mobject)

    def make_forward_pass_animation(self):
        return Create(self.image_mobject)

    def move_to(self, location):
        """Override of move to"""
        self.image_mobject.move_to(location)

    def get_right(self):
        """Override get right"""
        return self.image_mobject.get_right()

    @property
    def width(self):
        return self.image_mobject.width

class ImageToFeedForward(ConnectiveLayer):
    """Image Layer to FeedForward layer"""

    def __init__(self, input_layer, output_layer, animation_dot_color=RED,
                dot_radius=0.05):
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius

        self.feed_forward_layer = output_layer
        self.image_layer = input_layer
        super().__init__(input_layer, output_layer)

    def make_forward_pass_animation(self):
        """Makes dots diverge from the given location and move to the feed forward nodes decoder"""
        animations = []
        dots = []
        image_mobject = self.image_layer.image_mobject
        # Move the dots to the centers of each of the nodes in the FeedForwardLayer
        image_location  = image_mobject.get_center()
        for node in self.feed_forward_layer.node_group:
            new_dot = Dot(image_location, radius=self.dot_radius, color=self.animation_dot_color)
            per_node_succession = Succession(
                Create(new_dot),
                new_dot.animate.move_to(node.get_center()),
            )
            animations.append(per_node_succession)
            dots.append(new_dot)
        self.add(VGroup(*dots))
        animation_group = AnimationGroup(*animations)
        return animation_group

    @override_animation(Create)
    def _create_override(self):
        return AnimationGroup()

class FeedForwardToImage(ConnectiveLayer):
    """Image Layer to FeedForward layer"""

    def __init__(self, input_layer, output_layer, animation_dot_color=RED,
                dot_radius=0.05):
        self.animation_dot_color = animation_dot_color
        self.dot_radius = dot_radius
        
        self.feed_forward_layer = input_layer
        self.image_layer = output_layer
        super().__init__(input_layer, output_layer)

    def make_forward_pass_animation(self):
        """Makes dots diverge from the given location and move to the feed forward nodes decoder"""
        animations = []
        image_mobject = self.image_layer.image_mobject
        # Move the dots to the centers of each of the nodes in the FeedForwardLayer
        image_location  = image_mobject.get_center()
        for node in self.feed_forward_layer.node_group:
            new_dot = Dot(node.get_center(), radius=self.dot_radius, color=self.animation_dot_color)
            per_node_succession = Succession(
                Create(new_dot),
                new_dot.animate.move_to(image_location),
            )
            animations.append(per_node_succession)

        animation_group = AnimationGroup(*animations)
        return animation_group

    @override_animation(Create)
    def _create_override(self):
        return AnimationGroup()