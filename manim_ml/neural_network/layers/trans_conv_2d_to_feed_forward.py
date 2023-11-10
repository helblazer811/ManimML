from manim import *
from manim_ml.neural_network.layers.convolutional_2d_to_feed_forward import Convolutional2DToFeedForward
from manim_ml.neural_network.layers.trans_conv_2d import TransposeConvolution2DLayer
# from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer, ThreeDLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
# from manim_ml.neural_network.layers.convolutional_2d import Convolutional2DLayer

class TransConv2DToFeedForward(Convolutional2DToFeedForward):
    """Effectively the same as Convolutional2DToFeedForward, but with a different name
    in order to keep with the connection layer naming convention."""
    input_class = TransposeConvolution2DLayer
    output_class = FeedForwardLayer

