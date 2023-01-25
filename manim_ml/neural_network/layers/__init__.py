from manim_ml.neural_network.layers.convolutional_2d_to_feed_forward import (
    Convolutional2DToFeedForward,
)
from manim_ml.neural_network.layers.convolutional_2d_to_max_pooling_2d import (
    Convolutional2DToMaxPooling2D,
)
from manim_ml.neural_network.layers.image_to_convolutional_2d import (
    ImageToConvolutional2DLayer,
)
from manim_ml.neural_network.layers.max_pooling_2d_to_convolutional_2d import (
    MaxPooling2DToConvolutional2D,
)
from .convolutional_2d_to_convolutional_2d import Convolutional2DToConvolutional2D
from .convolutional_2d import Convolutional2DLayer
from .feed_forward_to_vector import FeedForwardToVector
from .paired_query_to_feed_forward import PairedQueryToFeedForward
from .embedding_to_feed_forward import EmbeddingToFeedForward
from .embedding import EmbeddingLayer
from .feed_forward_to_embedding import FeedForwardToEmbedding
from .feed_forward_to_feed_forward import FeedForwardToFeedForward
from .feed_forward_to_image import FeedForwardToImage
from .feed_forward import FeedForwardLayer
from .image_to_feed_forward import ImageToFeedForward
from .image import ImageLayer
from .parent_layers import ConnectiveLayer, NeuralNetworkLayer
from .triplet import TripletLayer
from .triplet_to_feed_forward import TripletToFeedForward
from .paired_query import PairedQueryLayer
from .paired_query_to_feed_forward import PairedQueryToFeedForward
from .max_pooling_2d import MaxPooling2DLayer

connective_layers_list = (
    EmbeddingToFeedForward,
    FeedForwardToEmbedding,
    FeedForwardToFeedForward,
    FeedForwardToImage,
    ImageToFeedForward,
    PairedQueryToFeedForward,
    TripletToFeedForward,
    PairedQueryToFeedForward,
    FeedForwardToVector,
    Convolutional2DToConvolutional2D,
    Convolutional2DToConvolutional2D,
    ImageToConvolutional2DLayer,
    Convolutional2DToFeedForward,
    Convolutional2DToMaxPooling2D,
    MaxPooling2DToConvolutional2D,
)
