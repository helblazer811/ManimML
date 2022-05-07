from .convolutional3d_to_convolutional3d import Convolutional3DToConvolutional3D
from .convolutional2d_to_convolutional2d import Convolutional2DToConvolutional2D
from .convolutional_3d import Convolutional3DLayer
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
    Convolutional3DToConvolutional3D,
)
