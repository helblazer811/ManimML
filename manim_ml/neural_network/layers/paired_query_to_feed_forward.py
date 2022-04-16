from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.paired_query import PairedQueryLayer
from manim_ml.neural_network.layers.parent_layers import ConnectiveLayer

class PairedQueryToFeedForward(ConnectiveLayer):
    input_class = PairedQueryLayer
    output_class = FeedForwardLayer

    def __init__(self, input_layer, output_layer):
        super().__init__(input_layer, output_layer, input_class=PairedQueryLayer, output_class=FeedForwardLayer)
        pass