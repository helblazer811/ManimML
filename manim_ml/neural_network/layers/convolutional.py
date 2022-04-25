
from manim import *
from manim_ml.neural_network.layers.parent_layers import VGroupNeuralNetworkLayer

class ConvolutionalLayer(VGroupNeuralNetworkLayer):
    """Handles rendering a convolutional layer for a nn"""

    def __init__(self, num_filters, filter_width, **kwargs):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_width = filter_width

        self._construct_neural_network_layer()

    def _construct_neural_network_layer(self):
        """Creates the neural network layer"""
        pass

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        # make highlight animation
        return None

    @override_animation(Create)
    def _create_override(self, **kwargs):
        pass
