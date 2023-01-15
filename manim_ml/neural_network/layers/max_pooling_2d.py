from manim import *

from manim_ml.neural_network.layers.parent_layers import ThreeDLayer, VGroupNeuralNetworkLayer

class MaxPooling2DLayer(VGroupNeuralNetworkLayer, ThreeDLayer):
    """Max pooling layer for Convolutional2DLayer

    Note: This is for a Convolutional2DLayer even though
    it is called MaxPooling2DLayer because the 2D corresponds
    to the 2 spatial dimensions of the convolution. 
    """

    def __init__(self, output_feature_map_size=(4, 4), kernel_size=2, stride=1, 
        cell_highlight_color=ORANGE, **kwargs):
        """Layer object for animating 2D Convolution Max Pooling

        Parameters
        ----------
        kernel_size : int or tuple, optional
            Width/Height of max pooling kernel, by default 2
        stride : int, optional
            Stride of the max pooling operation, by default 1
        """
        super().__init__(**kwargs)
        self.output_feature_map_size = output_feature_map_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.cell_highlight_color = cell_highlight_color

    def construct_layer(self, input_layer: 'NeuralNetworkLayer', output_layer: 'NeuralNetworkLayer', **kwargs):
        # Make the output feature maps
        feature_maps = self._make_output_feature_maps()
        self.add(feature_maps)
    
    def _make_output_feature_maps(self):
        """Makes a set of output feature maps"""
        # Compute the size of the feature maps 
        pass

    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        """Makes forward pass of Max Pooling Layer. 

        Parameters
        ----------
        layer_args : dict, optional
            _description_, by default {}
        """
        # 1. Draw gridded rectangle with kernel_size x kernel_size 
        #   box regions over the input feature map.
        # 2. Randomly highlight one of the cells in the kernel.
        # 3. Move and resize the gridded rectangle to the output 
        #   feature maps. 
        # 4. Make the gridded feature map(s) disappear. 
        pass

    @override_animation(Create)
    def _create_override(self, **kwargs):
        """Create animation for the MaxPooling operation"""
        pass
