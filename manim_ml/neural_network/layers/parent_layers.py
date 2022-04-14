from manim import *
from abc import ABC, abstractmethod

class NeuralNetworkLayer(ABC, Group):
    """Abstract Neural Network Layer class"""

    def __init__(self, **kwargs):
        super(Group, self).__init__()
        self.set_z_index(1)

    @abstractmethod
    def make_forward_pass_animation(self):
        pass

    def __repr__(self):
        return f"{type(self).__name__}"

class VGroupNeuralNetworkLayer(NeuralNetworkLayer):

    def __init__(self, **kwargs):
        super(NeuralNetworkLayer, self).__init__()

    @abstractmethod
    def make_forward_pass_animation(self):
        pass 

class ConnectiveLayer(VGroupNeuralNetworkLayer):
    """Forward pass animation for a given pair of layers"""

    @abstractmethod
    def __init__(self, input_layer, output_layer):
        super(VGroupNeuralNetworkLayer, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer

        self.set_z_index(-1)

    @abstractmethod
    def make_forward_pass_animation(self):
        pass