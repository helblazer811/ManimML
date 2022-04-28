from manim import *
from abc import ABC, abstractmethod

class NeuralNetworkLayer(ABC, Group):
    """Abstract Neural Network Layer class"""

    def __init__(self, text=None, **kwargs):
        super(Group, self).__init__()
        self.title_text = kwargs["title"] if "title" in kwargs else " "
        self.title = Text(self.title_text, font_size=DEFAULT_FONT_SIZE/3).scale(0.6)
        self.title.next_to(self, UP, 1.2)
        # self.add(self.title)

    @abstractmethod
    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        pass

    @override_animation(Create)
    def _create_override(self):
        pass

    def __repr__(self):
        return f"{type(self).__name__}"

class VGroupNeuralNetworkLayer(NeuralNetworkLayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def make_forward_pass_animation(self, **kwargs):
        pass 

    @override_animation(Create)
    def _create_override(self):
        return super()._create_override()

class ConnectiveLayer(VGroupNeuralNetworkLayer):
    """Forward pass animation for a given pair of layers"""

    @abstractmethod
    def __init__(self, input_layer, output_layer, input_class=None, output_class=None,
                **kwargs):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.input_class = input_class
        self.output_class = output_class
        # Handle input and output class
        assert isinstance(input_layer, self.input_class)
        assert isinstance(output_layer, self.output_class)

    @abstractmethod
    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        pass

    @override_animation(Create)
    def _create_override(self):
        return super()._create_override()