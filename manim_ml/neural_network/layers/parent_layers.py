from manim import *
from abc import ABC, abstractmethod

class NeuralNetworkLayer(ABC, Group):
    """Abstract Neural Network Layer class"""

    def __init__(self, text=None, *args, **kwargs):
        super(Group, self).__init__()
        self.title_text = kwargs["title"] if "title" in kwargs else " "
        self.title = Text(
            self.title_text, 
            font_size=DEFAULT_FONT_SIZE/3
        ).scale(0.6)
        self.title.next_to(self, UP, 1.2)
        # self.add(self.title)

    @abstractmethod
    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        pass

    @override_animation(Create)
    def _create_override(self):
        return AnimationGroup()

    def __repr__(self):
        return f"{type(self).__name__}"

class VGroupNeuralNetworkLayer(NeuralNetworkLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.camera = camera

    @abstractmethod
    def make_forward_pass_animation(self, **kwargs):
        pass 

    @override_animation(Create)
    def _create_override(self):
        return super()._create_override()

class ThreeDLayer(ABC):
    """Abstract class for 3D layers"""
    # Angle of ThreeD layers is static context
    three_d_x_rotation = 0 * DEGREES #-90 * DEGREES
    three_d_y_rotation = 75 * DEGREES # -10 * DEGREES

class ConnectiveLayer(VGroupNeuralNetworkLayer):
    """Forward pass animation for a given pair of layers"""

    @abstractmethod
    def __init__(self, input_layer, output_layer, **kwargs):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        self.input_layer = input_layer
        self.output_layer = output_layer
        # Handle input and output class
        # assert isinstance(input_layer, self.input_class), f"{input_layer}, {self.input_class}"
        # assert isinstance(output_layer, self.output_class), f"{output_layer}, {self.output_class}"

    @abstractmethod
    def make_forward_pass_animation(self, run_time=2.0, layer_args={}, **kwargs):
        pass

    @override_animation(Create)
    def _create_override(self):
        return super()._create_override()

class BlankConnective(ConnectiveLayer):
    """Connective layer to be used when the given pair of layers is undefined"""

    def __init__(self, input_layer, output_layer, **kwargs):
        super().__init__(input_layer, output_layer, **kwargs)

    def make_forward_pass_animation(self, run_time=1.5, layer_args={}, **kwargs):
        return AnimationGroup(run_time=run_time)

    @override_animation(Create)
    def _create_override(self):
        return super()._create_override()