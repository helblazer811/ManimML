from manim import *
import random

from manim_ml.neural_network.layers.parent_layers import VGroupNeuralNetworkLayer

class VectorLayer(VGroupNeuralNetworkLayer):
    """Shows a vector"""

    def __init__(self, num_values, value_func=lambda: random.uniform(0, 1),
                **kwargs):
        print("vector layer")
        super().__init__(**kwargs)
        print("after init")
        self.num_values = num_values
        self.value_func = value_func
        # Make the vector
        self.vector_label = self.make_vector()

    def make_vector(self):
        """Makes the vector"""
        if False:
            # TODO install Latex
            values = np.array([self.value_func() for i in range(self.num_values)])
            values = values[None, :].T
            vector = Matrix(values)

        vector_label = Text(f"[{self.value_func()}]")

        return vector_label

    def make_forward_pass_animation(self):
        return AnimationGroup()
        
    @override_animation(Create)
    def _create_override(self):
        """Create animation"""
        return Create(self.vector_label)