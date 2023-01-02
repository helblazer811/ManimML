from cv2 import exp
from manim import *
from manim_ml.neural_network.layers.embedding import EmbeddingLayer
from manim_ml.neural_network.layers.embedding_to_feed_forward import (
    EmbeddingToFeedForward,
)
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.feed_forward_to_embedding import (
    FeedForwardToEmbedding,
)
from manim_ml.neural_network.layers.feed_forward_to_feed_forward import (
    FeedForwardToFeedForward,
)
from manim_ml.neural_network.layers.image import ImageLayer
from manim_ml.neural_network.neural_network import (
    NeuralNetwork,
    FeedForwardNeuralNetwork,
)
from PIL import Image
import numpy as np

config.pixel_height = 720
config.pixel_width = 1280
config.frame_height = 6.0
config.frame_width = 6.0

"""
    Unit Tests
"""


def assert_classes_match(all_layers, expected_classes):
    assert len(list(all_layers)) == 5

    for index, layer in enumerate(all_layers):
        expected_class = expected_classes[index]
        assert isinstance(
            layer, expected_class
        ), f"Wrong layer class {layer.__class__} expected {expected_class}"


def test_embedding_layer():
    embedding_layer = EmbeddingLayer()

    neural_network = NeuralNetwork(
        [FeedForwardLayer(5), FeedForwardLayer(3), embedding_layer]
    )

    expected_classes = [
        FeedForwardLayer,
        FeedForwardToFeedForward,
        FeedForwardLayer,
        FeedForwardToEmbedding,
        EmbeddingLayer,
    ]

    assert_classes_match(neural_network.all_layers, expected_classes)


def test_remove_layer():
    embedding_layer = EmbeddingLayer()

    neural_network = NeuralNetwork(
        [FeedForwardLayer(5), FeedForwardLayer(3), embedding_layer]
    )

    expected_classes = [
        FeedForwardLayer,
        FeedForwardToFeedForward,
        FeedForwardLayer,
        FeedForwardToEmbedding,
        EmbeddingLayer,
    ]

    assert_classes_match(neural_network.all_layers, expected_classes)

    print("before removal")
    print(list(neural_network.all_layers))
    neural_network.remove_layer(embedding_layer)
    print("after removal")
    print(list(neural_network.all_layers))

    expected_classes = [
        FeedForwardLayer,
        FeedForwardToFeedForward,
        FeedForwardLayer,
    ]

    print(list(neural_network.all_layers))

    assert_classes_match(neural_network.all_layers, expected_classes)


class FeedForwardNeuralNetworkScene(Scene):
    def construct(self):
        nn = FeedForwardNeuralNetwork([3, 5, 3])
        self.play(Create(nn))
        self.play(Wait(3))


class NeuralNetworkScene(Scene):
    """Test Scene for the Neural Network"""

    def construct(self):
        # Make the Layer object
        layers = [FeedForwardLayer(3), FeedForwardLayer(5), FeedForwardLayer(3)]
        nn = NeuralNetwork(layers)
        nn.move_to(ORIGIN)
        # Make Animation
        self.add(nn)
        # self.play(Create(nn))
        forward_propagation_animation = nn.make_forward_pass_animation(
            run_time=5, passing_flash=True
        )

        self.play(forward_propagation_animation)


class GrayscaleImageNeuralNetworkScene(Scene):
    def construct(self):
        image = Image.open("images/image.jpeg")
        numpy_image = np.asarray(image)
        # Make nn
        layers = [
            FeedForwardLayer(3),
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            FeedForwardLayer(6),
            ImageLayer(numpy_image, height=1.4),
        ]
        nn = NeuralNetwork(layers)
        nn.scale(1.3)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        self.play(nn.make_forward_pass_animation(run_time=5))
        self.play(nn.make_forward_pass_animation(run_time=5))


class ImageNeuralNetworkScene(Scene):
    def construct(self):
        image = Image.open("../assets/gan/real_image.jpg")
        numpy_image = np.asarray(image)
        # Make nn
        layers = [
            FeedForwardLayer(3),
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            FeedForwardLayer(6),
            ImageLayer(numpy_image, height=1.4),
        ]
        nn = NeuralNetwork(layers)
        nn.scale(1.3)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        self.play(nn.make_forward_pass_animation(run_time=5))
        self.play(nn.make_forward_pass_animation(run_time=5))


class RecursiveNNScene(Scene):
    def construct(self):
        nn = NeuralNetwork(
            [
                NeuralNetwork([FeedForwardLayer(3), FeedForwardLayer(2)]),
                NeuralNetwork([FeedForwardLayer(2), FeedForwardLayer(3)]),
            ]
        )

        self.play(Create(nn))


class LayerInsertionScene(Scene):
    def construct(self):
        pass


class LayerRemovalScene(Scene):
    def construct(self):
        image = Image.open("images/image.jpeg")
        numpy_image = np.asarray(image)

        layer = FeedForwardLayer(5)
        layers = [
            ImageLayer(numpy_image, height=1.4),
            FeedForwardLayer(3),
            layer,
            FeedForwardLayer(3),
            FeedForwardLayer(6),
        ]

        nn = NeuralNetwork(layers)

        self.play(Create(nn))
        remove_animation = nn.remove_layer(layer)
        print("before remove")
        self.play(remove_animation)
        print(nn)
        print("after remove")


class LayerInsertionScene(Scene):
    def construct(self):
        image = Image.open("images/image.jpeg")
        numpy_image = np.asarray(image)

        layers = [
            ImageLayer(numpy_image, height=1.4),
            FeedForwardLayer(3),
            FeedForwardLayer(3),
            FeedForwardLayer(6),
        ]

        nn = NeuralNetwork(layers)

        self.play(Create(nn))

        layer = FeedForwardLayer(5)
        insert_animation = nn.insert_layer(layer, 4)
        self.play(insert_animation)
        print(nn)
        print("after remove")


if __name__ == "__main__":
    """Render all scenes"""
    # Feed Forward Neural Network
    ffnn_scene = FeedForwardNeuralNetworkScene()
    ffnn_scene.render()
    # Neural Network
    nn_scene = NeuralNetworkScene()
    nn_scene.render()
