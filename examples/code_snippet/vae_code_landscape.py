from manim import *
from manim_ml.neural_network.layers import FeedForwardLayer, ImageLayer, EmbeddingLayer
from manim_ml.neural_network.neural_network import NeuralNetwork
from PIL import Image
import numpy as np

config.pixel_height = 720
config.pixel_width = 720
config.frame_height = 6.0
config.frame_width = 6.0


class VAECodeSnippetScene(Scene):
    def make_code_snippet(self):
        code_str = """
            # Make Neural Network
            nn = NeuralNetwork([
                ImageLayer(numpy_image, height=1.2),
                FeedForwardLayer(5),
                FeedForwardLayer(3),
                EmbeddingLayer(),
                FeedForwardLayer(3),
                FeedForwardLayer(5),
                ImageLayer(numpy_image, height=1.2),
            ], layer_spacing=0.1)
            # Play animation
            self.play(nn.make_forward_pass_animation())
        """

        code = Code(
            code=code_str,
            tab_width=4,
            background_stroke_width=1,
            # background_stroke_color=WHITE,
            insert_line_no=False,
            background="window",
            # font="Monospace",
            style="one-dark",
            language="py",
        )

        return code

    def construct(self):
        image = Image.open("../../tests/images/image.jpeg")
        numpy_image = np.asarray(image)
        embedding_layer = EmbeddingLayer(dist_theme="ellipse", point_radius=0.04).scale(
            1.0
        )
        # Make nn
        nn = NeuralNetwork(
            [
                ImageLayer(numpy_image, height=1.2),
                FeedForwardLayer(5),
                FeedForwardLayer(3),
                embedding_layer,
                FeedForwardLayer(3),
                FeedForwardLayer(5),
                ImageLayer(numpy_image, height=1.2),
            ],
            layer_spacing=0.1,
        )

        nn.scale(1.1)
        # Center the nn
        nn.move_to(ORIGIN)
        # nn.rotate(-PI/2)
        # nn.all_layers[0].image_mobject.rotate(PI/2)
        # nn.all_layers[0].image_mobject.shift([0, -0.4, 0])
        # nn.all_layers[-1].image_mobject.rotate(PI/2)
        # nn.all_layers[-1].image_mobject.shift([0, -0.4, 0])
        nn.shift([0, -1.4, 0])
        self.add(nn)
        # Make code snippet
        code_snippet = self.make_code_snippet()
        code_snippet.scale(0.52)
        code_snippet.shift([0, 1.25, 0])
        # code_snippet.shift([-1.25, 0, 0])
        self.add(code_snippet)
        # Play animation
        self.play(nn.make_forward_pass_animation(), run_time=10)


if __name__ == "__main__":
    """Render all scenes"""
    # Neural Network
    nn_scene = VAECodeSnippetScene()
    nn_scene.render()
