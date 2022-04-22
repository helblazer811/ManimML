from manim import *

from manim_ml.neural_network.layers.convolutional import ConvolutionalLayer
from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

class SingleConvolutionalLayerScence(Scene):

    def construct(self):
        
        # Make nn
        layers = [
            ConvolutionalLayer()
        ]
        nn = NeuralNetwork(layers)
        nn.scale(1.3)
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        self.play(nn.make_forward_pass_animation(run_time=5))
        self.play(nn.make_forward_pass_animation(run_time=5))

class ThreeDLightSourcePosition(ThreeDScene, Scene):
    def construct(self):
        axes = ThreeDAxes()
        sphere = Surface(
            lambda u, v: np.array([
                u,
                v,
                0
            ]), v_range=[0, TAU], u_range=[-PI / 2, PI / 2],
            checkerboard_colors=[RED_D, RED_E], resolution=(15, 32)
        )
        self.renderer.camera.light_source.move_to(3*IN) # changes the source of the light
        self.set_camera_orientation(phi=90 * DEGREES, theta=0 * DEGREES)
        self.add(axes, sphere)

class CombinedScene(Scene):
    
    def constuct(self):
        pass