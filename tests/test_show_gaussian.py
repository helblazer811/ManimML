from manim import *
from manim_ml.probability import GaussianDistribution

class TestShowGaussian(Scene):

    def construct(self):
        axes = Axes()
        self.add(axes)
        gaussian = GaussianDistribution(
            axes,
            mean=np.array([0.0, 0.0]),
            cov=np.array([[2.0, 0.0], [0.0, 1.0]])
        )
        self.add(gaussian)
