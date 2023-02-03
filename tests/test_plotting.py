
from manim import *

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
plt.style.use('dark_background')

from manim_ml.utils.mobjects.plotting import convert_matplotlib_figure_to_image_mobject
from manim_ml.utils.testing.frames_comparison import frames_comparison

__module_test__ = "plotting"

@frames_comparison
def test_matplotlib_to_image_mobject(scene):
    # libraries & dataset
    df = sns.load_dataset('iris')
    # Custom the color, add shade and bandwidth
    matplotlib.use('Agg')
    plt.figure(figsize=(10,10), dpi=100)
    displot = sns.displot(
        x=df.sepal_width, 
        y=df.sepal_length, 
        cmap="Reds", 
        kind="kde",
    )
    plt.axis('off')
    fig = displot.fig
    image_mobject = convert_matplotlib_figure_to_image_mobject(fig)
    # Display the image mobject
    scene.add(image_mobject)

class TestMatplotlibToImageMobject(Scene):

    def construct(self):
        # Make a matplotlib plot
        # libraries & dataset
        df = sns.load_dataset('iris')
        # Custom the color, add shade and bandwidth
        matplotlib.use('Agg')
        plt.figure(figsize=(10,10), dpi=100)
        displot = sns.displot(
            x=df.sepal_width, 
            y=df.sepal_length, 
            cmap="Reds", 
            kind="kde",
        )
        plt.axis('off')
        fig = displot.fig
        image_mobject = convert_matplotlib_figure_to_image_mobject(fig)
        # Display the image mobject
        self.add(image_mobject)


class HexabinScene(Scene):

    def construct(self):
        # Fixing random state for reproducibility
        np.random.seed(19680801)
        n = 100_000
        x = np.random.standard_normal(n)
        y = x + 1.0 * np.random.standard_normal(n)
        xlim = -4, 4
        ylim = -4, 4

        fig, ax0 = plt.subplots(1, figsize=(5, 5))

        hb = ax0.hexbin(x, y, gridsize=50, cmap='inferno')
        ax0.set(xlim=xlim, ylim=ylim)

        self.add(convert_matplotlib_figure_to_image_mobject(fig))
