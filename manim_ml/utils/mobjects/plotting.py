from manim import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import io

def convert_matplotlib_figure_to_image_mobject(fig, dpi=200):
    """Takes a matplotlib figure and makes an image mobject from it

    Parameters
    ----------
    fig : matplotlib figure
        matplotlib figure
    """
    fig.tight_layout(pad=0)
    # plt.axis('off')
    fig.canvas.draw()
    # Save data into a buffer
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png', dpi=dpi)
    # Reopen in PIL and convert to numpy
    image = Image.open(image_buffer)
    image = np.array(image)
    # Convert it to an image mobject
    image_mobject = ImageMobject(image, image_mode="RGB")

    return image_mobject