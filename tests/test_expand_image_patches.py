"""
    Tests animation for converting 
"""

from manim import *

import sys
sys.path.append("..")
from examples.cross_attention_vis.cross_attention_vis import ImagePatches, ExpandPatches

config.background_color = WHITE

class ConvertImageToPatches(Scene):

    def construct(self):
        # Load image mobject
        image = ImageMobject("images/image.jpeg")
        image.move_to(ORIGIN)
        image.scale(10)
        self.add(image)
        self.wait(1)
        patches = ImagePatches(image, grid_width=2)
        self.remove(image)
        self.add(patches)
        self.wait(1)
        # Expand the patches
        self.play(ExpandPatches(patches))


