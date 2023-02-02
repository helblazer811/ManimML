from manim import *

# Import modules here


class BasicScene(ThreeDScene):
    def construct(self):
        # Your code goes here
        text = Text("Your first scene!")
        self.add(text)
