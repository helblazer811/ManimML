from manim import *


class LazyAnimation(Animation):
    def __init__(self, animation_function):
        self.animation_function = animation_function
        super.__init__()

    def begin(self):
        update_func_anim = UpdateFromFunc(self.neural_network, create_new_connective)
        self.add

        super.begin()
