import manim
from manim_ml.utils.colorschemes.colorschemes import light_mode, dark_mode, ColorScheme

class ManimMLConfig:

    def __init__(self, default_color_scheme=dark_mode):
        self._color_scheme = default_color_scheme

    @property
    def color_scheme(self):
        return self._color_scheme
    
    @color_scheme.setter
    def color_scheme(self, value):
        if isinstance(value, str):
            if value == "dark_mode":
                self._color_scheme = dark_mode
            elif value == "light_mode":
                self._color_scheme = light_mode
            else:
                raise ValueError(
                    "Color scheme must be either 'dark_mode' or 'light_mode'"
                )
        elif isinstance(value, ColorScheme):
            self._color_scheme = value
            
        manim.config.background_color = self.color_scheme.background_color

# These are accesible from the manim_ml namespace
config = ManimMLConfig()