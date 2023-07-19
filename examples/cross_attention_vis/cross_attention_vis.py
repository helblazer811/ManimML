"""
    Here I thought it would be interesting to visualize the cross attention
    maps produced by the UNet of a text-to-image diffusion model. 

    The key thing I want to show is how images and text are broken up into tokens (patches for images),
    and those tokens are used to compute a cross attention score, which can be interpreted
    as a 2D heatmap over the image patches for each text token. 
    

    Necessary operations:
        1. [X] Split an image into patches and "expand" the patches outward to highlight that they are 
            separate. 
        2. Split text into tokens using the tokenizer and display them. 
        3. Compute the cross attention scores (using DAAM) for each word/token and display them as a heatmap.
        4. Overlay the heatmap over the image patches. Show the overlaying as a transition animation. 
"""
import torch
import cv2
from manim import *
import numpy as np
from typing import List
from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
import torchvision
import matplotlib.pyplot as plt

class ImagePatches(Group):
    """Container object for a grid of ImageMobjects."""

    def __init__(self, image: ImageMobject, grid_width=4):
        """
        Parameters
        ----------
        image : ImageMobject
            image to split into patches
        grid_width : int, optional
            number of patches per row, by default 4
        """
        self.image = image
        self.grid_width = grid_width
        super(Group, self).__init__()
        self.patch_dict = self._split_image_into_patches(image, grid_width)

    def _split_image_into_patches(self, image, grid_width):
        """Splits the images into a set of patches

        Parameters
        ----------
        image : ImageMobject
            image to split into patches
        grid_width : int
            number of patches per row
        """
        patch_dict = {}
        # Get a pixel array of the image mobject
        pixel_array = image.pixel_array
        # Get the height and width of the image
        height, width = pixel_array.shape[:2]
        # Split the image into an equal width grid of patches
        assert height == width, "Image must be square"
        assert height % grid_width == 0, "Image width must be divisible by grid_width"
        patch_width, patch_height = width // grid_width, height // grid_width

        for patch_i in range(self.grid_width):
            for patch_j in range(self.grid_width):
                # Get patch pixels
                i_start, i_end = patch_i * patch_width, (patch_i + 1) * patch_width
                j_start, j_end = patch_j * patch_height, (patch_j + 1) * patch_height
                patch_pixels = pixel_array[i_start:i_end, j_start:j_end, :]
                # Make the patch
                image_patch = ImageMobject(patch_pixels)
                # Move the patch to the correct location on the old image
                image_width = image.get_width()
                image_center = image.get_center()
                image_patch.scale(image_width / grid_width / image_patch.get_width())
                patch_manim_space_width = image_patch.get_width()

                patch_center = image_center
                patch_center += (patch_j - self.grid_width / 2 + 0.5) * patch_manim_space_width * RIGHT
                # patch_center = image_center - (patch_i - self.grid_width / 2 + 0.5) * patch_manim_space_width * RIGHT
                patch_center -= (patch_i - self.grid_width / 2 + 0.5) * patch_manim_space_width * UP
                # + (patch_j - self.grid_width / 2) * patch_height / 2 * UP

                image_patch.move_to(patch_center)

                self.add(image_patch)
                patch_dict[(patch_i, patch_j)] = image_patch

        return patch_dict

class ExpandPatches(Animation):

    def __init__(self, image_patches: ImagePatches, expansion_scale=2.0):
        """
        Parameters
        ----------
        image_patches : ImagePatches
            set of image patches
        expansion_scale : float, optional
            scale factor for expansion, by default 2.0
        """
        self.image_patches = image_patches
        self.expansion_scale = expansion_scale
        super().__init__(image_patches)

    def interpolate_submobject(self, submobject, starting_submobject, alpha):
        """
        Parameters
        ----------
        submobject : ImageMobject
            current patch
        starting_submobject : ImageMobject
            starting patch
        alpha : float
            interpolation alpha
        """
        patch = submobject
        starting_patch_center = starting_submobject.get_center()
        image_center = self.image_patches.image.get_center()
        # Start image vector
        starting_patch_vector = starting_patch_center - image_center
        final_image_vector = image_center + starting_patch_vector * self.expansion_scale
        # Interpolate vectors
        current_location = alpha * final_image_vector + (1 - alpha) * starting_patch_center
        # # Need to compute the direction of expansion as the unit vector from the original image center
        # # and patch center. 
        patch.move_to(current_location)

class TokenizedText(Group):
    """Tokenizes the given text and displays the tokens as a list of Text Mobjects."""

    def __init__(self, text, tokenizer=None):
        self.text = text
        if not tokenizer is None:
            self.tokenizer = tokenizer
        else:
            # TODO load default stable diffusion tokenizer here
            raise NotImplementedError("Tokenizer must be provided")

        self.token_strings = self._tokenize_text(text)
        self.token_mobjects = self.make_text_mobjects(self.token_strings)

        super(Group, self).__init__()
        self.add(*self.token_mobjects)
        self.arrange(RIGHT, buff=-0.05, aligned_edge=UP)

        token_dict = {} 
        for token_index, token_string in enumerate(self.token_strings):
            token_dict[token_string] = self.token_mobjects[token_index]

        self.token_dict = token_dict

    def _tokenize_text(self, text):
        """Tokenize the text using the tokenizer.

        Parameters
        ----------
        text : str
            text to tokenize
        """
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.replace("</w>", "") for token in tokens]
        return tokens

    def make_text_mobjects(self, tokens_list: List[str]):
        """Converts the tokens into a list of TextMobjects."""
        # Make the tokens
        # Insert phantom
        tokens_list = ["l" + token + "g" for token in tokens_list]
        token_objects = [
            Text(token, t2c={'[-1:]': '#000000', '[0:1]': "#000000"}, font_size=64) 
            for token in tokens_list
        ]

        return token_objects

def compute_stable_diffusion_cross_attention_heatmaps(
    pipe,
    prompt: str,
    seed: int = 2,
    map_resolution=(32, 32)
):
    """Computes the cross attention heatmaps for the given prompt.

    Parameters
    ----------
    prompt : str
        the prompt
    seed : int, optional
        random seed, by default 0
    map_resolution : tuple, optional
        resolution to downscale maps to, by default (16, 16)

    Returns
    -------
    _type_
        _description_
    """
    # Get tokens
    tokens = pipe.tokenizer.tokenize(prompt)
    tokens = [token.replace("</w>", "") for token in tokens]
    # Set torch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    gen = set_seed(seed)  # for reproducibility

    heatmap_dict = {}
    image = None

    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        with trace(pipe) as tc:
            out = pipe(prompt, num_inference_steps=30, generator=gen)
            image = out[0][0]
            global_heat_map = tc.compute_global_heat_map()
            for token in tokens: 
                word_heat_map = global_heat_map.compute_word_heat_map(token)
                heatmap = word_heat_map.heatmap

                # Downscale the heatmap
                heatmap = heatmap.unsqueeze(0).unsqueeze(0)
                # Save the heatmap
                heatmap = torchvision.transforms.Resize(
                    map_resolution, 
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST
                )(heatmap)
                heatmap = heatmap.squeeze(0).squeeze(0)
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # plt.imshow(heatmap)
                # plt.savefig(f"{token}.png")
                # Convert heatmap to rgb color
                # normalize
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                cmap = plt.get_cmap('inferno')
                heatmap = cmap(heatmap) * 255
                print(heatmap)

                # print(heatmap)
                # Make an image mobject for each heatmap
                print(heatmap.shape)
                heatmap = ImageMobject(heatmap)
                heatmap.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
                heatmap_dict[token] = heatmap

    return image, heatmap_dict

# Make the scene
config.pixel_height = 1200
config.pixel_width = 1900
config.frame_height = 30.0
config.frame_width = 30.0

class StableDiffusionCrossAttentionScene(Scene):

    def construct(self):
        # Load the pipeline
        model_id = 'stabilityai/stable-diffusion-2-base'
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        prompt = "Astronaut riding a horse on the moon"
        # Compute the images and heatmaps
        image, heatmap_dict = compute_stable_diffusion_cross_attention_heatmaps(pipe, prompt)
        # 1. Show an image appearing
        image_mobject = ImageMobject(image)
        image_mobject.shift(DOWN)
        image_mobject.scale(0.7)
        image_mobject.shift(LEFT * 7)
        self.add(image_mobject)
        # 1. Show a text prompt and the corresponding generated image
        paragraph_prompt = '"Astronaut riding a\nhorse on the moon"'
        text_prompt = Paragraph(paragraph_prompt, alignment="center", font_size=64)
        text_prompt.next_to(image_mobject, UP, buff=1.5)
        prompt_title = Text("Prompt", font_size=72)
        prompt_title.next_to(text_prompt, UP, buff=0.5)
        self.play(Create(prompt_title))
        self.play(Create(text_prompt))        
        # Make an arrow from the text to the image
        arrow = Arrow(text_prompt.get_bottom(), image_mobject.get_top(), buff=0.5)
        self.play(GrowArrow(arrow))
        self.wait(1)
        self.remove(arrow)
        # 2. Show the image being split into patches
        # Make the patches
        patches = ImagePatches(image_mobject, grid_width=32)
        # Expand and contract 
        self.remove(image_mobject)
        self.play(ExpandPatches(patches, expansion_scale=1.2))
        # Play arrows for visual tokens
        visual_token_label = Text("Visual Tokens", font_size=72)
        visual_token_label.next_to(image_mobject, DOWN, buff=1.5)
        # Draw arrows 
        arrow_animations = []
        arrows = []
        for i in range(patches.grid_width):
            patch = patches.patch_dict[(patches.grid_width - 1, i)]
            arrow = Line(visual_token_label.get_top(), patch.get_bottom(), buff=0.3)
            arrow_animations.append(
                Create(
                    arrow
                )
            )
            arrows.append(arrow)
        self.play(AnimationGroup(*arrow_animations, FadeIn(visual_token_label), lag_ratio=0))
        self.wait(1)
        self.play(FadeOut(*arrows, visual_token_label))
        self.play(ExpandPatches(patches, expansion_scale=1/1.2))
        # 3. Show the text prompt and image being tokenized.
        tokenized_text = TokenizedText(prompt, pipe.tokenizer)
        tokenized_text.shift(RIGHT * 7)
        tokenized_text.shift(DOWN * 7.5)
        self.play(FadeIn(tokenized_text))
        # Plot token label
        token_label = Text("Textual Tokens", font_size=72)
        token_label.next_to(tokenized_text, DOWN, buff=0.5)
        arrow_animations = []
        self.play(Create(token_label)) 
        arrows = []
        for token_name, token_mobject in tokenized_text.token_dict.items():
            arrow = Line(token_label.get_top(), token_mobject.get_bottom(), buff=0.3)
            arrow_animations.append(
                Create(
                    arrow
                )
            )
            arrows.append(arrow)
        self.play(AnimationGroup(*arrow_animations, lag_ratio=0))
        self.wait(1)
        self.play(FadeOut(*arrows, token_label))
        # 4. Show the heatmap corresponding to the cross attention map for each image. 
        surrounding_rectangle = SurroundingRectangle(tokenized_text.token_dict["astronaut"], buff=0.1)
        self.add(surrounding_rectangle)
        # Show the heatmap for "astronaut"
        astronaut_heatmap = heatmap_dict["astronaut"]
        astronaut_heatmap.height = image_mobject.get_height()
        astronaut_heatmap.shift(RIGHT * 7)
        astronaut_heatmap.shift(DOWN)
        self.play(FadeIn(astronaut_heatmap))
        self.wait(3)
        self.remove(surrounding_rectangle)
        surrounding_rectangle = SurroundingRectangle(tokenized_text.token_dict["horse"], buff=0.1)
        self.add(surrounding_rectangle)
        # Show the heatmap for "horse"
        horse_heatmap = heatmap_dict["horse"]
        horse_heatmap.height = image_mobject.get_height()
        horse_heatmap.move_to(astronaut_heatmap)
        self.play(FadeOut(astronaut_heatmap))
        self.play(FadeIn(horse_heatmap))
        self.wait(3)
        self.remove(surrounding_rectangle)
        surrounding_rectangle = SurroundingRectangle(tokenized_text.token_dict["riding"], buff=0.1)
        self.add(surrounding_rectangle)
        # Show the heatmap for "riding"
        riding_heatmap = heatmap_dict["riding"]
        riding_heatmap.height = image_mobject.get_height()
        riding_heatmap.move_to(horse_heatmap)
        self.play(FadeOut(horse_heatmap))
        self.play(FadeIn(riding_heatmap))
        self.wait(3)
