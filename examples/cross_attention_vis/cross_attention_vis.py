"""
    Here I thought it would be interesting to visualize the cross attention
    maps produced by the UNet of a text-to-image diffusion model. 

    The key thing I want to show is how images and text are broken up into tokens (patches for images),
    and those tokens are used to compute a cross attention score, which can be interpreted
    as a 2D heatmap over the image patches for each text token. 
    

    Necessary operations:
        1. Split an image into patches and "expand" the patches outward to highlight that they are 
            separate. 
        2. Split text into tokens using the tokenizer and display them. 
        3. Compute the cross attention scores (using DAAM) for each word/token and display them as a heatmap.
        4. Overlay the heatmap over the image patches. Show the overlaying as a transition animation. 
"""

from manim import *
import numpy as np

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
        self._split_image_into_patches(image, grid_width)

    def _split_image_into_patches(self, image, grid_width):
        """Splits the images into a set of patches

        Parameters
        ----------
        image : ImageMobject
            image to split into patches
        grid_width : int
            number of patches per row
        """
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
                print(f"Image width: {image_width}")
                print(f"Image center: {image_center}")
                print(patch_width)
                image_patch.scale(image_width / grid_width / image_patch.get_width())
                patch_manim_space_width = image_patch.get_width()

                patch_center = image_center
                patch_center += (patch_j - self.grid_width / 2 + 0.5) * patch_manim_space_width * RIGHT
                # patch_center = image_center - (patch_i - self.grid_width / 2 + 0.5) * patch_manim_space_width * RIGHT
                patch_center -= (patch_i - self.grid_width / 2 + 0.5) * patch_manim_space_width * UP
                # + (patch_j - self.grid_width / 2) * patch_height / 2 * UP

                image_patch.move_to(patch_center)

                self.add(image_patch)

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
        alpha : _type_
            interpolation alpha
        """
        patch = submobject
        starting_center = starting_submobject.get_center()
        # Need to compute the direction of expansion as the unit vector from the original image center
        # and patch center. 
        expansion_direction = submobject.get_center() - self.image_patches.image.get_center()
        expansion_direction /= np.linalg.norm(expansion_direction)
        # Compute the patch final location
        final_location = starting_center + expansion_direction * self.expansion_scale * patch.get_width() / 2
        # Compute the current location by interpolating between staring and final locations
        current_location = alpha * final_location + (1 - alpha) * starting_center
        patch.move_to(current_location)
