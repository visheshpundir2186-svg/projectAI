import cv2
import numpy as np
from PIL import Image

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to a satellite / remote sensing image.

    Why CLAHE for groundwater imagery:
    1. Enhances subtle micro-contrast in vegetation stress and dry soil.
    2. Sharpens boundaries between parched land and water bodies.
    3. Prevents noise amplification in uniform regions (clouds, deep water).

    Converts RGB -> LAB color space and applies CLAHE strictly to the Luminance (L)
    channel to preserve spectral color balance.
    """
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        img_np = np.array(img.convert('RGB'))
    else:
        img_np = np.array(img)

    # Convert to LAB
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    # Merge channels and convert back to RGB
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    if is_pil:
        return Image.fromarray(rgb_enhanced)
    return rgb_enhanced


class CLAHETransform:
    """
    PyTorch Torchvision-compatible transform for CLAHE.
    Can be used directly inside torchvision.transforms.Compose.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        return apply_clahe(img, self.clip_limit, self.tile_grid_size)
