import warnings
from typing import Any

import numpy as np
import torch.nn.functional as F
from PIL import Image


def resize_tensor_batch(x, target_size=(128, 128), mode="bilinear", align_corners=False):
    """
    Resize a 5D tensor of shape [bs, win_len, H, W, C] to [bs, win_len, target_H, target_W, C].

    Args:
        x (torch.Tensor): Input tensor of shape [bs, win_len, H, W, C].
        target_size (tuple): Target size (height, width), e.g., (128, 128).
        mode (str): Interpolation mode, e.g., 'bilinear', 'nearest'.
        align_corners (bool): Align corners argument for interpolation.

    Returns:
        torch.Tensor: Resized tensor of shape [bs, 64, target_H, target_W, C].
    """
    bs, n, h, w, c = x.shape

    # Step 1: [bs, win_len, H, W, C] → [bs * win_len, C, H, W]
    x = x.view(-1, h, w, c).permute(0, 3, 1, 2)

    # Step 2: Resize
    x_resized = F.interpolate(x, size=target_size, mode=mode, align_corners=align_corners)

    # Step 3: [bs * win_len, C, H', W'] → [bs, win_len, H', W', C]
    h_new, w_new = target_size
    x_resized = x_resized.permute(0, 2, 3, 1).view(bs, n, h_new, w_new, c)

    return x_resized


def resize_numpy_array_pillow(
    input_array: np.ndarray,
    target_shape: tuple[int, int],
    resample: Any | None = None, 
) -> np.ndarray:
   
    if not isinstance(input_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if input_array.ndim < 2 or input_array.ndim > 3:
        raise ValueError(f"Input array must have 2 or 3 dimensions, but got {input_array.ndim}")

    if not (
        isinstance(target_shape, tuple)
        and len(target_shape) == 2
        and all(isinstance(dim, int) and dim > 0 for dim in target_shape)
    ):
        raise ValueError("target_shape must be a tuple of two positive integers (height, width).")

    if resample is None:
        resample = Image.Resampling.LANCZOS

    
    try:
        image = Image.fromarray(input_array)
    except TypeError as e:
        
        if input_array.dtype == np.float64:
            warnings.warn(
                f"Input array dtype is {input_array.dtype}, which might not be directly supported by Pillow. Attempting conversion to float32."
            )
            try:
                image = Image.fromarray(input_array.astype(np.float32))
            except TypeError:
                warnings.warn(
                    "Conversion to float32 also failed. Trying conversion to uint8 after scaling to [0, 255]. This assumes input range is [0, 1]."
                )
                scaled_array = np.clip(input_array * 255, 0, 255).astype(np.uint8)
                image = Image.fromarray(scaled_array)

        elif np.issubdtype(input_array.dtype, np.floating):
            warnings.warn(
                f"Input array dtype is {input_array.dtype}. Trying conversion to uint8 after scaling to [0, 255]. This assumes input range is [0, 1]."
            )
            scaled_array = np.clip(input_array * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(scaled_array)
        else:
            raise TypeError(f"Pillow could not handle input array dtype {input_array.dtype}. Error: {e}")

    target_height, target_width = target_shape
   
    target_size_pillow = (target_width, target_height)


    resized_image = image.resize(target_size_pillow, resample=resample)


    resized_array = np.array(resized_image)

    return resized_array
