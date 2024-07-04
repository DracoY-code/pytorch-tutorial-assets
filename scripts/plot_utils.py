"""
Plotting Utilities

This module defines various utility functions for plotting images and graphs.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

import tensor_utils


def plot_image(
    img: Tensor,
    is_norm: bool = True,
    *,
    label: Optional[str] = None,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
) -> None:
    """
    Plots the image.

    If normalized, the image tensor is denormalized with the
    provided mean and standard deviation values.

    Args:
        img (Tensor): The image tensor.
        is_norm (bool, optional): Flag indicating normalization. Defaults to True.
        label (Optional[str], optional): Label of the image. Defaults to None.
        mean (Optional[Tensor], optional):\
            Mean values for each channel. Defaults to None.
        std (Optional[Tensor], optional):\
            Standard deviation values for each channel. Defaults to None.
    """
    # Denormalize the image tensor, if normalized
    if is_norm:
        img = tensor_utils.denormalize(img, mean=mean, std=std)

    # Plot the image
    plt.imshow(np.transpose(img.numpy().squeeze(), (1, 2, 0)))
    plt.title(label)
