"""
Plotting Utilities

This module defines various utility functions for plotting images and graphs.
"""

from typing import Optional, Tuple

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
    is_one_channel: bool = False,
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
        is_one_channel (bool):\
            Flag indicating if monochromatic image is to plotted. Defaults to False.
    """
    # Ensure that one channel is used, if activated
    if is_one_channel:
        img = img.mean(dim=0)
        
    # Denormalize the image tensor, if normalized
    if is_norm:
        img = tensor_utils.denormalize(img, mean=mean, std=std)

    # Plot the image
    if is_one_channel:
        plt.imshow(img.numpy(), cmap='Greys')
    else:
        plt.imshow(np.transpose(img.numpy().squeeze(), (1, 2, 0)))
    plt.title(label)


def relplot(
    x: Tensor,
    y: Tensor,
    *,
    xlabel: Optional[str],
    ylabel: Optional[str],
    title: Optional[str],
    figsize: Tuple[int, int] = (6.4, 4.8),
) -> None:
    """
    Plots the relationship between two 1-D tensors.

    Args:
        x (Tensor): The tensor on the x-axis.
        y (Tensor): The tensor on the y-axis.
        xlabel (Optional[str]): The label on the x-axis.
        ylabel (Optional[str]): The label on the y-axis.
        title (Optional[str]): The title of the plot.
        figsize (Tuple[int, int], optional):\
            The size of the plot. Defaults to (6.4, 4.8).
    """
    plt.figure(figsize=figsize)
    plt.plot(x, y, '--')
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(title)
    plt.grid()
    plt.show()
