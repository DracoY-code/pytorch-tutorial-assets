"""
Tensor Utilities

This module defines various utility functions for tensor manipulation.
"""

from typing import Optional, Sequence, Union

from numpy.typing import ArrayLike

import torch
from torch import SymInt, Tensor
from torch.types import _bool, _dtype, _int, _layout, Device

# Import custom error handlers
from errors import ChannelMismatchError


def create_random_tensor(
    size: Sequence[Union[_int, SymInt]],
    *,
    out: Optional[Tensor] = None,
    dtype: Optional[_dtype] = None,
    layout: Optional[_layout] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[_bool] = False,
    requires_grad: Optional[_bool] = False,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Returns a tensor filled with random values with the shape defined by
    the variable argument `size`.

    Args:
        size (Sequence[Union[_int, SymInt]]):
            Shape of the output tensor.
        out (Optional[Tensor], optional):
            Output tensor. Defaults to None.
        dtype (Optional[_dtype], optional):
            Desired data type of the returned tensor.
            Defaults to None.
        layout (Optional[_layout], optional):
            Desired layout of the returned tensor.
            Defaults to None.
        device (Optional[Device], optional):
            Desired device of the returned tensor.
            Defaults to None.
        pin_memory (Optional[_bool], optional):
            If set, returned tensor would be allocated in pinned memory.
            Defaults to False.
        requires_grad (Optional[_bool], optional):
            If autograd should record operations on the returned tensor.
            Defaults to False.
        seed (Optional[int], optional):
            Seed for the random number generator.
            Defaults to None.

    Returns:
        Tensor: A tensor filled with random values.
    """
    # Set the random seed
    if seed is not None:
        torch.manual_seed(seed)

    # Generate the random tensor
    rand_tensor = torch.rand(
        size, out=out, dtype=dtype, layout=layout,
        device=device, pin_memory=pin_memory,
        requires_grad=requires_grad,
    )

    return rand_tensor


def normalize(x: Tensor, *, mean: ArrayLike, std: ArrayLike) -> Tensor:
    """
    Normalizes the input tensor using the provided mean and standard deviation
    across channels.

    Args:
        x (Tensor): Input tensor to be normalized.
        mean (ArrayLike): Mean values for each channel.
        std (ArrayLike): Standard deviation values for each channel.

    Raises:
        ChannelMismatchError: If the number of channels in the tensor `x` does\
            not match the length of `mean` and `std`.

    Returns:
        Tensor: Normalized tensor with the same shape as `x`.
    """
    # Convert arrays to tensors
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    
    # Check if the number of channels matches
    if x.size(1) != mean.size(0) or x.size(1) != std.size(0):
        raise ChannelMismatchError

    # Normalize the tensor
    return (x - mean) / std


def denormalize(x: Tensor, *, mean: ArrayLike, std: ArrayLike) -> Tensor:
    """
    Denormalizes the input tensor using the provided mean and standard deviation
    across channels.

    Args:
        x (Tensor): Input tensor to be denormalized.
        mean (ArrayLike): Mean values for each channel.
        std (ArrayLike): Standard deviation values for each channel.

    Raises:
        ChannelMismatchError: If the number of channels in the tensor `x` does\
            not match the length of `mean` and `std`.

    Returns:
        Tensor: Denormalized tensor with the same shape as `x`.
    """
    # Convert arrays to tensors
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    
    # Check if the number of channels matches
    if x.size(1) != mean.size(0) or x.size(1) != std.size(0):
        raise ChannelMismatchError

    # Denormalize the tensor
    return x * std + mean
