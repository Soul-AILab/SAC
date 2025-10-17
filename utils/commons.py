"""
Description:
    This script contains a collection of commonly used functions.
"""


import os
import torch
import numpy as np
from scipy import interpolate

ROOT_PATH = os.path.dirname(__file__).replace("/utils", "")


def scalar_to_dist(data: torch.Tensor, max_value: int):
    """convert scalar data to one-hot distribution.

    Args:
        data (torch.Tensor). (L,), a 1d sequence values.
        max_value: higher than this value will be cut-off

    Return:
        torch.Tensor: (L, max_value + 1)
    """

    dist = torch.zeros(max_value + 1, len(data))
    value = torch.ones(len(data))
    dist = dist.scatter(0, data.long().unsqueeze(0), value.unsqueeze(0))
    return dist


def interpolate1d(data: np.ndarray, factor: float) -> np.ndarray:
    """
    Perform upsampling or downsampling on a 1D array using linear interpolation.

    Args:
        data (np.ndarray): The input 1D array to be resampled.
        factor (float): Scaling factor for resampling. A value greater than 1.0 upsamples the data
                       (increases the number of points), while a value less than 1.0 downsamples it.

    Returns:
        np.ndarray: The resampled array with the new length determined by the original length and the scaling factor.
    """
    if factor == 1:
        return data
    origin_len = len(data)
    target_len = int(origin_len * factor)
    original_indices = np.arange(origin_len)
    new_indices = np.linspace(0, origin_len - 1, target_len)

    interpolator = interpolate.interp1d(original_indices, data, kind="linear")
    resampled_data = interpolator(new_indices)

    return resampled_data


def test_successful():
    print("+--------------------+\n|   Test Successful  |\n+--------------------+")


def shuffle_chunks(feature_tensor, chunk_size=50):
    # Get the dimensions of the tensor
    D, T = feature_tensor.shape

    # Calculate the number of chunks needed
    num_chunks = (
        T + chunk_size - 1
    ) // chunk_size  # Ensures we account for the last chunk even if it's smaller

    # Create an array of indices to indicate chunk boundaries
    indices = torch.arange(0, T + 1, chunk_size)
    if indices[-1] != T:
        indices = torch.cat(
            (indices, torch.tensor([T]))
        )  # Ensure the last index is included

    # Shuffle the indices of the chunks (excluding the last one if it's a partial chunk)
    shuffled_indices = torch.randperm(num_chunks)

    # Initialize the tensor for shuffled features
    shuffled_tensor = torch.zeros_like(feature_tensor)

    # Current position in the shuffled tensor
    current_position = 0

    # Shuffle the chunks
    for i in shuffled_indices:
        start_idx = indices[i]
        end_idx = indices[i + 1]
        chunk_length = end_idx - start_idx
        shuffled_tensor[:, current_position : current_position + chunk_length] = (
            feature_tensor[:, start_idx:end_idx]
        )
        current_position += chunk_length

    return shuffled_tensor
