"""
This module provides tools for plot various figures.
"""
from typing import List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(
    data: np.ndarray, figsize=(10, 6), color="blue", bins=50, alpha=0.7
):
    """
    plots a histogram for given numpy array.
    """
    # Plotting the histogram
    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, color=color, alpha=alpha)
    plt.title("Distribution")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_mel(array, path, fig_size=(36, 12), dpi=150):
    """
    plot mel and save it to path
    shape is (  # batch, # channel, # time step)
    """
    assert len(array.shape) == 3
    array = array[:1]  # we only use 1 of one batch
    plt.figure(figsize=(fig_size[0] * array.shape[0], fig_size[1]), dpi=dpi)
    for idx, x in enumerate(array, start=1):
        plt.subplot(array.shape[0], 1, idx)
        plt.imshow(x, aspect="auto", origin="lower", interpolation="none")
        plt.title("batch-{}".format(idx))
        plt.xlabel("Frames")
        plt.ylabel("Channels")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_spectrogram_to_numpy(spectrogram):
    """
    plot mel and return data
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
