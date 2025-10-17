"""
This module provides tools for logging and visualizing training metrics via Python's logging library and TensorBoard. 
It facilitates easy tracking of training progress through detailed logs and visual summaries.
"""

import os
import logging
import wandb
import socket
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from typing import List
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from utils.plot import plot_spectrogram_to_numpy


# State variables to manage logger and TensorBoard initialization.
logger_initialized = False
tensorboard_writer = None

logging.basicConfig(level=logging.INFO)
info = logging.info


def init(
    log_directory: Path,
    enable_tensorboard: bool = True,
    date: str = "",
    enable_wandb: bool = False,
    runs_name: str = "",
    config: DictConfig = None,
    project: str = "test",
):
    """
    Initialize file-based logging and, optionally, TensorBoard for training visualization.

    Parameters:
    - log_directory (Path): Path to the directory where logs will be stored.
    - enable_tensorboard (bool): If True, initializes TensorBoard logging. Defaults to True.
    - enable_wandb (bool): If True, initializes wandb logging. Defaults to False.

    This function sets up a file-based logger and, if specified, a TensorBoard SummaryWriter.
    """
    global logger_initialized, tensorboard_writer, logger, debug, info, warn, error

    basename = os.path.basename(log_directory)

    if not logger_initialized:
        # Ensure the log directory exists.
        os.makedirs(log_directory, exist_ok=True)

        # Set up a custom logger for application-wide logging.
        logger = logging.getLogger("TrainingLogger")
        logger.setLevel(logging.INFO)
        if date == "":
            date = f"{datetime.now():%Y%m%d-%H%M%S}"
        log_file = f"{log_directory}/{date}.log"
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler for logging to the console
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter("[%(levelname)s] | %(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # Assign simplified access to logging functions.
        debug, info, warn, error = (
            logger.debug,
            logger.info,
            logger.warning,
            logger.error,
        )

        # Prevent re-initialization.
        logger_initialized = True

    # Set up TensorBoard if requested and not previously initialized.
    if enable_tensorboard and tensorboard_writer is None:
        tensorboard_writer = SummaryWriter(log_directory)

    # Set up wandb if requested
    if enable_wandb:
        wandb.init(
            project=project,
            sync_tensorboard=False,
            notes=socket.gethostname(),
            name=runs_name,
            dir=log_directory,
            job_type="training",
            resume=True,
        )


def write_loss(tag: str, loss_metrics: dict, training_step: int):
    """
    Log loss metrics to TensorBoard under a specific tag.

    Parameters:
    - tag (str): The tag under which the metrics are grouped.
    - loss_metrics (dict): A dictionary containing the metric names and their corresponding loss values.
    - training_step (int): The current training step for associating the metrics.

    If TensorBoard is initialized, this function logs each metric under the given tag.
    """
    if tensorboard_writer is None:
        warn("TensorBoard is not initialized, skipping metric logging.")
        return

    for metric_name, metric_value in loss_metrics.items():
        tensorboard_writer.add_scalar(
            f"{tag}/{metric_name}", metric_value, training_step
        )

    if wandb.run is not None:
        wandb.log(
            {f"{tag}/{key}": value for key, value in loss_metrics.items()},
            step=training_step,
        )


def write_audio(tag: str, audio: np.ndarray, sample_rate: int, training_step: int):
    """
    Log audio to TensorBoard under a specific tag.

    Parameters:
    - tag (str): The tag under which the audio are grouped.
    - sample_rate (int): audio sampling rate
    - training_step (int): The current training step for associating the metrics.

    If TensorBoard is initialized, this function logs each metric under the given tag.
    """

    # Due to the large saved tensorbaord file, the audio will not be saved to tensorboard
    return

    if tensorboard_writer is None:
        warn("TensorBoard is not initialized, skipping metric logging.")
        return

    tensorboard_writer.add_audio(
        tag, audio, sample_rate=sample_rate, global_step=training_step
    )


def write_plot(tag: str, data: List[np.ndarray], label: List[str], training_step: int):
    """
    Log plot to TensorBoard under a specific tag.

    Parameters:
    - tag (str): The tag under which the plot are grouped.
    - data (List[np.ndarray]): example [predicts, ground-truth]
    - label (List[str]): example ['predict', 'ground-truth']
    - training_step (int): The current training step for associating the metrics.

    If TensorBoard is initialized, this function logs each plot under the given tag.
    """
    if tensorboard_writer is None:
        warn("TensorBoard is not initialized, skipping metric logging.")
        return

    fig = plt.figure()
    for data, lab in zip(data, label):
        plt.plot(data, label=lab)
    plt.legend()
    tensorboard_writer.add_figure(tag, fig, global_step=training_step)


def write_mel(tag: str, mel: np.ndarray, training_step: int):
    """
    Log mel spectrogram to TensorBoard under a specific tag.

    Parameters:
    - tag (str): The tag under which the mel spectrogram are grouped.
    - mel (np.ndarray): mel spectrogram
    - label (List[str]): example ['predict', 'ground-truth']
    - training_step (int): The current training step for associating the metrics.
    """
    if tensorboard_writer is None:
        warn("TensorBoard is not initialized, skipping metric logging.")
        return
    data = plot_spectrogram_to_numpy(mel)
    tensorboard_writer.add_image(
        tag, data, global_step=training_step, dataformats="HWC"
    )


# test
if __name__ == "__main__":
    init("egs/results")
    info("test")
