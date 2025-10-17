"""
Description:
    This script contains a collection of functions designed to handle various
    file reading and writing operations. It provides utilities to read from files,
    write data to files, and perform file manipulation tasks.
"""

import os
import json

from tqdm import tqdm
from typing import List, Dict
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def resolve_symbolic_link(symbolic_link_path: Path):
    """
    Resolves the absolute path of a symbolic link.

    Args:
        symbolic_link_path (Path): The path to the symbolic link.

    Returns:
        Path: The absolute path that the symbolic link points to.
    """

    link_directory = os.path.dirname(symbolic_link_path)
    target_path_relative = os.readlink(symbolic_link_path)
    return os.path.join(link_directory, target_path_relative)


def create_symbolic_link(
    target_file_path: Path, symbolic_link_name: Path, overwrite=True
):
    """
    Creates or updates a symbolic link pointing to a target file.

    Args:
        target_file_path (Path): The path to the file that the symbolic link will point to.
        symbolic_link_name (Path): The name of the symbolic link to be created or updated.
        overwrite (bool): If True, overwrite an existing symbolic link; otherwise, do nothing if it exists.
    """
    target_directory = os.path.dirname(target_file_path)
    symbolic_link_path = os.path.join(target_directory, symbolic_link_name)
    if os.path.islink(symbolic_link_path) and overwrite:
        os.remove(symbolic_link_path)
    os.symlink(os.path.basename(target_file_path), symbolic_link_path)



def write_jsonl(metadata: List[dict], file_path: Path):
    """Writes a list of dictionaries to a JSONL file.

    Args:
    metadata : List[dict]
        A list of dictionaries, each representing a piece of meta.
    file_path : Path
        The file path to save the JSONL file

    This function writes each dictionary in the list to a new line in the specified file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for meta in tqdm(metadata, desc="writing jsonl"):
            # Convert dictionary to JSON string and write it to the file with a newline
            json_str = json.dumps(meta, ensure_ascii=False) + "\n"
            f.write(json_str)
    print(f"jsonl saved to {file_path}")


def read_jsonl(file_path: Path) -> List[dict]:
    """
    Reads a JSONL file and returns a list of dictionaries.

    Args:
    file_path : Path
        The path to the JSONL file to be read.

    Returns:
    List[dict]
        A list of dictionaries parsed from each line of the JSONL file.
    """
    metadata = []
    # Open the file for reading
    with open(file_path, "r", encoding="utf-8") as f:
        # Split the file into lines
        lines = f.read().splitlines()
    # Process each line
    for line in lines:
        # Convert JSON string back to dictionary and append to list
        meta = json.loads(line)
        metadata.append(meta)
    # Return the list of metadata
    return metadata


def decode_unicode_strings(meta):
    processed_meta = {}
    for k, v in meta.items():
        if isinstance(v, str):
            processed_meta[k] = v.encode("utf-8").decode("unicode_escape")
        else:
            processed_meta[k] = v
    return processed_meta


def load_config(config_path: Path) -> DictConfig:
    """Loads a configuration file and optionally merges it with a base configuration.

    Args:
    config_path (Path): Path to the configuration file.
    """
    # Load the initial configuration from the given path
    config = OmegaConf.load(config_path)

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(config["base_config"])
        config = OmegaConf.merge(base_config, config)

    return config
