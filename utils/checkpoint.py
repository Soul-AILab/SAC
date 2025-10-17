import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn
import utils.log as log
from omegaconf import OmegaConf
from utils.file import create_symbolic_link, resolve_symbolic_link


def resume_checkpoint(models: Dict[str, nn.Module], model_dir: Path, resume_step: int, skip_models: list = None):
    if resume_step == -1:
        ckpt_path = resolve_symbolic_link(f"{model_dir}/last.pt")
    else:
        resume_step = str(resume_step).zfill(6)
        ckpt_path = f"{model_dir}/{resume_step}.pt"

    if not os.path.isfile(ckpt_path):
        raise ValueError(f"Not find checkpoint for {resume_step} step: {ckpt_path}")

    log.info("Resume: loading from checkpoint {}".format(ckpt_path))

    state_dict = {}
    state_dict.update(torch.load(ckpt_path, map_location="cpu"))

    for k in models.keys():
        if k in skip_models: continue
        missing_keys, unexpected_keys = models[k].load_state_dict(
            state_dict[k], strict=False
        )
        for key in missing_keys:
            log.info("missing tensor in {}: {}".format(k, key))
        for key in unexpected_keys:
            log.info("unexpected tensor in {}: {}".format(k, key))

    optim_conf = {}
    cfg_path = ckpt_path.replace(".pt", ".yaml")
    if os.path.isfile(cfg_path):
        config = OmegaConf.load(cfg_path)

    return models, config

def resume_ema_checkpoint(ema_model: nn.Module, model_dir: Path, resume_step: int, key_name: str = "ema_generator"):
    if resume_step == -1:
        ckpt_path = resolve_symbolic_link(f"{model_dir}/last.pt")
    else:
        resume_step = str(resume_step).zfill(6)
        ckpt_path = f"{model_dir}/{resume_step}.pt"

    if not os.path.isfile(ckpt_path):
        raise ValueError(f"Not find EMA checkpoint for {resume_step} step: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = state_dict.get(key_name, state_dict)
    log.info("Resume: loading EMA from checkpoint {}".format(ckpt_path))

    missing_keys, unexpected_keys = ema_model.load_state_dict(
        state_dict, strict=False
    )
    for key in missing_keys:
        log.info("missing tensor in ema model: {}".format(key))
    for key in unexpected_keys:
        log.info("unexpected tensor in ema model: {}".format(key))

    return ema_model


def load_checkpoint(
        models: Union[Dict[str, nn.Module], nn.Module], 
        ckpt_path: Path
    ):
    """
    Load models' states from a checkpoint file.

    Args:
        models (Union[Dict[str, nn.Module], nn.Module]): A single model or a dictionary of models to load states into.
        ckpt_path (Path): The path of the checkpoint file from which to load states.
    """
    log.info("Checkpoint: loading from checkpoint %s" % ckpt_path)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    if isinstance(models, dict):
        # If models is a dictionary, load state dictionaries individually
        for k in models:
            missing_keys, unexpected_keys = models[k].load_state_dict(
                state_dict[k], strict=False
            )
            # Log missing and unexpected tensors
            for key in missing_keys:
                log.info("missing tensor in model_{}: {}".format(k, key))
            for key in unexpected_keys:
                log.info("unexpected tensor in model_{}: {}".format(k, key))
    else:
        # If a single model is provided, load the state dictionary directly
        missing_keys, unexpected_keys = models.load_state_dict(
                state_dict, strict=False
            )
        # Log missing and unexpected tensors
        for key in missing_keys:
            log.info("missing tensor in model: {}".format(key))
        for key in unexpected_keys:
            log.info("unexpected tensor in model: {}".format(key))


def save_checkpoints(
        models: Union[Dict[str, nn.Module], nn.Module], 
        ckpt_path: Path
    ):
    """
    Saves the state dictionaries of provided models to a checkpoint file.

    Args:
        models (Union[Dict[str, nn.Module], nn.Module]): A model or a dictionary of models to save.
        ckpt_path (Path): The path to save the checkpoint file.
    """

    log.info("Checkpoint: save to checkpoint %s" % ckpt_path)

    if isinstance(models, dict):
        state_dict = {k: model.state_dict() for k, model in models.items()}
    else:
        state_dict = models.state_dict() 

    torch.save(state_dict, ckpt_path)
    

def filter_modules(model_state_dict, modules):
    new_mods = []
    incorrect_mods = []
    mods_model = model_state_dict.keys()
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]
    if incorrect_mods:
        log.warning(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        log.warning("for information, the existing modules in model are:")
        log.warning("%s", mods_model)

    return new_mods


def load_trained_modules(model: torch.nn.Module, args: None):
    # Load encoder modules with pre-trained model(s).
    enc_model_path = args.enc_init
    enc_modules = args.enc_init_mods
    main_state_dict = model.state_dict()
    log.warning("model(s) found for pre-initialization")
    if os.path.isfile(enc_model_path):
        log.info("Checkpoint: loading from checkpoint %s for CPU" % enc_model_path)
        model_state_dict = torch.load(enc_model_path, map_location="cpu")
        modules = filter_modules(model_state_dict, enc_modules)
        partial_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            if any(key.startswith(m) for m in modules):
                partial_state_dict[key] = value
        main_state_dict.update(partial_state_dict)
    else:
        log.warning("model was not found : %s", enc_model_path)

    model.load_state_dict(main_state_dict)
    configs = {}
    return configs


def clean_stale_checkpoints(directory: Path, latest_checkpoint: str, interval: int = 100000):
    """
    Removes old model files that do not meet the retention interval criteria, except the latest model file.
    
    Args:
        directory (Path): The directory containing model files.
        latest_checkpoint (str): The filename of the latest model to keep.
        interval (int): The interval at which to keep model files (based on the numeric value in their filenames).
    """
    for filename in os.listdir(directory):
        if os.path.islink(f'{directory}/{filename}'): continue
        if '.pt' not in filename and '.yaml' not in filename: continue
        basensme = os.path.splitext(filename)[0]
        if basensme != latest_checkpoint and int(basensme) % interval != 0:
            os.remove(os.path.join(directory, filename))


def strip_prefix(state_dict, prefix: str):
    """
    Remove a common string prefix from all keys in a state_dict, if present.
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out