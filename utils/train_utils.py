from __future__ import print_function
import copy
import random
import hydra
import os
from pathlib import Path
from typing import Dict
from omegaconf import OmegaConf

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live,
)
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

import utils.log as log
from utils.checkpoint import save_checkpoints
from utils.scheduler import (
    WarmupLR,
    NoamHoldAnnealing,
    WarmupAnnealSteps,
    ExponentialLR,
    NoamLR,
    WarmupLRX,
)
from utils.file import create_symbolic_link
from utils.checkpoint import resume_checkpoint, load_checkpoint, clean_stale_checkpoints


def add_model_args(parser):
    parser.add_argument("-c", "--config", required=True, help="config file")
    parser.add_argument(
        "-r",
        "--resume_step",
        default=0,
        type=int,
        help="resume checkpoint, -1 for the last checkpoint",
    )
    parser.add_argument("--log_dir", required=True, help="save results dir")
    parser.add_argument("--checkpoint", help="checkpoint of models")
    parser.add_argument("--seed", default=1234)
    return parser


def add_dataset_args(parser):
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="num of subprocess workers for reading",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="Use pinned memory buffers used for reading",
    )
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument("--persistent_workers", default=True, type=bool)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)
    return parser


def add_ddp_args(parser):
    parser.add_argument(
        "--ddp.dist_backend",
        dest="dist_backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="distributed backend",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Use automatic mixed precision training",
    )
    parser.add_argument(
        "--fp16_grad_sync",
        action="store_true",
        default=False,
        help="Use fp16 gradient sync for ddp",
    )
    return parser


def add_deepspeed_args(parser):
    parser.add_argument(
        "--timeout", default=30, type=int, help="timeout (in seconds) of group_join. "
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--deepspeed.save_states",
        dest="save_states",
        default="model+optimizer",
        choices=["model_only", "model+optimizer"],
        help="save model/optimizer states",
    )
    # DeepSpeed automaticly add '--deepspeed' and '--deepspeed_config' to parser
    # An example of the --deepspeed_config can be found in egs/wav_codec/rwc/config/ds.json
    parser = deepspeed.add_config_arguments(parser)
    return parser


def init_distributed(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    log.info(
        "training on multiple gpus, this gpu {}".format(local_rank)
        + ", rank {}, world_size {}".format(rank, world_size)
    )
    if args.train_engine == "torch_ddp":
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend)
    elif args.train_engine == "deepspeed":
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    else:
        log.error("not supported engine: {}".format(args.train_engine))
    return world_size, local_rank, rank


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_update_and_save_config(args, config):
    config["train_engine"] = args.train_engine

    if args.train_engine == "torch_ddp":
        if args.use_amp:
            config["dtype"] = "fp16"
        else:
            config["dtype"] = "fp32"
    elif args.train_engine == "deepspeed":
        assert args.deepspeed_config is not None
        ds_config = OmegaConf.load(args.deepspeed_config)
        config["ds_config"] = ds_config
        if "fp16" in ds_config and ds_config["fp16"]["enabled"]:
            config["dtype"] = "fp16"
        elif "bf16" in ds_config and ds_config["bf16"]["enabled"]:
            config["dtype"] = "bf16"
        else:
            config["dtype"] = "fp32"
        assert ds_config["train_micro_batch_size_per_gpu"] == 1
        assert ds_config["gradient_clipping"] == config["grad_clip"]
    else:
        raise ValueError(f"only torch_ddp and deepspeed are supported")

    # update config based on arguments
    assert args.log_dir is not None
    assert args.use_amp is not None
    assert args.save_states is not None
    assert args.seed is not None
    config["seed"] = args.seed
    config["use_amp"] = args.use_amp
    config["log_dir"] = args.log_dir
    config["model_dir"] = f"{args.log_dir}/ckpt"
    config["save_states"] = args.save_states
    if not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"], exist_ok=True)

    if args.train_data is not None:
        config["datasets"]["train"] = args.train_data
    if args.val_data is not None:
        config["datasets"]["val"] = args.val_data

    # Save config to log_dir/config.yaml for inference and export
    if int(os.environ.get("RANK", 0)) == 0:
        saved_config_path = os.path.join(args.log_dir, "config.yaml")
        with open(saved_config_path, "w") as fout:
            data = OmegaConf.to_yaml(config)
            fout.write(data)

        log.info(f"updated config is save to {saved_config_path}")

    return config


def init_dataset_and_dataloader(args, config, seed=777):
    generator = torch.Generator()
    generator.manual_seed(seed)

    # total_batches = estimate_total_batches(args.utts_num, config['static']['batch_size'])
    train_dataset_sampler = hydra.utils.instantiate(config["dataloader"], config)
    val_dataset_sampler = hydra.utils.instantiate(
        config["dataloader"], config, mode="val"
    )
    train_dataset = train_dataset_sampler.sample()
    val_dataset = val_dataset_sampler.sample()

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=None,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        generator=generator,
        prefetch_factor=args.prefetch,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=None,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        persistent_workers=True,
        generator=generator,
        prefetch_factor=args.prefetch,
    )

    return train_data_loader, val_data_loader


def init_model(model_config):
    # Using Hydra to achieve recursive instantiation of models.
    # NOTE(xinsheng): Older versions of Hydra may not support recursive instantiation.
    #      Support version: hydra-core==1.3.2
    return hydra.utils.instantiate(model_config)


def init_models(args, config):
    # Managing multi models with a dictionary.
    models = {}
    for model_name in config.model.keys():
        models[model_name] = init_model(config.model[model_name])

    skip_models = [k for k in config["model"] if config["model"][k]["no_grad"]]

    if args.resume_step != 0:
        models, history_config = resume_checkpoint(
            models, config["model_dir"], args.resume_step, skip_models
        )
        step, epoch = history_config["current_step"], history_config["current_epoch"]
        log.info(
            "resume checkpoint from {} with step {}".format(config["model_dir"], step)
        )
        config["current_step"] = step
        config["current_epoch"] = epoch

    # load checkpoint from pre-trained model
    elif args.checkpoint is not None:
        models = load_checkpoint(models, args.checkpoint)
        log.info("load pre-trained generator model from {}".format(args.checkpoint))

    # load checkoint from pre-trained model respectively
    for model_name in config.model:
        if config.model[model_name]["checkpoint"] is not None:
            state_dict = torch.load(
                config.model[model_name]["checkpoint"], map_location="cpu"
            )[model_name]
            missing_keys, unexpected_keys = models[model_name].load_state_dict(
                state_dict, strict=False
            )
            for key in missing_keys:
                print("missing tensor {}".format(key))
            for key in unexpected_keys:
                print("unexpected tensor {}".format(key))
    return models, config


def freeze_model_parameters(models, config):
    # Disable gradient computations for parameters of specific models as defined in the configuration.
    for model_name in config.model:
        if config.model[model_name].get("no_grad", False):
            for param in models[model_name].parameters():
                param.requires_grad = False


def params_statistic(models: Dict[str, nn.Module]):
    # statistic model parameter number
    num_param_dict = {}
    num_trainable_param_dict = {}

    for model_name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        num_param_dict[model_name] = total_params
        num_trainable_param_dict[model_name] = trainable_params

    total_num_params = sum(num_param_dict.values())
    total_num_trainable_params = sum(num_trainable_param_dict.values())

    num_param_info = "".join([
        "{} params scale {:.2f}M, trainable {:.2f}M\n".format(
            k, num_param_dict[k], num_trainable_param_dict[k]
        )
        for k in models.keys()
    ])
    num_param_info += "total params scale {:.2f}M, trainable {:.2f}M\n".format(
        total_num_params, total_num_trainable_params
    )

    if int(os.environ.get("RANK", 0)) == 0:
        log.info("Model parameters statistic:\n{}".format(num_param_info))


def print_model(models):
    if int(os.environ.get("RANK", 0)) == 0:
        for k, model in models.items():
            log.info("Network of {}: \n{}".format(k, model))


def wrap_cuda_model(args, models):
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    for k, model in models.items():
        if args.train_engine == "torch_ddp":  # native pytorch ddp
            assert torch.cuda.is_available()
            model.cuda()
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model)
            if args.fp16_grad_sync:
                from torch.distributed.algorithms.ddp_comm_hooks import (
                    default as comm_hooks,
                )

                model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

        elif args.train_engine == "deepspeed":  # deepspeed
            # look in detail how the memory estimator API works:
            # https://deepspeed.readthedocs.io/en/latest/memory.html#discussion
            if int(os.environ.get("RANK", 0)) == 0:
                log.info("Estimating model states memory needs (zero2)...")
                estimate_zero2_model_states_mem_needs_all_live(
                    model,
                    num_gpus_per_node=local_world_size,
                    num_nodes=world_size // local_world_size,
                )
                log.info("Estimating model states memory needs (zero3)...")
                estimate_zero3_model_states_mem_needs_all_live(
                    model,
                    num_gpus_per_node=local_world_size,
                    num_nodes=world_size // local_world_size,
                )
            pass  # Init DeepSpeed later
        else:
            log.error("not supported engine: {}".format(args.train_engine))


def save_models(models, config):
    rank = int(os.environ.get("RANK", 0))
    tag = str(config["current_step"]).zfill(6)
    model_dir = config["model_dir"]
    save_model_path = os.path.join(model_dir, "{}.pt".format(tag))
    # save ckpt
    if config["train_engine"] == "deepspeed":
        # All ranks should call this API, but only rank 0
        # save the general model params. see:
        # https://github.com/microsoft/DeepSpeed/issues/2993
        state_dict = {}
        if isinstance(models, dict):
            for k, model in models.items():
                sub_tag = f"{tag}_{k}"
                if k == "ema_generator":
                    if rank == 0:
                        torch.save(model.state_dict(), f"{model_dir}/{sub_tag}.pt")
                        state_dict[k] = torch.load(f"{model_dir}/{sub_tag}.pt", map_location="cpu")
                        os.system(f"rm -rf {model_dir}/{sub_tag}.pt")
                    continue
                if config["model"][k]["no_grad"]:
                    continue
                with torch.no_grad():
                    model.save_checkpoint(
                        save_dir=model_dir,
                        tag=sub_tag,
                        client_state={"step": config["current_step"]},
                    )

                    if rank == 0:
                        convert_zero_checkpoint_to_fp32_state_dict(
                            model_dir,
                            "{}/{}.pt".format(model_dir, sub_tag),
                            tag=sub_tag,
                        )

                        state_dict[k] = torch.load(
                            "{}/{}_{}.pt".format(model_dir, tag, k)
                        )
                        os.system("rm -rf {}/{}".format(model_dir, sub_tag))
                        os.system("rm -rf {}/{}".format(model_dir, f"{sub_tag}.pt"))

        else:
            with torch.no_grad():
                model.save_checkpoint(
                    save_dir=model_dir,
                    tag=tag,
                    client_state={"step": config["current_step"]},
                )

                if rank == 0:
                    convert_zero_checkpoint_to_fp32_state_dict(
                        model_dir, "{}/{}.pt".format(model_dir, tag), tag=tag
                    )
                    state_dict = torch.load("{}/{}.pt".format(model_dir, tag))
                    os.system("rm -rf {}/{}".format(model_dir, tag))
                    os.system("rm -rf {}/{}".format(model_dir, f"{tag}.pt"))

        if rank == 0:
            torch.save(state_dict, save_model_path)
            create_symbolic_link(save_model_path, "last.pt")

    elif rank == 0:
        # For torch_ddp, only rank-0 should call this.
        save_checkpoints(models, save_model_path)
        create_symbolic_link(save_model_path, "last.pt")

    # save yaml
    if rank == 0:
        with open("{}/{}.yaml".format(model_dir, tag), "w") as fout:
            data = OmegaConf.to_yaml(config)
            fout.write(data)
        create_symbolic_link("{}/{}.yaml".format(model_dir, tag), "last.yaml")

        clean_stale_checkpoints(model_dir, tag, config["keep_interval"])


def log_per_epoch(writer, config):
    epoch = config["current_epoch"]
    loss_dict = config["loss_dict"]
    if int(os.environ.get("RANK", 0)) == 0:
        writer.add_scalar("epoch/lr", config["current_lr"], epoch)
        for name, value in loss_dict.items():
            writer.add_scalar("epoch/{}".format(name), value, epoch)


def init_optimizer_and_scheduler(args, config, models):
    optimizers = dict()
    schedulers = dict()

    for k, model in models.items():
        if model is not None:
            if config.model[k]["no_grad"]:
                optimizers[k] = None
                schedulers[k] = None
                continue
            if config.model[k]["optim"] == "adam":
                optimizers[k] = optim.Adam(
                    model.parameters(), **config.model[k]["optim_conf"]
                )
            elif config.model[k]["optim"] == "adamw":
                optimizers[k] = optim.AdamW(
                    model.parameters(), **config.model[k]["optim_conf"]
                )
            else:
                raise ValueError("unknown optimizer: " + config.model[k]["optim"])

            if config.model[k]["scheduler"] == "warmuplr":
                schedulers[k] = WarmupLR(
                    optimizers[k], **config.model[k]["scheduler_conf"]
                )
            elif config.model[k]["scheduler"] == "exponentiallr":
                schedulers[k] = ExponentialLR(
                    optimizers[k], **config.model[k]["scheduler_conf"]
                )
            elif config.model[k]["scheduler"] == "NoamHoldAnnealing":
                schedulers[k] = NoamHoldAnnealing(
                    optimizers[k], **config.model[k]["scheduler_conf"]
                )
            elif config.model[k]["scheduler"] == "warmupas":
                schedulers[k] = WarmupAnnealSteps(
                    optimizers[k], **config.model[k]["scheduler_conf"]
                )
            elif config.model[k]["scheduler"] == "noamlr":
                schedulers[k] = NoamLR(
                    optimizers[k], **config.model[k]["scheduler_conf"]
                )
            else:
                raise ValueError("unknown scheduler: " + config.model[k]["scheduler"])

    # Custom optimizer might yield poor performance when
    # zero-offload is enabled, if you do want to offload optimizer to CPU,
    # please set optimizer in ds_config.json, see:
    # (https://www.deepspeed.ai/docs/config-json/#optimizer-parameters)
    if args.train_engine == "deepspeed":
        # Initialize DeepSpeed with a single optimizer for the entire model.
        # The model parameters are divided into multiple groups, such as the generator and discriminator parameters.
        # DeepSpeed automatically handles optimization of different parameter groups, applying separate learning rates,
        # weight decays, and other optimizer settings as specified.

        for k, model in models.items():
            if optimizers[k] is not None:
                models[k], optimizers[k], _, schedulers[k] = deepspeed.initialize(
                    args=args,
                    model=model,
                    optimizer=optimizers[k],
                    lr_scheduler=schedulers[k],
                    model_parameters=model.parameters(),
                )
            else:
                models[k] = model
                optimizers[k] = None
                schedulers[k] = None

    step = config["current_step"]
    [
        scheduler.set_step(step)
        for k, scheduler in schedulers.items()
        if scheduler is not None
    ]

    return models, optimizers, schedulers


def uneven_check(group_join, batch_idx, train_engine, timeout=None):
    # DeepSpeed does not support uneven data. When using custom
    # dataset, we need to manually ensure that the data is evenly distributed
    # across all processe.
    # ref: https://github.com/microsoft/DeepSpeed/issues/2223

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    if batch_idx == 0 or train_engine == "torch_ddp":
        #  skip first batch because its processing time includes
        #  dataloader initialization time, which may exceed 30 seconds
        return False

    try:
        #  Why we need a new group?
        #  Because Deepspeed has its own group where all the relevant communication
        #  operations are executed. If we add a communication operation that is not
        #  managed by Deepspeed in this group, it's highly likely to cause
        #  communication chaos, resulting in hard-to-troubleshoot hangs.
        dist.monitored_barrier(group=group_join, timeout=timeout)
    except RuntimeError as e:
        log.info(
            "Detected uneven workload distribution: {}\n".format(e)
            + "Break current worker to manually join all workers, "
            + "world_size {}, current rank {}, current local_rank {}\n".format(
                world_size, rank, local_rank
            )
        )
        return True

    return False


def toggle_grad(model, require_grads=False):
    for param in model.parameters():
        param.requires_grad = require_grads


def estimate_total_batches(total_num, batch_size):
    return int(total_num // batch_size)
