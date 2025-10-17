"""
Main script for training.
"""

from __future__ import print_function

import argparse
import datetime
import os

import hydra
import torch.distributed as dist
import utils.log as log
from torch.distributed.elastic.multiprocessing.errors import record
from utils.checkpoint import resume_ema_checkpoint
from utils.file import load_config
from utils.train_utils import (
    add_dataset_args,
    add_ddp_args,
    add_deepspeed_args,
    add_model_args,
    check_update_and_save_config,
    freeze_model_parameters,
    init_dataset_and_dataloader,
    init_distributed,
    init_models,
    init_optimizer_and_scheduler,
    params_statistic,
    print_model,
    seed_everything,
    wrap_cuda_model,
)

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_engine",
        default="torch_ddp",
        choices=["torch_ddp", "deepspeed"],
        help="Engine for paralleled training",
    )
    parser.add_argument(
        "--date",
        default="",
        type=str,
        help="log date",
    )
    parser.add_argument(
        '--project',
        type=str,
        default='test'
    )
    parser.add_argument(
        '--enable_wandb',
        action='store_true'
    )
    parser.add_argument(
        '--wandb_runs_name',
        type=str,
        default='',
        help='Optional name for wandb run'
    )
    parser = add_model_args(parser)
    parser = add_dataset_args(parser)
    parser = add_ddp_args(parser)
    parser = add_deepspeed_args(parser)
    args = parser.parse_args()
    if args.train_engine == "deepspeed":
        args.deepspeed = True
        assert args.deepspeed_config is not None
    return args


@record
def main():
    args = get_args()

    # Set random seed
    seed_everything(args.seed)

    # Load config
    config = load_config(args.config)

    # Init custom log output
    log.init(args.log_dir, date=args.date, enable_wandb=args.enable_wandb, project=args.project, runs_name=args.wandb_runs_name)

    # Init env for ddp OR deepspeed
    init_distributed(args)

    # Update config based on arguments
    config = check_update_and_save_config(args, config)

    # Get dataloaders
    train_data_loader, val_data_loader = init_dataset_and_dataloader(
        args, config, args.seed
    )

    # Init model from config
    # 'init_infos' will be added to config
    models, config = init_models(args, config)

    # Statistic model scale
    params_statistic(models)

    wrap_cuda_model(args, models)

    # Disable gradient for specific models
    freeze_model_parameters(models, config)

    # Print model archtectures
    print_model(models)

    # Dispatch model from cpu to gpu

    # Get optimizer & scheduler
    models, optimizers, schedulers = init_optimizer_and_scheduler(args, config, models)

    if config.get('ema_update', False) and int(os.environ.get("RANK", 0)) == 0:
        from ema_pytorch import EMA
        generator_engine = models['generator']
        online = generator_engine.module
        ema_model = EMA(online, include_online_model=False)
        ema_model.to(generator_engine.device)
        if args.resume_step != 0:
            ema_model = resume_ema_checkpoint(ema_model, config["model_dir"], args.resume_step, key_name='ema_generator')
    else:
        ema_model = None

    # Get Trainer
    trainer = hydra.utils.instantiate(config["trainer"], config)

    # Start training loop
    start_epoch = config["current_epoch"]
    end_epoch = config["max_epoch"]
    assert start_epoch <= end_epoch

    for epoch in range(start_epoch, end_epoch):

        # Ensure all ranks start Train at the same time.
        dist.barrier()
        group_join = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=args.timeout)
        )
        
        trainer.train(
            models,
            optimizers,
            schedulers,
            train_data_loader,
            val_data_loader,
            group_join,
            epoch,
            datetime.timedelta(seconds=args.timeout),
            ema_model=ema_model,
        )

        # Ensure all ranks start val at the same time.
        dist.barrier()
        trainer.validate(models, val_data_loader)
        
        if trainer.step > trainer.total_step:
            break

if __name__ == "__main__":
    main()