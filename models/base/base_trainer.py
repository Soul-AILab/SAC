import gc
import os
import time
from typing import Dict, Union
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import utils.log as log
from utils.train_utils import save_models


class BaseTranier:
    def __init__(self, cfg: DictConfig = None, **kwargs):
        """Base Trainer class from which all trainer classes should inherit."""

        self.step = cfg["current_step"]
        self.total_step = cfg["total_step"]
        self.config = cfg
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.train_engine = cfg["train_engine"]
        self.scaler = None
        self.dtype = None
        self.pre_save_time = time.time()
        if cfg["use_amp"]:
            self.scaler = torch.cuda.amp.GradScaler()

    def batch_to_cuda(self, batch: dict):
        for k in batch:
            if type(batch[k]) is torch.Tensor:
                batch[k] = batch[k].cuda(self.local_rank)
        return batch

    def update_batch(self, batch, *args, **kwargs):
        return batch

    def dtype_parser(self):
        dtype = self.config["dtype"]
        if dtype == "fp16":
            dtype = torch.float16
        elif dtype == "bf16":
            dtype = torch.bfloat16
        else:  # fp32
            dtype = None
        self.dtype = dtype

    def optimizer_zero_grad(self, optimizer: optim.Optimizer):
        if self.train_engine == "deepspeed":
            # Zeroing the gradients is handled automatically by DeepSpeed
            # after the weights have been updated using a mini-batch.
            #   `ds_model.step() = clip_grad_norm_() + optimizer.step()
            #                     + optimizer.zero_grad() + scheduler.step()`
            return
        else:
            optimizer.zero_grad()

    def batch_backward(self, model: nn.Module, loss: torch.Tensor):

        if self.train_engine == "deepspeed":
            # `model.backward(loss)` is equivalent to
            # `scale_loss_wrt_accum_grad + loss.backward()`
            # ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api
            model.backward(loss)

        elif self.train_engine == "torch_ddp":
            if self.config["use_amp"]:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

    def update_parameter_and_lr(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
    ):
        grad_norm = 0.0
        if self.train_engine == "deepspeed":
            # The step() function in DeepSpeed engine updates the
            # model parameters as well as the learning rate.
            # Zeroing the gradients is handled automatically by
            # DeepSpeed after the weights have been updated using a mini-batch.
            # DeepSpeed also performs gradient averaging automatically at the
            # gradient accumulation boundaries and addresses clip_grad_norm internally.
            # `ds_model.step() = clip_grad_norm_() + optimizer.step()
            #                   + optimizer.zero_grad() + scheduler.step()`
            # ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api
            model.step()
            grad_norm = model.get_global_grad_norm()
        else:
            # Use mixed precision training
            if self.config["use_amp"]:
                self.scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(
                    model.parameters(), self.config["grad_clip"]
                )
                # Must invoke scaler.update() if unscale_() is used in
                # the iteration to avoid the following error:
                #   RuntimeError: unscale_() has already been called
                #   on this optimizer since the last update().
                # We don't check grad here since that if the gradient
                # has inf/nan values, scaler.step will skip
                # optimizer.step().
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                grad_norm = clip_grad_norm_(
                    model.parameters(), self.config["grad_clip"]
                )
                if torch.isfinite(grad_norm):
                    optimizer.step()
            scheduler.step()
            grad_norm = grad_norm.item()

    def get_lr(
        self, optimizers: Union[Dict[str, optim.Optimizer], optim.Optimizer]
    ) -> Union[Dict[str, float], float]:
        """
        Get the learning rate from an optimizer or a dictionary of optimizers.

        Args:
            optimizers (Union[Dict[str, optim.Optimizer], optim.Optimizer]): An optimizer or a dictionary
                of optimizer names to their corresponding optim.Optimizer objects.

        Returns:
            Union[Dict[str, float], float]: A dictionary mapping each optimizer to its current learning rate,
                or a single float representing the learning rate if a single optimizer is provided.
        """
        if isinstance(optimizers, dict):
            return {
                k: optimizer.param_groups[0]["lr"]
                for k, optimizer in optimizers.items()
                if optimizer is not None
            }

        else:
            return optimizers.param_groups[0]["lr"]

    def check_empty_cache(self):
        """Periodically clears the GPU cache to free up unused memory."""
        if self.step % self.config["empty_cache_interval"] == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def log_training_step(
        self,
        epoch: int,
        loss_dict: Dict[str, float],
        optimizers: Union[Dict[str, optim.Optimizer], optim.Optimizer],
    ):
        """Log training progress and statistics.

        Only logs if the instance is on the main processing rank (rank 0) and at specified logging intervals.

        Args:
            epoch (int): Current epoch number.
            loss_dict (Dict[str, float]): loss in dictionary
            optimizers (Union[Dict[str, optim.Optimizer], optim.Optimizer]): An optimizer or a dictionary
                of optimizer names to their corresponding optim.Optimizer objects.

        Returns:
            Union[Dict[str, float], float]: Updated dictionary including learning rates, if logging occurred.
        """
        if self.rank != 0:
            return

        config = self.config

        if self.step % config["log_interval"] == 0:
            lrs = self.get_lr(optimizers)
            if not isinstance(lrs, dict):
                lr_dict = {"model": lrs}
            else:
                lr_dict = lrs

            # Construct log message with losses and learning rates
            loss_msgs = ", ".join(
                f"{k}: {v:.6f}" for k, v in loss_dict.items() if isinstance(v, float)
            )
            lr_msgs = ", ".join(f"lr_{k}: {v:.2e}" for k, v in lr_dict.items())
            msg = f"Train | Step {self.step}/{self.total_step}, Epoch {epoch}, {loss_msgs}, {lr_msgs}"

            # Calculate elapsed and total estimated time if timestamps are available
            if hasattr(self, "start_time") and self.start_step is not None:
                elapsed_time = (time.time() - self.start_time) / 3600
                total_time = (
                    elapsed_time
                    / (self.step - self.start_step)
                    * (self.total_step - self.start_step)
                )
                msg += f", Time: {elapsed_time:.1f}/{total_time:.1f} hour."

            # Start the timer at the first logging step
            else:
                self.start_step = self.step
                self.start_time = time.time()

            # Update loss dictionary with learning rates and log the message
            loss_dict.update({f"lr_{k}": v for k, v in lr_dict.items()})
            log.info(msg)
            log.write_loss("train", loss_dict, self.step)

            return lrs

    def log_validation_step(self, loss_dict: Dict[str, float]):
        """
        Logs the evaluation metrics if the current instance is on the main processing rank (rank 0).

        Args:
            loss_dict (Dict[str, float]): Dictionary containing loss metrics where keys are metric names
                                          and values are their corresponding loss values.
        """
        # Only proceed with logging if this is the main process (rank 0)
        if self.rank != 0:
            return

        # Ensure there are metrics to log
        if not loss_dict:
            log.warning(
                "Attempted to log evaluation data, but no metrics were provided."
            )
            return

        # Construct the log message
        metric_messages = [
            f"{k}: {v:.6f}"
            for k, v in loss_dict.items()
            if isinstance(v, float) or isinstance(v, int)
        ]
        msg = "Evaluation, " + ", ".join(metric_messages)
        log.info(msg)
        log.write_loss("eval", loss_dict, self.step)

    @torch.no_grad()
    def check_save_model(
        self,
        epoch: int,
        lrs: Union[Dict[str, float], float],
        models: Union[Dict[str, nn.Module], nn.Module],
        ema_model = None,
    ):
        """
        Save the model under the following two conditions:
            - at every save interval which should be defined in config['save_interval']
            - every interval duration defined in config["tmp_save_time"]
        """
        if (
            self.step % self.config["save_interval"] == 0
            and self.step != 0
            or time.time() - self.pre_save_time > self.config["tmp_save_time"]
            and self.step % 1000 == 0
        ):
            # Update info_dict
            self.config.update(
                {"current_step": self.step, "current_epoch": epoch, "current_lr": lrs}
            )

            if ema_model is not None and self.rank == 0:
                models["ema_generator"] = ema_model

            save_models(models, self.config)
            self.pre_save_time = time.time()

            if "ema_generator" in models:
                models.pop("ema_generator")

    @torch.no_grad()
    def log_syn_wav(self, *args, **kwargs):
        raise NotImplementedError("This method needs to be implemented by subclasses.")
