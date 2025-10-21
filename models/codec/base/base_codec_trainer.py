"""
Description:
    Base trainer class for wave codec models.
"""


from typing import Dict, Literal, TypedDict, List
from omegaconf import DictConfig

import os
import gc
import soundfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributed.distributed_c10d import GroupMember
from contextlib import nullcontext

import utils.log as log
from models.base.base_trainer import BaseTranier
from utils.train_utils import uneven_check, toggle_grad


class Batch(TypedDict):
    wav: torch.Tensor  # size [B, 1, T]    target
    wav_in: torch.Tensor  # size [B, 1, L]  input
    index: str


class BaseCodecTrainer(BaseTranier):
    """Base Trainer class for wave codec training.

    All trainer classes for wave codec should inherit from here.
    """

    def __init__(self, cfg: DictConfig = None):
        super().__init__(cfg)

    def batch_forward(
        self,
        models: Dict[str, nn.Module],
        batch: Batch,
        optimizers: Dict[str, optim.Optimizer] = None,
        schedulers: Dict[str, optim.lr_scheduler._LRScheduler] = None,
        mode: Literal["train", "val"] = "train",
    ):
        batch = self.batch_to_cuda(batch)
        batch = self.update_batch(batch)
        adv_loss_on = self.step > self.config["generator_warmup_steps"]
        loss_dict = dict()
        if self.train_engine == "deepspeed":
            # deepspeed
            # perform amp autocast when dtype != fp32
            with torch.cuda.amp.autocast(
                enabled=self.dtype is not None, dtype=self.dtype, cache_enabled=False
            ):

                outputs = models["generator"](batch["wav_in"])
                L = outputs["recons"].shape[-1]
                outputs["audios"] = batch["wav"].unsqueeze(1)[:, :, :L]

                # discriminator step
                if adv_loss_on:
                    # NOTE(xinsheng): In the GAN scenario, we have to freeze the state for the parameters being \
                    # handled by other optimizers during an update. Otherwise, it raises a set of issues like:
                    # "runtime error: no attribute 'ipg_index' and 'ipg_buffer' is None ". Please refer to:
                    # https://github.com/microsoft/DeepSpeed/issues/430#issuecomment-698227960
                    toggle_grad(models["discriminator"], True)
                    dis_loss = models["discriminator"].discriminative_loss(outputs)
                    if mode == "train":
                        self.optimizer_zero_grad(optimizers["discriminator"])
                        self.batch_backward_discriminator(
                            models["discriminator"], dis_loss["loss"]
                        )
                        self.update_parameter_and_lr(
                            models["discriminator"],
                            optimizers["discriminator"],
                            schedulers["discriminator"],
                        )
                    loss_dict["d_loss"] = dis_loss["d_loss"]
                    toggle_grad(models["discriminator"], False)

                # generator step
                gen_loss = models["generator"].generative_loss(outputs)
                if adv_loss_on:
                    loss_dict_adv = models["discriminator"].adversarial_loss(outputs)
                    gen_loss["loss"] += loss_dict_adv["loss"]
                    loss_dict_adv.pop("loss")
                    gen_loss.update(loss_dict_adv)

                loss_dict.update(gen_loss)

        else:
            # torch_ddp
            # autocast context
            # The more details about amp can be found in
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            with torch.cuda.amp.autocast(self.scaler is not None):
                outputs = models["generator"](batch["wav_in"])
                L = outputs["recons"].shape[-1]
                outputs["audios"] = batch["wav"].unsqueeze(1)[:, :, :L]

                # discriminator step
                if adv_loss_on:
                    dis_loss = models["discriminator"].discriminative_loss(outputs)
                    if mode == "train":
                        self.optimizer_zero_grad(optimizers["discriminator"])
                        self.batch_backward_discriminator(
                            models["discriminator"], dis_loss["loss"]
                        )
                        self.update_parameter_and_lr(
                            models["discriminator"],
                            optimizers["discriminator"],
                            schedulers["discriminator"],
                        )
                    loss_dict["d_loss"] = dis_loss["d_loss"]

                # generator step
                gen_loss = models["generator"].generative_loss(outputs)
                if adv_loss_on:
                    loss_dict_adv = models["discriminator"].adversarial_loss(outputs)
                    gen_loss["loss"] += loss_dict_adv["loss"]
                    loss_dict_adv.pop("loss")
                    gen_loss.update(loss_dict_adv)

                loss_dict.update(gen_loss)
        return loss_dict, outputs

    def batch_backward_discriminator(self, model: nn.Module, loss: torch.Tensor):

        if self.train_engine == "deepspeed":
            # `model.backward(loss)` is equivalent to
            # `scale_loss_wrt_accum_grad + loss.backward()`
            # ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api
            scaled_loss = model.backward(loss)

        elif self.train_engine == "torch_ddp":
            scaled_loss = loss
            if self.config["use_amp"]:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

    def train(
        self,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, optim.Optimizer],
        schedulers: Dict[str, optim.lr_scheduler._LRScheduler],
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        group_join: GroupMember,
        epoch: int,
        timeout: int = 30,
        ema_model: nn.Module = None,
    ):
        """Train one epoch"""
        [model.train() for _, model in models.items()]
        if ema_model is not None and self.rank == 0:
            ema_model.eval()

        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(models["generator"], torch.nn.parallel.DistributedDataParallel):
            model_g_context = models["generator"].join
        else:
            model_g_context = nullcontext

        with model_g_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                self.batch_idx = batch_idx
                if uneven_check(group_join, batch_idx, self.config, timeout):
                    break

                loss_dict, _ = self.batch_forward(
                    models, batch_dict, optimizers, schedulers
                )

                self.optimizer_zero_grad(optimizers["generator"])
                self.batch_backward(models["generator"], loss_dict["loss"])
                self.update_parameter_and_lr(
                    models["generator"],
                    optimizers["generator"],
                    schedulers["generator"],
                )

                if ema_model is not None and self.rank == 0:
                    ema_model.update()

                # log per step
                lr_dict = self.log_training_step(epoch, loss_dict, optimizers)

                if self.step % self.config["val_interval"] == 0 and self.step != 0:
                    validation_loss_dict = self.validate(models, val_data_loader)
                    [model.train() for _, model in models.items()]
                    self.log_validation_step(validation_loss_dict)

                # NOTE(xinsheng): In deepspeed, all rank should call save_checkpoint
                self.check_save_model(epoch, lr_dict, models, ema_model=ema_model)
                self.check_empty_cache()
                if self.step > self.total_step:
                    break
                self.step += 1

    @torch.no_grad()
    def validate(self, models: Dict[str, nn.Module], val_data_loader: DataLoader):
        """Evaluation"""
        [model.eval() for _, model in models.items()]
        total_utts = 0  # Total utterances processed
        max_val_utts = self.config["max_val_utts"]
        accumulated_loss = {}

        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(val_data_loader):

                utts_count = len(batch_dict["index"])
                if utts_count == 0:
                    continue

                loss_dict_batch, outputs = self.batch_forward(
                    models, batch_dict, mode="val"
                )
                self.log_syn_wav(outputs, batch_dict["index"])

                total_utts += utts_count
                for loss_name, loss_value in loss_dict_batch.items():
                    if loss_value is not None and "loss" in loss_name:
                        accumulated_loss[loss_name] = (
                            accumulated_loss.get(loss_name, 0) + loss_value * utts_count
                        )

                if total_utts >= max_val_utts:
                    break

        for loss_name in accumulated_loss:
            accumulated_loss[loss_name] /= total_utts

        # Perform garbage collection to release memory
        del batch_dict, outputs
        gc.collect()
        torch.cuda.empty_cache()
        return accumulated_loss

    def log_syn_wav(self, outputs: Dict[str, torch.Tensor], indexs: List[str]):
        """Save synthetic audio to local and log to tensorboard."""
        if self.step % self.config["syn_interval"] != 0 or self.rank != 0:
            return

        step = self.step
        sample_rate = self.config["sample_rate"]
        save_dir = os.path.join(self.config["log_dir"], f"val_results/{step}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        raw_wavs = outputs["audios"].squeeze(1).cpu().numpy()
        rec_wavs = outputs["recons"].squeeze(1).detach().cpu().numpy()
        for raw_wav, rec_wav, index in zip(raw_wavs, rec_wavs, indexs):
            log.write_audio(f"{index}/rec", rec_wav, sample_rate, step)
            log.write_audio(f"{index}/raw", raw_wav, sample_rate, step)

            soundfile.write(f"{save_dir}/{index}_rec.wav", rec_wav, sample_rate)
            soundfile.write(f"{save_dir}/{index}_raw.wav", raw_wav, sample_rate)