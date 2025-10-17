# Modified from ESPnet(https://github.com/espnet/espnet)
#               NeMo(https://github.com/NVIDIA/NeMo)

from typing import Union

import math
import warnings
import torch
from torch.optim.lr_scheduler import _LRScheduler, LRScheduler
from torch.optim.lr_scheduler import LambdaLR


class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        last_epoch: int = -1,
        min_lr: float = 1e-6,
    ):
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]

    def set_step(self, step: int):
        self.last_epoch = step


class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
        min_lr: float = 1e-6,
    ):
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            return [lr * step_num**-0.5 for lr in self.base_lrs]
        elif step_num < self.warmup_steps:
            return [
                (
                    lr
                    * self.warmup_steps**0.5
                    * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
                )
                for lr in self.base_lrs
            ]
        else:
            return [
                (
                    max(
                        lr
                        * self.warmup_steps**0.5
                        * min(step_num**-0.5, step_num * self.warmup_steps**-1.5),
                        self.min_lr,
                    )
                )
                for lr in self.base_lrs
            ]

    def set_step(self, step: int):
        self.last_epoch = step


class NoamLR(_LRScheduler):
    """
    Noam Learning Rate Scheduler based on the formula from the "Attention is All You Need" paper.
    It includes a learning rate warmup period and then decreases the learning rate proportionally
    to the inverse square root of the step number, with an optional minimum learning rate.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        warmup_steps (Union[int, float]): Number of warmup steps, during which the learning rate increases linearly.
                                          After that, it decreases according to the inverse square root of the step number.
                                          Default is 25000.
        last_epoch (int): The index of the last epoch when resuming training. Default is -1.
        min_lr (float): Minimum learning rate. Default is 1e-6.
        model_size (int): Dimensionality of the model, used to scale the learning rate. Default is 256.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
        min_lr: float = 1e-6,
        model_size: int = 256,
    ):
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.model_size = model_size

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            return [lr * step_num**-0.5 for lr in self.base_lrs]
        elif step_num < self.warmup_steps:
            return [
                (
                    lr
                    * self.model_size**-0.5
                    * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
                )
                for lr in self.base_lrs
            ]
        else:
            return [
                (
                    max(
                        lr
                        * self.model_size**-0.5
                        * min(step_num**-0.5, step_num * self.warmup_steps**-1.5),
                        self.min_lr,
                    )
                )
                for lr in self.base_lrs
            ]

    def set_step(self, step: int):
        self.last_epoch = step


class WarmupLRX(_LRScheduler):
    """
    warmup lr for xcodec
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        down_steps: int = 50000,
        last_epoch: int = -1,
        min_lr: float = 1e-6,
        max_lr: float = 1e-4,
    ):

        self.alpha = (max_lr - 1e-5) / warmup_steps**2
        self.warmup_steps = warmup_steps
        self.down_steps = down_steps
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        init_lr = 1e-5
        s1, s2 = self.warmup_steps, self.warmup_steps + self.down_steps
        if step_num < s1:
            return [init_lr + self.alpha * step_num**2 for _ in self.base_lrs]
        elif s1 <= step_num < s2:
            return [
                (self.max_lr - self.min_lr) / (s1 - s2) * step_num
                + (self.min_lr * s1 - self.max_lr * s2) / (s1 - s2)
                for _ in self.base_lrs
            ]
        else:
            return [self.min_lr for _ in self.base_lrs]

    def set_step(self, step: int):
        self.last_epoch = step


class WarmupPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert (
            warmup_ratio is None or max_steps is not None
        ), "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed "
                "by the scheduler, please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        step = self.last_epoch

        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class SquareRootConstantPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(
        self,
        optimizer,
        *,
        constant_steps=None,
        constant_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (
            constant_steps is not None and constant_ratio is not None
        ), "Either use particular number of step or ratio"
        assert (
            constant_ratio is None or max_steps is not None
        ), "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if constant_steps is not None:
            self.constant_steps = constant_steps
        elif constant_ratio is not None:
            self.constant_steps = int(constant_ratio * max_steps)
        else:
            self.constant_steps = 0

        self.constant_lr = 1 / (constant_steps**0.5)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed "
                "by the scheduler, please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        step = self.last_epoch

        if step <= self.constant_steps:
            return [self.constant_lr for _ in self.base_lrs]

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class WarmupHoldPolicy(WarmupPolicy):
    """Variant of WarmupPolicy which maintains high
       learning rate for a defined number of steps.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        hold_steps: Number of training steps to
                    hold the learning rate after warm up
        hold_ratio: Ratio of hold steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        hold_steps=None,
        hold_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (
            hold_steps is not None and hold_ratio is not None
        ), "Either use particular number of step or ratio"
        assert (
            hold_ratio is None or max_steps is not None
        ), "If there is a ratio, there should be a total steps"

        self.min_lr = min_lr
        self._last_warmup_lr = 0.0

        # Necessary to duplicate as class attributes are hidden in inner class
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if hold_steps is not None:
            self.hold_steps = hold_steps + self.warmup_steps
        elif hold_ratio is not None:
            self.hold_steps = int(hold_ratio * max_steps) + self.warmup_steps
        else:
            self.hold_steps = 0

        super().__init__(
            optimizer,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            max_steps=max_steps,
            last_epoch=last_epoch,
            min_lr=min_lr,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler,"
                " "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        step = self.last_epoch

        # Warmup phase
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        # Hold phase
        if (step >= self.warmup_steps) and (step < self.hold_steps):
            return self.base_lrs

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)


class WarmupAnnealHoldPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
        min_lr: Minimum lr to hold the learning rate after decay at.
        constant_steps: Number of steps to keep lr constant at.
        constant_ratio: Ratio of steps to keep lr constant.
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        constant_steps=None,
        constant_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert not (
            constant_steps is not None and constant_ratio is not None
        ), "Either use constant_steps or constant_ratio"
        assert (
            warmup_ratio is None or max_steps is not None
        ), "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps

        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if constant_steps is not None:
            self.constant_steps = constant_steps
        elif constant_ratio is not None:
            self.constant_steps = int(constant_ratio * max_steps)
        else:
            self.constant_steps = 0

        self.decay_steps = max_steps - (self.constant_steps + self.warmup_steps)

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed "
                "by the scheduler, please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        step = self.last_epoch

        # Warmup steps
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self._get_warmup_lr(step)

        # Constant steps after warmup and decay
        if (
            self.constant_steps > 0
            and (self.warmup_steps + self.decay_steps) < step <= self.max_steps
        ):
            return self._get_constant_lr(step)

        # Min lr after max steps of updates
        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_constant_lr(self, step):
        return [self.min_lr for _ in self.base_lrs]

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class WarmupAnnealSteps(LambdaLR):
    """
    Linear warmup and then inverse square root decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        anneal_steps,
        anneal_rate,
        final_lr=None,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        self.anneal_rate = anneal_rate
        self.decay_factor = warmup_steps**0.5
        self.final_lr = final_lr
        super(WarmupAnnealSteps, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        step += 1
        lr = (
            min(float(step) ** -0.5, self.warmup_steps**-1.5 * step)
            * self.warmup_steps**0.5
        )

        for s in self.anneal_steps:
            if step > s:
                lr = lr * self.anneal_rate
        if self.final_lr is not None:
            lr = max(self.final_lr, lr)
        # actually, the lr ratio
        return lr

    def set_step(self, step: int):
        self.last_epoch = step


def _squareroot_annealing(initial_lr, step, max_steps, min_lr):
    mult = ((max_steps - step) / max_steps) ** 0.5
    out_lr = initial_lr * mult
    out_lr = max(out_lr, min_lr)
    return out_lr


def _square_annealing(initial_lr, step, max_steps, min_lr):
    mult = ((max_steps - step) / max_steps) ** 2
    out_lr = initial_lr * mult
    out_lr = max(out_lr, min_lr)
    return out_lr


def _cosine_annealing(initial_lr, step, max_steps, min_lr):
    mult = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    out_lr = (initial_lr - min_lr) * mult + min_lr
    return out_lr


def _linear_warmup_with_cosine_annealing(
    max_lr, warmup_steps, step, decay_steps, min_lr
):
    assert max_lr > min_lr
    # Use linear warmup for the initial part.
    if warmup_steps > 0 and step <= warmup_steps:
        return max_lr * float(step) / float(warmup_steps)

    # For any steps larger than `decay_steps`, use `min_lr`.
    if step > warmup_steps + decay_steps:
        return min_lr

    # If we are done with the warmup period, use the decay style.
    num_steps_ = step - warmup_steps
    decay_steps_ = decay_steps
    decay_ratio = float(num_steps_) / float(decay_steps_)
    assert decay_ratio >= 0.0
    assert decay_ratio <= 1.0
    delta_lr = max_lr - min_lr

    coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)

    return min_lr + coeff * delta_lr


def _poly_decay(initial_lr, step, decay_steps, power, min_lr, cycle):
    if cycle:
        multiplier = 1.0 if step == 0 else math.ceil(step / decay_steps)
        decay_steps *= multiplier
    else:
        step = min(step, decay_steps)
    p = step / decay_steps
    lr = (initial_lr - min_lr) * math.pow(1.0 - p, power)
    lr += min_lr
    return lr


def _noam_hold_annealing(
    initial_lr, step, warmup_steps, hold_steps, decay_rate, min_lr
):
    # hold_steps = total number of steps
    # to hold the LR, not the warmup + hold steps.
    T_warmup_decay = max(1, warmup_steps**decay_rate)
    T_hold_decay = max(1, (step - hold_steps) ** decay_rate)
    lr = (initial_lr * T_warmup_decay) / T_hold_decay
    lr = max(lr, min_lr)
    return lr


class SquareAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=1e-5, last_epoch=-1, **kwargs):
        super().__init__(
            optimizer=optimizer,
            max_steps=max_steps,
            last_epoch=last_epoch,
            min_lr=min_lr,
            **kwargs,
        )

    def _get_lr(self, step):
        new_lrs = [
            _square_annealing(
                initial_lr=initial_lr,
                step=step - self.warmup_steps,
                max_steps=self.max_steps - self.warmup_steps,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class SquareRootAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0, last_epoch=-1, **kwargs):
        super().__init__(
            optimizer=optimizer,
            max_steps=max_steps,
            last_epoch=last_epoch,
            min_lr=min_lr,
            **kwargs,
        )

    def _get_lr(self, step):
        new_lrs = [
            _squareroot_annealing(
                initial_lr=initial_lr,
                step=step,
                max_steps=self.max_steps,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class CosineAnnealing(WarmupAnnealHoldPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0, last_epoch=-1, **kwargs):
        super().__init__(
            optimizer=optimizer,
            max_steps=max_steps,
            last_epoch=last_epoch,
            min_lr=min_lr,
            **kwargs,
        )

    def _get_lr(self, step):
        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate "
                    f"that was lower than the minimum learning rate."
                )

        if self.constant_steps is None or self.constant_steps == 0:
            new_lrs = [
                _cosine_annealing(
                    initial_lr=initial_lr,
                    step=step - self.warmup_steps,
                    max_steps=self.max_steps - self.warmup_steps,
                    min_lr=self.min_lr,
                )
                for initial_lr in self.base_lrs
            ]
        else:
            new_lrs = self._get_linear_warmup_with_cosine_annealing_lr(step)
        return new_lrs

    def _get_warmup_lr(self, step):
        if self.constant_steps is None or self.constant_steps == 0:
            return super()._get_warmup_lr(step)
        else:
            # Use linear warmup for the initial part.
            return self._get_linear_warmup_with_cosine_annealing_lr(step)

    def _get_constant_lr(self, step):
        # Only called when `constant_steps` > 0.
        return self._get_linear_warmup_with_cosine_annealing_lr(step)

    def _get_linear_warmup_with_cosine_annealing_lr(self, step):
        # Cosine Schedule for Megatron LM,
        # slightly different warmup schedule + constant LR at the end.
        new_lrs = [
            _linear_warmup_with_cosine_annealing(
                max_lr=self.base_lrs[0],
                warmup_steps=self.warmup_steps,
                step=step,
                decay_steps=self.decay_steps,
                min_lr=self.min_lr,
            )
            for _ in self.base_lrs
        ]
        return new_lrs


class NoamAnnealing(_LRScheduler):
    def __init__(
        self,
        optimizer,
        *,
        d_model,
        warmup_steps=None,
        warmup_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        self._normalize = d_model ** (-0.5)
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert (
            warmup_ratio is None or max_steps is not None
        ), "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed "
                "by the scheduler, please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        step = max(1, self.last_epoch)

        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate "
                    f"that was lower than the minimum learning rate."
                )

        new_lrs = [
            self._noam_annealing(initial_lr=initial_lr, step=step)
            for initial_lr in self.base_lrs
        ]
        return new_lrs

    def _noam_annealing(self, initial_lr, step):
        if self.warmup_steps > 0:
            mult = self._normalize * min(
                step ** (-0.5), step * (self.warmup_steps ** (-1.5))
            )
        else:
            mult = self._normalize * step ** (-0.5)

        out_lr = initial_lr * mult
        if step > self.warmup_steps:
            out_lr = max(out_lr, self.min_lr)
        return out_lr


class NoamHoldAnnealing(WarmupHoldPolicy):
    def __init__(
        self,
        optimizer,
        *,
        max_steps,
        decay_rate=0.5,
        min_lr=0.0,
        last_epoch=-1,
        **kwargs,
    ):
        """
        From Nemo:
        Implementation of the Noam Hold Annealing policy
        from the SqueezeFormer paper.

        Unlike NoamAnnealing, the peak learning rate
        can be explicitly set for this scheduler.
        The schedule first performs linear warmup,
        then holds the peak LR, then decays with some schedule for
        the remainder of the steps.
        Therefore the min-lr is still dependent
        on the hyper parameters selected.

        It's schedule is determined by three factors-

        Warmup Steps: Initial stage, where linear warmup
            occurs uptil the peak LR is reached. Unlike NoamAnnealing,
            the peak LR is explicitly stated here instead of a scaling factor.

        Hold Steps: Intermediate stage, where the peak LR
            is maintained for some number of steps. In this region,
            the high peak LR allows the model to converge faster
            if training is stable. However the high LR
            may also cause instability during training.
            Should usually be a significant fraction of training
            steps (around 30-40% of the entire training steps).

        Decay Steps: Final stage, where the LR rapidly decays
            with some scaling rate (set by decay rate).
            To attain Noam decay, use 0.5,
            for Squeezeformer recommended decay, use 1.0.
            The fast decay after prolonged high LR during
            hold phase allows for rapid convergence.

        References:
            - [Squeezeformer:
            An Efficient Transformer for Automatic Speech Recognition]
            (https://arxiv.org/abs/2206.00888)

        Args:
            optimizer: Pytorch compatible Optimizer object.
            warmup_steps: Number of training steps in warmup stage
            warmup_ratio: Ratio of warmup steps to total steps
            hold_steps: Number of training steps to
                        hold the learning rate after warm up
            hold_ratio: Ratio of hold steps to total steps
            max_steps: Total number of steps while training or `None` for
                infinite training
            decay_rate: Float value describing the polynomial decay
                        after the hold period. Default value
                        of 0.5 corresponds to Noam decay.
            min_lr: Minimum learning rate.
        """
        self.decay_rate = decay_rate
        super().__init__(
            optimizer=optimizer,
            max_steps=max_steps,
            last_epoch=last_epoch,
            min_lr=min_lr,
            **kwargs,
        )

    def _get_lr(self, step):
        if self.warmup_steps is None or self.warmup_steps == 0:
            raise ValueError("Noam scheduler cannot be used without warmup steps")

        if self.hold_steps > 0:
            hold_steps = self.hold_steps - self.warmup_steps
        else:
            hold_steps = 0

        new_lrs = [
            _noam_hold_annealing(
                initial_lr,
                step=step,
                warmup_steps=self.warmup_steps,
                hold_steps=hold_steps,
                decay_rate=self.decay_rate,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs

    def set_step(self, step: int):
        self.last_epoch = step
