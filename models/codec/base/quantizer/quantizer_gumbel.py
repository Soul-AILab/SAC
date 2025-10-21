# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/lucidrains/vector-quantize-pytorch under MIT License.


import typing as tp
import warnings
from functools import partial

from torch import nn, einsum
from einops import rearrange, repeat, reduce, pack, unpack
import torch
from torch import nn
import torch.nn.functional as F

from models.codec.base.quantizer import distrib


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def identity(t):
    return t


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def cdist(x, y):
    x2 = reduce(x**2, "b n d -> b n", "sum")
    y2 = reduce(y**2, "b n d -> b n", "sum")
    xy = einsum("b i d, b j d -> b i j", x, y) * -2
    return (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy).sqrt()


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum("i d, j d -> i j", normed_codes, normed_codes)
    return (cosine_sim**2).sum() / (h * n**2) - (1 / n)


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    logits,
    temperature=1.0,
    stochastic=False,
    straight_through=False,
    reinmax=False,
    dim=-1,
    training=True,
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (
        reinmax and not straight_through
    ), "reinmax can only be turned on if using straight through gumbel softmax"

    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        π0 = logits.softmax(dim=dim)
        π1 = (one_hot + (logits / temperature).softmax(dim=dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim=1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim=dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (float): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: float = 2,
        gumbel_sample=gumbel_sample,
    ):
        super().__init__()
        self.decay = decay
        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = (
            uniform_init if not kmeans_init else torch.zeros
        )
        embed = init_fn(codebook_size, dim)

        if not kmeans_init:
            embed = l2norm(embed)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return torch.tensor(0)

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return torch.tensor(len(self.cluster_size))
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        distrib.broadcast_tensors(self.buffers())

        return len(expired_codes) - sum(expired_codes)

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()

        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, training=self.training
        )

        return embed_ind, embed_onehot

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):

        dtype = x.dtype
        x, ps = pack_one(x, " * d")

        self.init_embed_(x)

        embed_ind, embed_onehot = self.quantize(x)
        # embed_ind = self.postprocess_emb(embed_ind, shape)
        embed_ind = unpack_one(embed_ind, ps, "*")

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, " * c")
            quantize = einsum("b n c,  c d -> b n d", unpacked_onehot, self.embed)
        else:
            quantize = self.dequantize(embed_ind)

        cluster_size = torch.zeros(1)
        active_num = (embed_onehot.sum(0) > 0).sum()
        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            active_num = self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        avg_probs = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantize, embed_ind, perplexity, active_num.float()


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (float): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: float = 2,
        commitment_weight: float = 1.0,
        orthogonal_reg_weight: float = 0.0,
        stochastic_sample_codes: bool = False,
        sample_codebook_temp: float = 1.0,
        straight_through: bool = False,
        reinmax=False,  # using reinmax for improved straight-through, assuming straight through helps at all
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        gumbel_sample_fn = partial(
            gumbel_sample,
            temperature=sample_codebook_temp,
            stochastic=stochastic_sample_codes,
            reinmax=reinmax,
            straight_through=straight_through,
        )

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight
        self.orthogonal_reg_weight = orthogonal_reg_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
            gumbel_sample=gumbel_sample_fn,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch_size, dim, seq_length)
        """

        # breakpoint()
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        quantize, embed_ind, perplexity, cluster_size = self._codebook(x)
        if self.training:
            quantize = x + (quantize - x).detach()
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        commit_loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        orthogonal_reg_loss = torch.tensor(
            [0.0], device=device, requires_grad=self.training
        )

        if self.training:
            # warnings.warn('When using RVQ in training model, first check '
            #               'https://github.com/facebookresearch/encodec/issues/25 . '
            #               'The bug wasn\'t fixed here for reproducibility.')
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook
                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return (
            quantize,
            embed_ind,
            (loss, commit_loss, orthogonal_reg_loss),
            perplexity,
            cluster_size,
        )


if __name__ == "__main__":
    model = VectorQuantization(
        30,
        codebook_size=1024,
        codebook_dim=256,
        orthogonal_reg_weight=1.0,
        straight_through=True,
        reinmax=True,
        sample_codebook_temp=1.0,
    )
    x = torch.randn(2, 30, 10)
    (
        quantize,
        embed_ind,
        losses,
        perplexity,
        cluster_size,
    ) = model(x)
    loss, commit_loss, orthogonal_reg_loss = losses
    print("quantize shape", quantize.shape)
    print("embed_ind", embed_ind)
    print("commit loss", commit_loss)
    print("orthogonal loss", orthogonal_reg_loss)