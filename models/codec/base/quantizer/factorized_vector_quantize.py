
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils.parametrizations import weight_norm
from models.codec.base.quantizer.distrib import broadcast_tensors

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def kmeans(samples, num_clusters: int, num_iters: int=10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means,
                                                                "c d -> () c d")
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

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)

def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num, ), device=device)

    return samples[indices]

class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=0.15,
        codebook_loss_weight=1.0,
        use_l2_normlize=True,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        forced_activation: bool = False,
        ema_update: bool = False,
        kmeans_init: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.use_l2_normlize = use_l2_normlize
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.forced_activation = forced_activation
        self.ema_update = ema_update

        if self.input_dim != self.codebook_dim:
            self.in_project = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(
                self.codebook_dim, self.input_dim, kernel_size=1
            )

        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))

    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size,
                                     self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size), self.codebook.weight)
        self.codebook.weight.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

    def forward(self, z):
        """
        Parameters
        ----------
        z: torch.Tensor[B x D x T]

        Returns
        -------
        z_q: torch.Tensor[B x D x T]
            Quantized continuous representation of input
        commit_loss: Tensor[B]
            Commitment loss to train encoder to predict vectors closer to codebook entries
        codebook_loss: Tensor[B]
            Codebook loss to update the codebook
        indices: torch.Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        z_e: torch.Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes project input into low-dimensional space if self.input_dim != self.codebook_dim
        z_e = self.in_project(z)
        z_q, indices, dists = self.decode_latents(z_e)

        # statistic the usage of codes
        embed_onehot = F.one_hot(indices, self.codebook_size).type(z_e.dtype)
        avg_probs = torch.mean(embed_onehot.reshape(-1, self.codebook_size), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        active_num = (embed_onehot.sum(0).sum(0) > 0).sum()
        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            if not self.forced_activation:
                ema_inplace(self.cluster_size, embed_onehot.sum(0).sum(0), self.decay)
            else:
                z_e = rearrange(z_e, "b d t -> b t d")
                self.expire_codes_(z_e)
                ema_inplace(self.cluster_size, embed_onehot.sum(0).sum(0), self.decay)

                if self.ema_update: # why not sync here 🤔
                    embed_sum = z_e.t() @ embed_onehot
                    ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
                    cluster_size = (
                        laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                        * self.cluster_size.sum()
                    )
                    embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                    self.codebook.weight.data.copy_(embed_normalized)
                
                z_e = rearrange(z_e, "b t d -> b d t")

            active_num = sum(self.cluster_size > self.threshold_ema_dead_code)

        # Compute commitment loss and codebook loss
        if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
                * self.commitment
            )
            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
                * self.codebook_loss_weight
            )
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)
            codebook_loss = torch.zeros(z.shape[0], device=z.device)

        z_q = z_e + (z_q - z_e).detach()

        z_q = self.out_project(z_q)

        return z_q, indices, commit_loss, codebook_loss, dists,  perplexity, active_num.float()

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # if use_l2_normlize is True, the distance is equal to cosine distance
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)

        return z_q, indices, dist

    def vq2emb(self, vq, out_proj=True):
        emb = self.decode_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb

    def latent2dist(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # if use_l2_normlize is True, the distance is equal to cosine distance
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )  # (b*t, k)

        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        dist = rearrange(dist, "(b t) k -> b t k", b=latents.size(0))
        z_q = self.decode_code(indices)

        return -dist, indices, z_q

if __name__ == "__main__":
    quantizer = FactorizedVectorQuantize(
      input_dim= 1024,
      codebook_size= 8192,
      codebook_dim= 8,
      commitment= 0.25,
      codebook_loss_weight= 4.0,
      use_l2_normlize= True,
      threshold_ema_dead_code= 0.2
    )

    z = torch.randn(1, 1024, 10)
    z_q, indices, commit_loss, codebook_loss, dists, perplexity, active_num = quantizer(z)
    print ("zq", z_q.shape)
    print ("indices", indices.shape)
    print ("test", quantizer.vq2emb(indices).shape)
