# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_latents(z):
    if z.ndim == 3:
        return z.mean(dim=1)
    if z.ndim == 2:
        return z
    raise ValueError(f"Expected latents with shape [B, D] or [B, N, D], got {tuple(z.shape)}")


def predictive_variance(predictions):
    if predictions.ndim < 3:
        raise ValueError("Expected predictions with shape [B, K, ...]")
    centered = predictions - predictions.mean(dim=1, keepdim=True)
    return centered.flatten(start_dim=2).pow(2).sum(dim=-1).mean(dim=1)


def discounted_survival_targets(collapse_step, episode_length, gamma=0.97, device=None):
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    collapse_step = torch.as_tensor(collapse_step, device=device)
    steps = torch.arange(episode_length, device=collapse_step.device).view(1, -1)
    tau = collapse_step.view(-1, 1)
    stable = tau < 0
    targets = torch.ones(collapse_step.numel(), episode_length, device=collapse_step.device)
    if stable.all():
        return targets
    valid = (steps <= tau) & (~stable)
    numerator = 1.0 - gamma ** (tau - steps)
    denominator = torch.clamp(1.0 - gamma**tau, min=1.0e-6)
    collapsed_targets = numerator / denominator
    targets = torch.where(valid, collapsed_targets, targets)
    targets = torch.where((steps > tau) & (~stable), torch.zeros_like(targets), targets)
    return targets.clamp(0.0, 1.0)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        layers = []
        dim = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU(), norm_layer(hidden_dim)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LatentDynamicsMLP(nn.Module):
    def __init__(
        self,
        latent_dim,
        action_dim,
        hidden_dim=1024,
        depth=3,
        residual=True,
        action_hidden_dim=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.residual = residual
        action_hidden_dim = action_hidden_dim or latent_dim
        self.action_encoder = MLP(action_dim, action_hidden_dim, hidden_dim, 2)
        self.predictor = MLP(latent_dim + action_hidden_dim, latent_dim, hidden_dim, depth)

    def forward(self, z, action):
        squeeze_tokens = False
        if z.ndim == 2:
            z = z.unsqueeze(1)
            squeeze_tokens = True
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape [B, D] or [B, N, D], got {tuple(z.shape)}")
        if action.ndim == 3:
            action = action[:, -1]
        if action.ndim != 2:
            raise ValueError(f"Expected action with shape [B, A] or [B, T, A], got {tuple(action.shape)}")
        a = self.action_encoder(action).unsqueeze(1).expand(-1, z.size(1), -1)
        delta = self.predictor(torch.cat([z, a], dim=-1))
        out = z + delta if self.residual else delta
        return out.squeeze(1) if squeeze_tokens else out


class PerturbationGenerator(nn.Module):
    def __init__(
        self,
        noise_dim,
        latent_dim,
        action_dim,
        hidden_dim=512,
        depth=3,
        latent_scale=0.02,
        action_scale=0.02,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.latent_scale = latent_scale
        self.action_scale = action_scale
        self.net = MLP(noise_dim, latent_dim + action_dim, hidden_dim, depth)

    def forward(self, noise, token_count=None):
        out = torch.tanh(self.net(noise))
        delta_z, delta_action = out.split([self.latent_dim, self.action_dim], dim=-1)
        delta_z = delta_z * self.latent_scale
        delta_action = delta_action * self.action_scale
        if token_count is not None:
            delta_z = delta_z.unsqueeze(1).expand(-1, token_count, -1)
        return delta_z, delta_action


class StabilityHead(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512, depth=3):
        super().__init__()
        self.regressor = MLP(latent_dim, 1, hidden_dim, depth)

    def forward(self, z):
        return torch.sigmoid(self.regressor(pool_latents(z))).squeeze(-1)


class LatentVarianceAssessment(nn.Module):
    def __init__(
        self,
        latent_dim,
        action_dim,
        noise_dim=128,
        dynamics_hidden_dim=1024,
        dynamics_depth=3,
        perturb_hidden_dim=512,
        perturb_depth=3,
        latent_perturb_scale=0.02,
        action_perturb_scale=0.02,
        stability_hidden_dim=512,
        stability_depth=3,
        beta=1.0,
    ):
        super().__init__()
        self.beta = beta
        self.dynamics = LatentDynamicsMLP(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=dynamics_hidden_dim,
            depth=dynamics_depth,
        )
        self.perturbation = PerturbationGenerator(
            noise_dim=noise_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=perturb_hidden_dim,
            depth=perturb_depth,
            latent_scale=latent_perturb_scale,
            action_scale=action_perturb_scale,
        )
        self.stability_head = StabilityHead(
            latent_dim=latent_dim,
            hidden_dim=stability_hidden_dim,
            depth=stability_depth,
        )
        self.noise_dim = noise_dim

    def dynamics_loss(self, z, action, z_next):
        pred = self.dynamics(z, action)
        return F.mse_loss(pred, z_next), pred

    def sample_perturbed_predictions(self, z, action, perturbation_count):
        if action.ndim == 3:
            action = action[:, -1]
        token_count = z.size(1) if z.ndim == 3 else None
        batch_size = z.size(0)
        noise = torch.randn(batch_size * perturbation_count, self.noise_dim, device=z.device, dtype=z.dtype)
        delta_z, delta_action = self.perturbation(noise, token_count=token_count)
        z_rep = z.repeat_interleave(perturbation_count, dim=0)
        action_rep = action.repeat_interleave(perturbation_count, dim=0)
        preds = self.dynamics(z_rep + delta_z, action_rep + delta_action)
        return preds.view(batch_size, perturbation_count, *preds.shape[1:])

    def variance_score(self, z, action, perturbation_count=16):
        preds = self.sample_perturbed_predictions(z, action, perturbation_count)
        return predictive_variance(preds), preds

    def calibration_loss(self, z, action, stability, perturbation_count=16):
        variance, preds = self.variance_score(z, action, perturbation_count)
        target = self.beta * (1.0 - stability.float())
        return F.mse_loss(variance, target), variance, preds

    def stability_loss(self, z, stability):
        pred = self.stability_head(z)
        return F.mse_loss(pred, stability.float()), pred
