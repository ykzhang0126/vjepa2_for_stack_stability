# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import torch

from src.models.lva import discounted_survival_targets


def _load_payload(path):
    path = Path(path)
    if path.suffix in (".pt", ".pth"):
        return torch.load(path, map_location="cpu")
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: torch.as_tensor(data[k]) for k in data.files}
    raise ValueError(f"Unsupported LVA dataset file: {path}")


def _first(payload, names, default=None):
    for name in names:
        if name in payload:
            return payload[name]
    return default


class LVATrajectoryDataset(torch.utils.data.Dataset):
    """
    Loads trajectory tensors for LVA Stage 2/3 training.

    Supported dict/npz keys:
      latents or z: [E, T, D] or [E, T, N, D]
      actions or a: [E, T-1, A]
      stability or y: [E, T] or [E, T-1]
      collapse_step: [E], with -1 for episodes that never collapse
    """

    def __init__(self, path, gamma=0.97):
        payload = _load_payload(path)
        if isinstance(payload, (list, tuple)):
            self.samples = payload
            self.latent_dim = self.samples[0]["z"].shape[-1]
            self.action_dim = self.samples[0]["action"].shape[-1]
            return

        latents = _first(payload, ("latents", "z", "features"))
        actions = _first(payload, ("actions", "a"))
        stability = _first(payload, ("stability", "y", "labels"))
        collapse_step = _first(payload, ("collapse_step", "collapse_steps"))
        if latents is None or actions is None:
            raise ValueError("LVA dataset requires `latents`/`z` and `actions`/`a` tensors")

        latents = latents.float()
        actions = actions.float()
        if latents.ndim not in (3, 4):
            raise ValueError("latents must have shape [episodes, steps, dim] or [episodes, steps, tokens, dim]")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [episodes, steps-1, action_dim]")
        if latents.size(0) != actions.size(0) or latents.size(1) < actions.size(1) + 1:
            raise ValueError("latents/actions must describe matching trajectories")

        if stability is None:
            if collapse_step is None:
                stability = torch.ones(latents.size(0), latents.size(1))
            else:
                stability = discounted_survival_targets(collapse_step.long(), latents.size(1), gamma=gamma)
        stability = stability.float()
        if stability.ndim == 1:
            stability = stability[:, None].expand(-1, latents.size(1))

        self.samples = []
        for episode in range(actions.size(0)):
            for step in range(actions.size(1)):
                self.samples.append(
                    {
                        "z": latents[episode, step],
                        "z_next": latents[episode, step + 1],
                        "action": actions[episode, step],
                        "stability": stability[episode, min(step, stability.size(1) - 1)],
                    }
                )
        self.latent_dim = latents.shape[-1]
        self.action_dim = actions.shape[-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (
            sample["z"].float(),
            sample["action"].float(),
            sample["z_next"].float(),
            torch.as_tensor(sample["stability"]).float(),
        )


def init_data(path, batch_size, num_workers=0, pin_mem=False, shuffle=True, gamma=0.97, drop_last=False):
    dataset = LVATrajectoryDataset(path=path, gamma=gamma)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )
    return dataset, loader
