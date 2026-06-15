# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the root directory of this source tree.

import argparse

import torch
import yaml

from app.lva.train import predict
from src.models.lva import LatentVarianceAssessment


def load_model(checkpoint_path, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    args = checkpoint.get("args", {})
    cfg_model = args.get("model", {})
    cfg_loss = args.get("loss", {})
    model = LatentVarianceAssessment(
        latent_dim=cfg_model["latent_dim"],
        action_dim=cfg_model["action_dim"],
        noise_dim=cfg_model.get("noise_dim", 128),
        dynamics_hidden_dim=cfg_model.get("dynamics_hidden_dim", 1024),
        dynamics_depth=cfg_model.get("dynamics_depth", 3),
        perturb_hidden_dim=cfg_model.get("perturb_hidden_dim", 512),
        perturb_depth=cfg_model.get("perturb_depth", 3),
        latent_perturb_scale=cfg_model.get("latent_perturb_scale", 0.02),
        action_perturb_scale=cfg_model.get("action_perturb_scale", 0.02),
        stability_hidden_dim=cfg_model.get("stability_hidden_dim", 512),
        stability_depth=cfg_model.get("stability_depth", 3),
        beta=cfg_loss.get("beta", 1.0),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True, help=".pt file with `z` and `action` tensors")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--perturbation-count", type=int, default=32)
    args = parser.parse_args()

    model, _ = load_model(args.checkpoint)
    sample = torch.load(args.sample, map_location="cpu")
    z = sample["z"].float()
    action = sample["action"].float()
    if z.ndim in (1, 2):
        z = z.unsqueeze(0)
    if action.ndim == 1:
        action = action.unsqueeze(0)
    out = predict(model, z, action, perturbation_count=args.perturbation_count, threshold=args.threshold)
    serializable = {k: v.detach().cpu().tolist() if torch.is_tensor(v) else v for k, v in out.items()}
    print(yaml.safe_dump(serializable, sort_keys=True))


if __name__ == "__main__":
    main()
