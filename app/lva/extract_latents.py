# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the root directory of this source tree.

import argparse
from pathlib import Path

import numpy as np
import torch

import src.models.vision_transformer as vit


def _load_tensor_payload(path):
    path = Path(path)
    if path.suffix in (".pt", ".pth"):
        return torch.load(path, map_location="cpu")
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: torch.as_tensor(data[k]) for k in data.files}
    raise ValueError(f"Unsupported input file: {path}")


def _load_checkpoint(model, checkpoint_path, checkpoint_key):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint[checkpoint_key] if checkpoint_key in checkpoint else checkpoint
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    return model.load_state_dict(state, strict=False)


@torch.no_grad()
def extract_latents(args):
    payload = _load_tensor_payload(args.input)
    frames = payload[args.frames_key].float()
    actions = payload.get(args.actions_key)
    stability = payload.get(args.stability_key)
    collapse_step = payload.get(args.collapse_step_key)

    if frames.ndim != 6:
        raise ValueError("frames must have shape [episodes, steps, channels, frames, height, width]")
    episodes, steps = frames.shape[:2]
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    encoder = vit.__dict__[args.model_name](
        img_size=args.crop_size,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        use_sdpa=args.use_sdpa,
        use_rope=args.use_rope,
        uniform_power=args.uniform_power,
    ).to(device)
    if args.checkpoint:
        msg = _load_checkpoint(encoder, args.checkpoint, args.checkpoint_key)
        print(msg)
    encoder.eval()

    latents = []
    flat = frames.view(episodes * steps, *frames.shape[2:])
    for start in range(0, flat.size(0), args.batch_size):
        batch = flat[start : start + args.batch_size].to(device, non_blocking=True)
        latents.append(encoder(batch).cpu())
    latents = torch.cat(latents, dim=0).view(episodes, steps, *latents[0].shape[1:])

    out = {"latents": latents}
    if actions is not None:
        out["actions"] = actions.float()
    if stability is not None:
        out["stability"] = stability.float()
    if collapse_step is not None:
        out["collapse_step"] = collapse_step.long()
    torch.save(out, args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=".pt/.npz with frame trajectories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-key", default="frames")
    parser.add_argument("--actions-key", default="actions")
    parser.add_argument("--stability-key", default="stability")
    parser.add_argument("--collapse-step-key", default="collapse_step")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-key", default="target_encoder")
    parser.add_argument("--model-name", default="vit_large")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-sdpa", action="store_true")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--uniform-power", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    extract_latents(parser.parse_args())


if __name__ == "__main__":
    main()
