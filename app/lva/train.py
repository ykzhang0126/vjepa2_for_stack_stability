# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the root directory of this source tree.

from pathlib import Path

import torch

from app.lva.data import init_data
from src.models.lva import LatentVarianceAssessment
from src.utils.logging import AverageMeter, CSVLogger, get_logger

logger = get_logger(__name__, force=True)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _save_checkpoint(path, model, optimizer, epoch, args, metrics):
    checkpoint = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def _load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "opt" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt"])
    return checkpoint


def main(args, resume_preempt=False):
    folder = Path(args.get("folder", "logs/lva"))
    folder.mkdir(parents=True, exist_ok=True)

    cfg_data = args.get("data", {})
    cfg_model = args.get("model", {})
    cfg_loss = args.get("loss", {})
    cfg_opt = args.get("optimization", {})
    cfg_meta = args.get("meta", {})

    seed = cfg_meta.get("seed", 0)
    torch.manual_seed(seed)
    device = _device()

    dataset, loader = init_data(
        path=cfg_data["dataset"],
        batch_size=cfg_data.get("batch_size", 64),
        num_workers=cfg_data.get("num_workers", 0),
        pin_mem=cfg_data.get("pin_mem", False),
        shuffle=cfg_data.get("shuffle", True),
        gamma=cfg_loss.get("gamma", 0.97),
        drop_last=cfg_data.get("drop_last", False),
    )

    latent_dim = cfg_model.get("latent_dim", dataset.latent_dim)
    action_dim = cfg_model.get("action_dim", dataset.action_dim)
    cfg_model["latent_dim"] = latent_dim
    cfg_model["action_dim"] = action_dim
    model = LatentVarianceAssessment(
        latent_dim=latent_dim,
        action_dim=action_dim,
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
    ).to(device)

    stage = cfg_opt.get("stage", "joint")
    if stage not in ("dynamics", "calibration", "joint"):
        raise ValueError("optimization.stage must be one of: dynamics, calibration, joint")
    if stage == "calibration":
        for p in model.dynamics.parameters():
            p.requires_grad = False
    elif stage == "dynamics":
        for p in model.perturbation.parameters():
            p.requires_grad = False
        for p in model.stability_head.parameters():
            p.requires_grad = False

    lr = cfg_opt.get("lr", 1.0e-4)
    wd = cfg_opt.get("weight_decay", 1.0e-4)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)

    resume_path = cfg_meta.get("resume_checkpoint")
    latest_path = folder / "latest.pt"
    if resume_preempt and latest_path.exists():
        resume_path = str(latest_path)
    start_epoch = 0
    if resume_path:
        checkpoint = _load_checkpoint(resume_path, model, optimizer=optimizer, map_location=device)
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resumed LVA checkpoint from {resume_path} at epoch {start_epoch}")

    perturbation_count = cfg_loss.get("perturbation_count", 16)
    stability_weight = cfg_loss.get("stability_weight", 1.0)
    calibration_weight = cfg_loss.get("calibration_weight", 1.0)
    dynamics_weight = cfg_loss.get("dynamics_weight", 1.0)
    epochs = cfg_opt.get("epochs", 20)

    csv_logger = CSVLogger(
        str(folder / "log.csv"),
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.6f", "loss"),
        ("%.6f", "dynamics_loss"),
        ("%.6f", "calibration_loss"),
        ("%.6f", "stability_loss"),
        mode="+a",
    )

    for epoch in range(start_epoch, epochs):
        loss_meter = AverageMeter()
        dyn_meter = AverageMeter()
        cal_meter = AverageMeter()
        stab_meter = AverageMeter()
        model.train()
        for itr, batch in enumerate(loader):
            z, action, z_next, stability = batch
            z = z.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            z_next = z_next.to(device, non_blocking=True)
            stability = stability.to(device, non_blocking=True)

            optimizer.zero_grad()
            dynamics_loss = z.new_tensor(0.0)
            calibration_loss = z.new_tensor(0.0)
            stability_loss = z.new_tensor(0.0)

            if stage in ("dynamics", "joint"):
                dynamics_loss, _ = model.dynamics_loss(z, action, z_next)
            if stage in ("calibration", "joint"):
                calibration_loss, _, _ = model.calibration_loss(
                    z.detach() if stage == "calibration" else z,
                    action,
                    stability,
                    perturbation_count=perturbation_count,
                )
                stability_loss, _ = model.stability_loss(z.detach() if stage == "calibration" else z, stability)

            loss = (
                dynamics_weight * dynamics_loss
                + calibration_weight * calibration_loss
                + stability_weight * stability_loss
            )
            loss.backward()
            optimizer.step()

            loss_meter.update(float(loss))
            dyn_meter.update(float(dynamics_loss))
            cal_meter.update(float(calibration_loss))
            stab_meter.update(float(stability_loss))
            csv_logger.log(
                epoch + 1,
                itr,
                float(loss),
                float(dynamics_loss),
                float(calibration_loss),
                float(stability_loss),
            )

        metrics = {
            "loss": loss_meter.avg,
            "dynamics_loss": dyn_meter.avg,
            "calibration_loss": cal_meter.avg,
            "stability_loss": stab_meter.avg,
        }
        logger.info(
            "LVA epoch %d/%d loss %.5f dyn %.5f cal %.5f stability %.5f"
            % (epoch + 1, epochs, loss_meter.avg, dyn_meter.avg, cal_meter.avg, stab_meter.avg)
        )
        if cfg_meta.get("save_every_freq", 1) > 0 and (
            (epoch + 1) % cfg_meta.get("save_every_freq", 1) == 0 or epoch + 1 == epochs
        ):
            _save_checkpoint(latest_path, model, optimizer, epoch + 1, args, metrics)

    return model


@torch.no_grad()
def predict(model, z, action, perturbation_count=32, threshold=None):
    was_training = model.training
    model.eval()
    variance, _ = model.variance_score(z, action, perturbation_count=perturbation_count)
    stability = model.stability_head(z)
    if was_training:
        model.train()
    if threshold is None:
        return {"instability_score": variance, "stability": stability}
    return {
        "instability_score": variance,
        "stability": stability,
        "is_unstable": variance > threshold,
    }
