# LVA: Latent Variance Assessment for Stack Stability

This repository implements **Latent Variance Assessment (LVA)** for stack-stability prediction.
LVA uses V-JEPA-style latent world representations to estimate whether a stack is stable by measuring
how much predicted future latents vary under small imagined perturbations.

Stable stacks should produce consistent predicted futures. Unstable stacks should produce higher
variance under perturbation. LVA turns that variance into an interpretable instability score.

## What Is Included

- LVA latent dynamics model
- Learned latent/action perturbation generator
- Predictive-variance instability scoring
- Variance calibration against continuous stability labels
- Optional auxiliary stability head
- V-JEPA latent extraction utility
- Config-driven training entrypoint
- Inference CLI
- Unit tests for LVA tensor shapes and label utilities

## Key Files

- [src/models/lva.py](src/models/lva.py): LVA modules, losses, variance scoring, discounted labels.
- [app/lva/train.py](app/lva/train.py): LVA training entrypoint.
- [app/lva/data.py](app/lva/data.py): trajectory latent dataset loader.
- [app/lva/extract_latents.py](app/lva/extract_latents.py): extracts V-JEPA latents from frame trajectories.
- [app/lva/inference.py](app/lva/inference.py): inference from a trained LVA checkpoint.
- [configs/train/lva/example.yaml](configs/train/lva/example.yaml): example training config.
- [tests/models/test_lva.py](tests/models/test_lva.py): focused LVA tests.

## Data Format

LVA training expects a `.pt` or `.npz` file with trajectory-level latent states and actions:

```text
latents: [episodes, steps, dim] or [episodes, steps, tokens, dim]
actions: [episodes, steps - 1, action_dim]
stability: optional [episodes, steps] continuous labels in [0, 1]
collapse_step: optional [episodes], with -1 for episodes that never collapse
```

If `stability` is omitted, the loader builds discounted survival labels from `collapse_step`.

## Extract Latents

For frame trajectories shaped as `[episodes, steps, channels, frames, height, width]`, extract latents:

```bash
python -m app.lva.extract_latents \
  --input stack_frames.pt \
  --output lva_latents.pt \
  --checkpoint /path/to/vjepa2_checkpoint.pt \
  --model-name vit_large \
  --use-sdpa --use-rope --uniform-power
```

The output can be used directly by the LVA trainer.

## Train

Edit [configs/train/lva/example.yaml](configs/train/lva/example.yaml) so `data.dataset` points to your
latent trajectory file, then run:

```bash
python -m app.main --fname configs/train/lva/example.yaml --devices cuda:0
```

The trainer supports:

- `optimization.stage: dynamics`
- `optimization.stage: calibration`
- `optimization.stage: joint`

Use `dynamics` to train the action-conditioned latent dynamics model first. Use `calibration` to train
the perturbation generator and stability head with the dynamics model frozen.

## Inference

Prepare a `.pt` sample containing:

```text
z: [dim] or [tokens, dim]
action: [action_dim]
```

Then run:

```bash
python -m app.lva.inference \
  --checkpoint /tmp/lva-example/latest.pt \
  --sample sample.pt \
  --threshold 0.5
```

The output includes:

- `instability_score`: predictive variance under perturbations
- `stability`: auxiliary stability-head prediction
- `is_unstable`: thresholded decision, if `--threshold` is provided

## Repository Layout

```text
app/lva/                  LVA training, data loading, inference, latent extraction
src/models/lva.py         LVA model components
configs/train/lva/        LVA configs
tests/models/test_lva.py  LVA tests
app/vjepa*/               V-JEPA training code used as the latent foundation
src/models/               V-JEPA model components
```

## Setup

```bash
conda create -n lva python=3.12
conda activate lva
pip install -e .
```

On macOS, the default `decord` package may not work. If you need to run video decoding locally,
install a compatible replacement before using the V-JEPA video dataset utilities.

## Notes

- This repository is focused on LVA stack-stability modeling.
- V-JEPA code remains in the tree because LVA uses it for latent representation learning and feature extraction.
- Large datasets and model checkpoints are not included.
