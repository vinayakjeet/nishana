# Kaggle session protocol

Kaggle gives 30 GPU-hours a week on 2x T4 and kills each session when its
wall-clock cap arrives, without ceremony. The training run is designed around
that reality rather than around the hope that it will not happen.

## What transfers between sessions

The LoRA adapter only: a few tens of megabytes of safetensors plus
`adapter-manifest.json`. Optimizer and scheduler state do not transfer. The
precedent for this is the free-tier fine-tune in arXiv 2504.15610, which
resumed a three-epoch QLoRA run across two different GPUs (P100 to T4) carrying
the adapter alone and still converged (loss 1.01 to 0.34). The cost is a small
loss jump at each boundary; kaggle/train.py prints that jump so it is measured
rather than hidden.

## Session 1

1. Upload or clone the repo. `uv` is unavailable on Kaggle kernels, so install
   the two runtime deps the trainer needs directly:
   `!pip install -q unsloth datasets`
2. Build the variant files locally first (`scripts/build_variants.py`) and
   upload `var/training/<tier>-<size>.jsonl`, or rebuild them in-kernel.
3. Start fresh:
   `!python kaggle/train.py --tier synthetic-only --size 500 --rank 16`
4. Watch the step rate in the log. When it stabilises, note seconds per step;
   the next session's plan is only as good as this number.
5. The script stops itself at its planned budget and saves
   `checkpoints/<variant>/adapter_model.safetensors` plus the manifest.
   Download both before the kernel dies.

## Session 2 (possibly a different GPU)

1. Same command plus `--resume`, with `--seconds-per-step` set from session 1's
   measurement. The manifest check refuses to resume if the base model,
   revision, rank, variant, or dataset hash differs in any way.
2. Repeat until the script reports training complete.

## After the last session

```
!python kaggle/train.py --tier synthetic-only --size 500 --rank 16 --predict
```

writes `results/ablation/<tier>-<size>-r<rank>.jsonl` over the held-out eval
set. Bring that file back into the repo, add it to `results/runs.yaml`, and run
`make train-eval`.

## Documenting the actual runs

This file describes the protocol. Once a real two-session run has happened,
record it here under "Run log" with dates, GPU model per session, steps per
session, the loss jump at the boundary, and the checkpoint hash. A resume
mechanism without a documented instance of using it is a claim, not a result.

## Run log

(empty until the first Kaggle run)
