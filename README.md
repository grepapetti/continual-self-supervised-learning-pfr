# Continual Self-Supervised Learning with PFR — a reproduction

Reviving a four-year-old research codebase and reproducing **Projected Functional
Regularization** (Gomez-Villa et al., *CVPR Workshop on Continual Learning in
Computer Vision*, 2022) — then finding out why the result does not come back.

Project for the *Streaming Data Analytics* course, **Politecnico di Milano** —
Prof. Emanuele Della Valle. Author: **Greta Papetti**.

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/PyTorch%20Lightning-1.5.4-792EE5)
![CIFAR-100](https://img.shields.io/badge/CIFAR--100-4%20tasks-006699)
![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-FFBE00?logo=weightsandbiases&logoColor=black)
![Reproducibility](https://img.shields.io/badge/focus-reproducibility-lightgrey)

---

## The short version

The pipeline reproduces. **The result does not** — and that turned out to be the
interesting part.

PFR is supposed to beat naive fine-tuning by preventing catastrophic forgetting.
In my runs it ties with it. The reason is not a bug: at 100 epochs per task,
**naive fine-tuning doesn't forget in the first place**, so there is nothing for
PFR to prevent. Meanwhile the feature-distillation baseline collapses by 13
points — because its regularization weight was tuned for a budget five times
longer than mine.

Both findings come from the same place: *published hyperparameters are not
properties of a method, they are properties of a method in a setup.*

## Setup

Barlow Twins + ResNet-18 on **CIFAR-100 split into 4 tasks of 25 disjoint
classes**, class-incremental, 100 epochs per task on an NVIDIA L4. Four
conditions, differing in one thing only — the distiller wrapped around the
self-supervised method:

| Condition | Run |
|---|---|
| Upper bound | **Joint** — all 100 classes at once |
| No continual strategy | **FT** — Barlow Twins trained sequentially |
| Baseline | **FD** — feature distillation, λ = 25 |
| Paper's method | **PFR** — projected functional regularization |

## Results

Paper protocol (100-class linear probe). A = final accuracy, F = forgetting,
FWT = forward transfer. The paper reports accuracy only.

| Condition | Method | A | F | FWT | A (paper) |
|---|---|---|---|---|---|
| Full training | Joint | 59.0 | — | — | 60.6 |
| No CL | FT | 54.0 | 0.91 | 41.9 | 56.8 |
| CL baseline | FD | **41.1** | 0.19 | 34.0 | 57.2 |
| CL (paper) | PFR | 54.2 | 1.28 | 41.2 | 60.1 |

The paper's ranking is FT < FD < PFR, with PFR nearly touching the upper bound.
Mine is FD ≪ FT ≈ PFR, with PFR five points short. Note FD's forgetting of
0.19 — the lowest of all, and worthless: it forgets nothing because it learned
nothing. Accuracy and forgetting have to be read together.

## Three findings

**Feature distillation didn't over-protect — it never learned.** I ruled out the
evaluation protocol, then the code (rewrote the Barlow Twins loss from the paper
formula and matched the repo's to ~1e-4; confirmed the frozen teacher receives no
gradient and that the distillation term does reach the encoder). The cause: the
distillation gradient on the encoder is **~250× larger** than the task loss
gradient. The model spends every step obeying the constraint instead of learning.
`--lars` hides this — it rescales step *length* per layer but not *direction*, so
the weights grow in scale while barely rotating. Re-running with λ = 5 and
nothing else changed lifted mean accuracy from 41% to 50% and plasticity from
+1 to +7 points. Not a bug: a hyperparameter correct at 500 epochs and wrong at
100.

**Most "forgetting" here is class interference, not information loss.** The same
FT model, on the same representations, scores **BWT +3.9** under task-aware
evaluation and **−6.2** under seen-pooled (kNN agrees: +2.9 vs −4.5). Task-aware
asks *are the first task's classes still separable from each other?* — yes, and
increasingly so. Seen-pooled asks *are they still separable from everything
else?* — and that degrades as each new group of classes adds candidates to
confuse them with. A task-aware protocol cannot see this at all.

**Reviving the code took three patches, all validated rather than assumed.**
`torch._six` (removed in PyTorch 2.0), `compute_on_step` (removed from
torchmetrics), and a `_LRScheduler` isinstance check in Lightning 1.5.4 — plus
`pip<24.1`, because the pinned `torch>=1.7.*` specifier is not valid for modern
pip. See [`scripts/apply_compat_patches.py`](scripts/apply_compat_patches.py).

## What's here

| | |
|---|---|
| [`PFR_riproduzione.ipynb`](PFR_riproduzione.ipynb) | End-to-end pipeline: setup, patches, four trainings, linear evaluations, R matrices, CL metrics, probe/kNN variants, the FD investigation |
| [`report/PFR_report_Papetti.pdf`](report/PFR_report_Papetti.pdf) | Full write-up — theory-to-code mapping, methodology, results, the investigation (in Italian) |
| [`COME_RIPRODURRE.md`](COME_RIPRODURRE.md) | Step-by-step reproduction guide, Colab and command line (in Italian) |
| [`figures/`](figures) | All plots, generated from the logs |
| `main_pretrain.py`, `main_continual.py`, `main_linear.py` | Entry points inherited from the upstream codebase |
| `cassle/` | The CaSSLe framework: SSL methods and continual distillers |

## Reproducing

Full instructions in [`COME_RIPRODURRE.md`](COME_RIPRODURRE.md). In short: open the
notebook in Colab on a GPU runtime, run Section 1 (clone, dependencies, W&B login,
compatibility patches), then the sections in order. Budget roughly 20 minutes per
task per method on an L4, plus ~4 GB for checkpoints. CIFAR-100 downloads itself.

A Weights & Biases account is needed for the sections that read metrics back from
the run history; `WANDB_MODE=offline` plus `wandb sync` also works.

## Credits

Original method and code: [alviur/CVPR_PFR](https://github.com/alviur/CVPR_PFR),
itself a fork of the CaSSLe framework. Upstream README kept as
[`README_upstream.md`](README_upstream.md). Project brief from the
[Streaming Data Analytics](https://emanueledellavalle.org/teaching/streaming-data-analytics-2025-26/)
course; original assignment repository
[here](https://github.com/Streaming-Data-Analytics/2025-2026_10b_CL-Code-Reproducibility_PFR).
MIT licensed.
