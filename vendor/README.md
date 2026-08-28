# Vendored third-party sources

This directory carries the third-party code and data the study depends on, so the archive can be
audited and rerun without fetching anything from an external repository. Each subtree keeps its own
upstream `LICENSE` file unchanged. Our own modifications are **not** mixed into the upstream trees;
they live in `patches/` so it is obvious what is ours and what is not.

## `TinyRecursiveModels/`
The Tiny Recursive Models codebase our training runner builds on.

| | |
|---|---|
| Upstream | `SamsungSAILMontreal/TinyRecursiveModels` |
| Commit | `c01103738605ba39d1430519b1ee0c62f4c707f8` (2026-03-31) |
| License | **MIT**, Copyright (c) 2025 Samsung Electronics Co., Ltd. |
| State | **pristine** — exported with `git archive` at the pinned commit, unmodified |

## `patches/`
The two changes we apply on top of the pristine tree. Both are ours and are MIT-licensed with the rest
of this package.

- `0001-emit-per-step-progress-and-eval-metrics.patch` — adds 23 lines to `pretrain.py` so each training
  step appends a JSON line to `$TRM_PROGRESS_LOG` and each evaluation prints an `EVAL_METRICS_JSON`
  line. Without this the upstream loop reports only to Weights & Biases, and the per-step learning
  curves in `data/` could not be produced. It changes logging only, never the model, optimizer, data
  pipeline, or loss.
- `adam_atan2.py` — an AdamW shim. The compiled `adam-atan2` optimizer does not build for Blackwell
  (sm_120), so it is dropped in as `adam_atan2.py` inside the TRM tree. Applied identically to every
  configuration, so all comparisons remain matched; the substitution is disclosed in the manuscript.

Apply them with:
```bash
cd TinyRecursiveModels
git apply ../patches/0001-emit-per-step-progress-and-eval-metrics.patch   # or: patch -p1 < ...
cp ../patches/adam_atan2.py .
```
`code/rebuild_env.sh` does this for you.

## `arc-agi-1-raw/`
The raw ARC-AGI-1 tasks consumed by the ARC experiments, assembled into the Kaggle-combined layout that
`dataset/build_arc_dataset.py` expects.

| | |
|---|---|
| Upstream | `fchollet/ARC-AGI` |
| Commit | `399030444e0ab0cc8b4e199870fb20b863846f34` (2025-04-04) |
| License | **Apache-2.0** (no `NOTICE` file exists upstream) |
| Contents | 400 training + 400 evaluation tasks, challenges and solutions |

Note the honest deviation from the original TRM recipe recorded in the manuscript: the `concept`
(ConceptARC) subset is **not** included, only ARC-AGI-1 training + evaluation.

## What is deliberately *not* vendored
The Sudoku-Extreme and Maze-Hard benchmarks (`sapientinc/sudoku-extreme`,
`sapientinc/maze-30x30-hard-1k` on Hugging Face) are fetched at build time by TRM's own
`dataset/build_*_dataset.py`. We do not redistribute them, for two reasons: **neither dataset declares a
license**, so redistribution rights are unclear, and Sudoku-Extreme alone is ~762 MB. This is the single
remaining network dependency of the package; the build is deterministic given those inputs.
