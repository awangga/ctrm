# Experimental Protocol — Energy–Accuracy Frontier of Recursion Depth

Versioned protocol for reproducing the study. Single consumer GPU (calibrated on NVIDIA RTX 5060 Ti
16 GB, Blackwell sm_120). Built on the Tiny Recursive Models (TRM) codebase
(github.com/SamsungSAILMontreal/TinyRecursiveModels).

## 1. Design plane (P × D)
- **Parameter axis P**: `hidden_size` ∈ {128, 192, 256, 384, 512, 768} (with `L_layers`, `expansion`).
  Parameter count ≈ c·hidden_size² and is **invariant to recursion depth** (verified, see `data/`).
- **Recursion-depth axis D**: `D_eff = H_cycles × L_cycles` ∈ {4, 6, 9, 12, 18, 24, 32}.
  Increasing D raises FLOPs/energy without changing P.

## 2. Tasks
- Primary: **Sudoku-Extreme** (`dataset/build_sudoku_dataset.py`, subsample 1000 × aug).
- Cross-task generality: **Maze-Hard**, **ARC-AGI-1** (subset).

## 3. Primary metric — Joules-to-target-accuracy
Cumulative **net** energy to reach a fixed accuracy target τ per task (not kWh/run of differing
duration). Build the Pareto frontier in (energy, accuracy); locate energy-optimal depth D* and the
saturation point per task.

## 4. Energy measurement (mandatory corrections)
1. Log with **CodeCarbon**; cross-check against integrated `nvidia-smi --query-gpu=power.draw` (1 Hz)
   on **every** run (target agreement > 95%). Achieved: **98.8–99.95% on all 49 faithful-recipe runs**;
   the earlier pilot runs, which contribute no reported number, fall as low as 95.9%. Per-run agreement
   for both regimes is tabulated in `data/ablation_reports/run_energy_reconciliation.md`.
2. Measure **idle/baseline** power and **subtract** to obtain net energy (additive-bias correction).
3. **Lock GPU clocks**, same GPU for all runs (eliminate inter-device variance); log temperature.
4. Report **training and inference energy separately**.
5. Do **not** rely on raw `nvidia-smi` alone (sensor undersampling is a known bias).

## 5. ⚠ Evaluation-cost rule (critical, from calibration)
Training is cheap (a few optimizer steps/epoch on the subsample); **evaluation dominates** — a full
pass over the Sudoku-Extreme test set (~4.2×10⁵ instances) with 16-step recursive inference does not
finish in 900 s. **Evaluate on a fixed subset at fixed step intervals**, never the full set per
checkpoint. This study uses **512 puzzles** (set by `rebuild_env.sh` and `make_small_eval.py`), held
identical across every configuration so evaluation cost cannot bias a comparison. Use **fixed recursion depth per configuration** (no depth-curriculum) to avoid
confounding the energy measurement.

## 6. IsoFLOP / budget procedure
For each of ~5–6 fixed compute/energy budgets, sweep (P, D) at that budget; the loss/energy minimum is
the compute-optimal allocation. Fit P*(C), D*(C), L*(C) on the smaller budgets and **validate by
predicting** the held-out larger budget; report relative prediction error.

## 7. Statistics
- Power-law fits in log space via `scipy.optimize.curve_fit`; **bootstrap confidence intervals** on
  exponents (≥1000 resamples), consistent with the actual run count.
- Replicate key points (frontier minima + hold-out) 3×; others 1–2×.
- Baselines matched on **iso-FLOP/energy** (non-recursive + HRM variants in the codebase).

## 8. Reproduce the calibration
```bash
# env (Blackwell needs cu128)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --torch-backend=cu128
uv pip install -r requirements.txt
# microbenchmark (params + throughput + energy sweep)
python code/microbench.py            # writes data/microbench_results.csv
# full-pipeline calibration point (uses TRM repo + CodeCarbon/nvidia-smi sampler)
bash code/calibrate_pipeline.sh
```

## 9. Files
Code:
- `code/run_recipe.py` — **faithful-recipe runner** (real-accuracy regime): trains TRM with EMA on the
  augmented dataset, evaluates on a fixed subset at fixed intervals, and emits per run
  `progress_<tag>.jsonl` (train loss/step + eval/checkpoint), `pw_<tag>.csv` (1 Hz power),
  `emissions_<tag>.csv` (CodeCarbon), plus a self-describing summary row. Env vars:
  `HIDDEN/DEPTH/BATCH/STEPS/SEED/DATA/OUT_DIR/NEVAL/EMA`.
- `code/stats_table.py` — Welch t-test, one-way ANOVA, Bonferroni from the summary CSVs (reproduces the
  reported significance numbers).
- `code/joules_to_target.py`, `code/fit_extrapolation.py`, `code/energy_xval_report.py`,
  `code/analyze_crosstask.py`, `code/consolidate_ablation.py` — analysis (Joules-to-target, energy
  extrapolation hold-out, CodeCarbon-vs-nvidia-smi cross-validation, cross-task, ablation table).
- `code/microbench.py` — instantiates TRM at each (hidden, depth); counts params; times forward+backward;
  samples power; writes per-config throughput/energy/memory.
- `code/calibrate_pipeline.sh` — short end-to-end training run with GPU power sampling.
- `code/adam_atan2_fallback.py` — AdamW shim used when the compiled `adam-atan2` optimizer is unavailable.
- `vendor/` — third-party sources vendored so the package needs no network clone: the upstream Tiny
  Recursive Models tree pinned at commit `c011037` (MIT, Samsung Electronics), the raw ARC-AGI-1 tasks
  (Apache-2.0, `fchollet/ARC-AGI` @ `3990304`), and in `vendor/patches/` the two changes we apply to TRM
  (per-step progress logging; AdamW shim for sm_120). The upstream trees are unmodified; see
  `vendor/README.md` and `THIRD_PARTY_LICENSES.md`.
- `code/reconcile_totals.py` — recomputes run counts, net energy totals, and per-run
  CodeCarbon-vs-`nvidia-smi` agreement ranges from every committed summary CSV, splitting the
  faithful-recipe and pilot regimes. Run as `python3 code/reconcile_totals.py --data-root data`.
- `code/resolve_doi.py`, `code/resolve_arxiv.py` — auxiliary: resolve reference DOIs (CrossRef/arXiv).

Data:
- `data/microbench_results.csv` — measured calibration data (RTX 5060 Ti).
- `data/recipe_out/` — real-accuracy Sudoku-Extreme (depth axis h512 3 seeds; width axis h256/h512/h768).
  `recipe_summary.csv` + per-run `progress_/pw_/emissions_`.
- `data/maze_depth_out/` — real-accuracy Maze-Hard depth grid (D9/D18/D36 @ h256, 5 seeds).
- `data/arc_depth_out/` — real-accuracy ARC-AGI-1 depth grid (D9/D18/D36 @ h256, 5 seeds). Built from
  ARC-AGI-1 training+evaluation (no ConceptARC), 1000x augmentation, test truncated to 512; `GROUPS=3080`
  (= 800 groups x mean_puzzle_examples 3.85) so labelled steps equal actual optimizer steps.
- `data/maze_real_out/` — Maze faithful-recipe probe.
- `data/ablation_reports/` — analysis reports, the consolidated ablation table (`ablation_master.csv`),
  pooled learning curves (`learning_curves_all.csv`), per-seed accuracies, Joules-to-target, significance
  and cross-task reports, and `run_energy_reconciliation.md` (run counts and energy totals).
- `data/budget_out/`, `data/frontier_out/`, `data/isoflop_out/`, `data/scale_out/`,
  `data/converge_out/`, `data/maze_out/`, `data/arc_smoke_out/` — **pilot (toy, under-trained)**; kept as
  a feasibility record only, superseded by the faithful-recipe regime above. Nine replication runs in
  `budget_out` (seeds 0–2) have accuracy but no attributable per-seed energy, because their power logs
  were written without a seed suffix and overwrote one another; they are excluded from energy totals.

## 10. Real-accuracy regime (faithful recipe — the manuscript results)
The pilot (§11) validated the pipeline but is under-trained. The reported results use the faithful TRM
recipe (EMA on, augmented dataset ~1M examples, thousands of optimizer steps, hidden >= 256), at
**iso-compute** (params x D_eff x batch x steps held constant across a comparison), e.g.:
```bash
# Sudoku depth axis @ h512 (D9 double the steps of D18/D36 to hold compute constant)
env DATA=data/sudoku-aug1k OUT_DIR=out/recipe HIDDEN=512 EMA=True NEVAL=... \
    DEPTH=9  BATCH=192 STEPS=50000 SEED=0 python code/run_recipe.py
env ... DEPTH=18 BATCH=192 STEPS=25000 SEED=0 python code/run_recipe.py
env ... DEPTH=36 BATCH=96  STEPS=25000 SEED=0 python code/run_recipe.py
# repeat SEED=1,2; then stats_table.py over recipe_summary.csv
```
Metric: exact accuracy for Sudoku (discriminating, 36–62%); **token accuracy for Maze** (exact = 0 there,
uninformative). Energy is idle-corrected and cross-validated CodeCarbon vs `nvidia-smi` (98.8–99.95%).

## 11. Pilot frontier (validasi pipeline end-to-end)
```bash
# 1) test set kecil (atasi bottleneck eval)
python code/make_small_eval.py data/sudoku-cal data/sudoku-pilot 512
# 2) buat pretrain.py mencetak metrik eval (default hanya ke W&B)
python code/patch_pretrain_print_metrics.py pretrain.py
# 3) sapu recursion depth pada budget tetap, rekam akurasi/loss + energi net
bash code/run_frontier.sh
# 4) ringkas jadi tabel + report
python code/analyze_frontier.py
```
Pilot (hidden=256, 1562 step/config, eval subset 512) memvalidasi alur dan mereproduksi
**energi net ∝ recursion depth** dalam run training nyata (D_eff 9/18/36 → 3.34/5.94/11.40 Wh).
Akurasi pilot ≈ 0 (under-trained); frontier sebenarnya menaikkan budget hingga mencapai τ.
