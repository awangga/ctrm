# Experimental Protocol: the Energy Cost of Recursion Depth

Protocol for the manuscript *"The energy cost of recursion depth: half the joules for the same accuracy"*
(target: *Sustainable Computing: Informatics and Systems*).

Versioned protocol for reproducing the study. Single consumer GPU (calibrated on NVIDIA RTX 5060 Ti
16 GB, Blackwell sm_120). Built on the Tiny Recursive Models (TRM) codebase
(github.com/SamsungSAILMontreal/TinyRecursiveModels).

## 1. Design plane (P × D)
- **Parameter axis P**: `hidden_size` ∈ {128, 192, 256, 384, 512, 768} (with `L_layers`, `expansion`).
  Parameter count ≈ c·hidden_size² and is **invariant to recursion depth** (verified, see `data/`).
- **Recursion-depth axis D**: `D_eff = H_cycles × L_cycles`. The pilot swept
  {4, 6, 9, 12, 18, 24, 32}; the faithful-recipe grids that produce every reported result use
  **{9, 18, 36}**. Increasing D raises FLOPs/energy without changing P.

## 2. Tasks
- Primary: **Sudoku-Extreme** (`dataset/build_sudoku_dataset.py`, subsample 1000 × aug).
- Additional tasks: **Maze-Hard**, **ARC-AGI-1** (evaluation subset built per §5.2). Each task's metric is validated against a
  trivial baseline (§5.1) before any depth effect is read from it; a task that fails the check keeps
  its energy comparison and loses its accuracy claim.

## 3. Primary metric: Joules-to-target-accuracy
Cumulative **net** energy to reach a fixed accuracy target τ per task (not kWh/run of differing
duration). Each measured configuration is placed in the (energy, accuracy) plane and the comparison is
read off the measured points: per depth, the energy needed to reach τ, and the energy each depth needs to
reach the accuracy the deepest setting attains. **Correction (September 2026):** earlier versions of this
protocol said to build a Pareto frontier and locate an energy-optimal depth `D*` per task. The grid is
three measured depths {9, 18, 36}; nothing on the accuracy axis is fitted or interpolated, so no `D*` is
estimated and none is reported (see §6). A task whose metric fails §5.1 (below its trivial baseline)
yields no accuracy claim at all and contributes only the energy comparison.

## 4. Energy measurement (mandatory corrections)
1. Log with **CodeCarbon**; cross-check against integrated `nvidia-smi --query-gpu=power.draw` (1 Hz)
   on **every** run (target agreement > 95%). NOTE: both readers use the same NVML sensor, so the
   agreement bounds the 1 Hz integration error, NOT the accuracy of the sensor itself; absolute energies
   carry the sensor's uncertainty and only the ordering is protected by the iso-environment.
   Instruments used here: CodeCarbon 3.2.8 `OfflineEmissionsTracker` (`measure_power_secs=5`,
   `tracking_mode=machine`, ISO `IDN`); GPU driver 595.71.05 for the first 37 faithful-recipe runs and 595.84
   for the last 35 (12 seed extensions plus 23 runs of phases BM and BS). GPU energy only: CodeCarbon's CPU (~6.5 W, TDP-derived) and RAM (20 W default) figures are
   models and are excluded; PSU, host idle, embodied carbon and inference energy are out of scope. Achieved: **98.82–99.95% on all 72 faithful-recipe runs**
   (99.11–99.95% on the 37 runs under the first driver, 98.82–98.97% on the 35 under the second);
   the earlier pilot runs, which contribute no reported number, fall as low as 95.9%. Per-run agreement
   is the `gross_agree_pct` column of each summary CSV; `data/ablation_reports/run_energy_reconciliation.md`
   gives its range per regime.
2. Subtract a fixed idle constant (`IDLE_W = 4.7` W in `code/run_recipe.py`, just above the 4.04 W sampler
   floor observed across all released logs) to obtain net energy; discard runs shorter than 120 s as
   aborted, and check every reported run by hand for completeness (wall time and step count). The 120 s
   filter alone is **not** sufficient: a truncated 161 s run once passed it, so the manual check on wall
   time and step count is mandatory, not advisory. Reading ~150 W on a GPU that should idle at 4-5 W is
   the signal that a second run is sharing the card; verify with
   `nvidia-smi --query-compute-apps`, which must show 0 processes before a launch and 1 after it.
3. Same GPU for all runs (eliminates inter-device variance). *Planned, not applied:* clock locking and
   temperature logging.
4. *Planned, not applied:* separate training and inference energy. The reported figure is net training
   energy including the pinned periodic evaluation.
5. Do **not** rely on raw `nvidia-smi` alone (sensor undersampling is a known bias).

## 5. Evaluation-cost rule (critical, from calibration)
Training is cheap (a few optimizer steps/epoch on the subsample); **evaluation dominates**: a full
pass over the Sudoku-Extreme test set (~4.2×10⁵ instances) with 16-step recursive inference does not
finish in 900 s. **Evaluate on a fixed subset at fixed step intervals**, never the full set per
checkpoint. This study uses **512 puzzles** on Sudoku-Extreme and Maze-Hard (set by `rebuild_env.sh` and
`make_small_eval.py`) and, on ARC-AGI-1, the 419 unaugmented test examples of the 400 evaluation tasks
(§5.2), each held identical across every configuration so evaluation cost cannot bias a comparison. Use **fixed recursion depth per configuration** (no depth-curriculum) to avoid
confounding the energy measurement.

## 5.1 Three measurement pitfalls (the methodological contribution)
These generalise beyond recursion depth; each one is a rule this protocol enforces.

1. **Evaluation cost grows with the knob under test.** Deeper recursion makes each evaluation pass more
   expensive, so an unpinned evaluation charges deep configurations for their own measurement. Fix the
   subset and the step interval across the whole comparison (§5).
2. **A reduced-scale screen can rank the eventual winner last.** The pilot regime (§11) is retained in
   this package precisely because its ordering does not survive the faithful recipe. Screen cheaply if
   you must, but decide on the faithful recipe.
3. **A proxy metric must be scored against a trivial baseline before any effect is read from it.**
   Run `code/trivial_baselines.py` (majority-class and copy-input on the same evaluation subset)
   first. On Maze-Hard copy-input scores **87.51%** token accuracy while every trained configuration
   reaches 86.5-86.8%, so that metric measures nothing about task mastery at this budget and supports no
   depth conclusion. Measured floors (majority / copy-input): Sudoku exact 0.00%, ARC token 41.82%/60.75% on
   the valid subset of §5.2 (the 25.00% of v5 came from the defective subset), Maze token 50.03%/87.51%.

## 5.2 Building a valid evaluation subset (corrected September 2026, v6)
`code/make_small_eval.py` takes the **first N rows** of the test split. That is valid **only for test
splits that are not augmented**: on Sudoku-Extreme and Maze-Hard the first 512 rows are 512 distinct
puzzles. The ARC-AGI-1 test split is augmented to about 1001 examples per task, so its first 512 rows all
belong to the first task. The subset every ARC run before v6 used, `arc1-aug1k-e512`, therefore held
augmentations of a **single** task (512 examples, 72 distinct output grids, 48 labelled positions in
every example). We found this in an internal audit.

Protocol step: for any task whose test split is augmented, build the subset with
`code/make_group_eval.py`, which keeps the **first puzzle of every group**, i.e. the original unaugmented
example of each task, copies the training split unchanged and does not remap puzzle identifiers:
```bash
python code/make_group_eval.py data/arc1-aug1k data/arc1-aug1k-g400
```
For ARC-AGI-1 this gives `arc1-aug1k-g400`: **400 tasks, 419 examples** (some tasks carry two test
examples), 419 distinct output grids. Before training on a new subset, check that the number of distinct
labels is close to the number of examples, then recompute the trivial baselines on it (§5.1, pitfall 3):
on `arc1-aug1k-g400` majority scores 41.82% and copy-input 60.75% token accuracy. The 25 ARC runs scored on
the old subset keep their energy records, which are valid, and support no accuracy claim.

## 6. Compute budget: what was actually run (corrected September 2026)
**Superseded plan, kept for the record.** This section previously specified a multi-budget IsoFLOP
procedure: for each of ~5–6 fixed compute/energy budgets, sweep (P, D) at that budget, fit `P*(C)`,
`D*(C)`, `L*(C)` on the smaller budgets, and validate by predicting the held-out larger budget. That
procedure was **never executed**. It is recorded here rather than deleted, because the planning history
should not be rewritten, but it describes nothing in the released data.

**What was run.** One fixed compute budget per comparison (iso-compute, §10: params × D_eff × batch ×
steps held constant across the configurations being compared), with recursion depth swept over
{9, 18, 36} on Sudoku-Extreme and Maze-Hard and over its end points {9, 36} on ARC-AGI-1 (valid subset) and, on Sudoku-Extreme only, width over {256, 512, 768}. There is no sweep
over 5–6 budgets and no held-out budget. The accuracy axis is reported as measured, per task: no curve is
fitted to it and no extrapolation is made to budgets or scales that were not run. The measured range is
roughly 1.5–2 dex on a single GPU, below the range at which an extrapolation claim would carry weight,
which is why the study reports measurements on that axis and calls none of them a law.

**Where fitting is used.** Quantitative fitting is confined to the energy cost model (§7): per-step energy
against (params × D_eff), fitted on the 14 microbenchmark configurations, exponent b = 0.84 with
bootstrap 95% CI [0.75, 0.92] and R² = 0.98. `code/validate_costmodel.py` tests that fit against realised
per-run energy and reports the regimes in which it fails, so its stated scope is enforced rather than
assumed.

## 7. Statistics
- **Scoring rule, identical on every task:** a configuration is scored by its BEST evaluation checkpoint
  (`best_exact_pct` on Sudoku, `best_token_pct` on Maze and ARC). `best_token_pct` is derived from the
  per-step logs by `code/add_best_token.py`; `final_token_pct` is kept alongside so the final-checkpoint
  reading can be recomputed. `code/estimator_sensitivity.py` reports the depth contrast under four
  checkpoint rules (best, final, last-five mean, median) plus an exact permutation test, so the verdict
  is not an artifact of the rule.
- Welch two-sided t-tests and one-way ANOVA (`code/stats_table.py`), **Bonferroni 0.05/3 per axis**
  (threshold 0.0167), fixed by pre-registration because the depth design has three levels. Across the
  whole manuscript 13 contrasts are reported; a reader who wants one family applies 0.05/13 = 0.0038.
- Power-law fits in log space via `scipy.optimize.curve_fit`, applied to the **energy cost model only**
  (§6) and never to the accuracy axis; **bootstrap confidence intervals** on exponents (≥1000 resamples),
  consistent with the actual run count. `code/validate_costmodel.py` checks
  the fitted cost model against realised per-run energy and reports where it fails.
- Replicate key points 3×; five seeds on the Maze-Hard and ARC-AGI-1 depth grids.
- Where memory forces the deepest cell to a smaller batch, restore its effective batch with gradient
  accumulation (`ACCUM=2`) so the contrast tests depth alone. Applied to ARC D36 on the valid subset
  (24 x 2 = 48) and, as a robustness check, to Sudoku D36 (96 x 2 = 192).
- Baselines matched on **iso-FLOP/energy**: a non-recursive transformer on Sudoku-Extreme only (the ARC
  counterpart failed on a configuration error and was not rerun; the HRM variant was not retrained),
  plus the trivial baselines of §5.1.
- **Adding seeds requires pre-registration first** (`PREREGISTRATION.md`, `prereg_BM.md`): the run count,
  the targeted contrast and the commitment to report any outcome are written and committed before launch,
  and the analysis is run once, after every run finishes.

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
- `code/run_recipe.py`: **faithful-recipe runner** (real-accuracy regime): trains TRM with EMA on the
  augmented dataset, evaluates on a fixed subset at fixed intervals, and emits per run
  `progress_<tag>.jsonl` (train loss/step + eval/checkpoint), `pw_<tag>.csv` (1 Hz power),
  `emissions_<tag>.csv` (CodeCarbon), plus a self-describing summary row. Env vars:
  `HIDDEN/DEPTH/BATCH/STEPS/SEED/DATA/OUT_DIR/NEVAL/EMA/GROUPS/ARCH/TRM_DIR`, plus `ACCUM` (micro-batches
  per optimizer step, effective batch = `BATCH`x`ACCUM`) and `SAVE_PREDS` (directory for per-instance
  evaluation records); the last two require the third vendor patch and default to off.
- `code/stats_table.py`: Welch t-test, one-way ANOVA, Bonferroni from the summary CSVs (reproduces the
  reported significance numbers); `code/sig_test.py` is the single-contrast helper it grew from.
- `code/trivial_baselines.py`: majority-class and copy-input predictors scored on the same evaluation
  subset with the same per-sequence metric as the training loop. Run this before reading any
  effect from a proxy metric (§5.1).
- `code/add_best_token.py`: derives the `best_token_pct` column from `progress_<tag>.jsonl` for every run,
  so the best-checkpoint rule is applied identically on every task; `final_token_pct` is kept alongside.
- `code/estimator_sensitivity.py`: the depth contrast recomputed under four checkpoint rules (best, final,
  last-five mean, median) with an exact permutation test, so the verdict is not an artifact of the rule.
  Its ARC part still reads the old-subset grid (`arc_depth_out/`) and supports no claim.
- `code/validate_costmodel.py`: compares the microbenchmark cost model against realised per-run energy and
  reports the regimes where it fails.
- `code/make_manuscript_figures.py`: regenerates every manuscript figure, and the faithful-recipe
  Joules-to-target and iso-accuracy energies, from the committed summary CSVs.
- `code/run_BM_chain.sh`: the phase BM run chain (§12); `code/analyze_BM.py`: its pre-specified analysis,
  written and committed before any run of that batch finished; `code/run_accum_check.sh`: the equivalence
  check that had to pass before the chain was launched: ARC `D9` at batch 48 x accum 1 against batch 24 x
  accum 2, whose train-loss curves must coincide if the accumulation patch is correct (record in
  `data/accum_check_out/`).
- `code/make_group_eval.py`: builds a valid evaluation subset from an augmented test split (§5.2);
  `code/run_arc_g400.sh`: the phase BS chain (ARC D9 and accumulated D36 on `arc1-aug1k-g400`, 10 runs);
  `code/analyze_BS.py`: its pre-specified analysis (amendment 5), committed before any model accuracy on
  the new subset was read.
- `code/patch_pretrain_accum_preds.py`: the third TRM patch (identical to
  `vendor/patches/0003-accumulation-and-per-instance-eval.py`): `TRM_ACCUM` for gradient accumulation and
  `TRM_EVAL_PREDS` for per-instance evaluation logging, both inert unless the variable is set.
- `code/patch_pretrain_print_metrics.py`, `code/make_small_eval.py`: the pilot-path equivalent of patch
  0001 and the first-N-rows evaluation subset builder, valid only for test splits that are not augmented
  (§5.2).
- `code/rebuild_env.sh`: rebuilds the TRM tree and venv from `vendor/` and applies patch 0001 and the
  AdamW shim; the third patch is applied separately, before the phase BM chain only.
- `code/run_frontier.sh`, `code/run_isoflop.py`, `code/run_budget.py`, `code/run_scale_sweep.py`,
  `code/run_converge.py`, `code/run_maze_sweep.py`, `code/run_replicate.py`, `code/run_xval_seeds.py`,
  `code/plot_frontier.py`, `code/plot_isoflop.py`, `code/plot_budget.py`: the pilot sweeps and their plots
  (§11); `code/run_arcdepth_s34.sh`: the pre-registered ARC seed 3/4 chain.
- `code/joules_to_target.py` (pilot grids only; the faithful-recipe Joules-to-target and iso-accuracy
  energies are computed by `code/make_manuscript_figures.py`), `code/fit_extrapolation.py`, `code/energy_xval_report.py`,
  `code/analyze_crosstask.py`, `code/consolidate_ablation.py`: analysis (Joules-to-target, energy
  extrapolation hold-out, CodeCarbon-vs-nvidia-smi cross-validation, cross-task, ablation table).
- `code/microbench.py`: instantiates TRM at each (hidden, depth); counts params; times forward+backward;
  samples power; writes per-config throughput/energy/memory.
- `code/calibrate_pipeline.sh`: short end-to-end training run with GPU power sampling.
- `code/adam_atan2_fallback.py`: AdamW shim used when the compiled `adam-atan2` optimizer is unavailable.
- `vendor/`: third-party sources vendored so the package needs no network clone: the upstream Tiny
  Recursive Models tree pinned at commit `c011037` (MIT, Samsung Electronics), the raw ARC-AGI-1 tasks
  (Apache-2.0, `fchollet/ARC-AGI` @ `3990304`), and in `vendor/patches/` the three changes we apply to TRM:
  `0001-emit-per-step-progress-and-eval-metrics.patch` (per-step progress logging), `adam_atan2.py` (AdamW
  shim for sm_120), and `0003-accumulation-and-per-instance-eval.py` (gradient accumulation and
  per-instance evaluation logging, both off unless `TRM_ACCUM` / `TRM_EVAL_PREDS` are set, so the released
  runs are unaffected). The upstream trees are unmodified; see `vendor/README.md` and
  `THIRD_PARTY_LICENSES.md`.
- `code/reconcile_totals.py`: recomputes run counts, net energy totals, and per-run
  CodeCarbon-vs-`nvidia-smi` agreement ranges from every committed summary CSV, splitting the
  faithful-recipe and pilot regimes. Run as `python3 code/reconcile_totals.py --data-root data`.
- `code/resolve_doi.py`, `code/resolve_arxiv.py`: auxiliary: resolve reference DOIs (CrossRef/arXiv).

Data:
- `data/microbench_results.csv`: measured calibration data (RTX 5060 Ti).
- `data/recipe_out/`: real-accuracy Sudoku-Extreme (depth axis h512 3 seeds; width axis h256/h512/h768).
  `recipe_summary.csv` + per-run `progress_/pw_/emissions_`.
- `data/maze_depth_out/`: real-accuracy Maze-Hard depth grid (D9/D18/D36 @ h256, 5 seeds).
- `data/sudoku_d36_accum_out/`: Sudoku D36 at effective batch 192 (96 x 2 accumulation), 3 seeds (BM A2).
- `data/arc_d9_g400_out/`, `data/arc_d36_g400_out/`: ARC-AGI-1 D9 and D36 (24 x 2 accumulation, effective
  batch 48) on the valid subset `arc1-aug1k-g400`, 5 seeds each. Every reported ARC result comes from these.
- `data/arc_depth_out/`: ARC-AGI-1 depth grid (D9/D18/D36 @ h256, 5 seeds) scored on the DEFECTIVE subset
  `arc1-aug1k-e512` (§5.2); energy record only. `data/arc_d36_accum_out/` (BM A1) and
  `data/arc_d9_preds_out/` (BM C) are on the same defective subset and are likewise a record only. Built from
  ARC-AGI-1 training+evaluation (no ConceptARC), 1000x augmentation; `GROUPS=3080`
  (= 800 groups x mean_puzzle_examples 3.85) so labelled steps equal actual optimizer steps. **Pass it as
  `env GROUPS=3080 ... python code/run_recipe.py`, never with `export`:** `GROUPS` is a special bash
  variable and is not exported, so `export GROUPS=3080` is silently ignored and the runner falls back to
  its default of 1000, which changes the epoch count without any error being raised.
- `data/maze_real_out/`: Maze faithful-recipe probe.
- `data/ablation_reports/`: analysis reports, the consolidated ablation table (`ablation_master.csv`),
  pooled learning curves (`learning_curves_all.csv`), per-seed accuracies, Joules-to-target, significance
  and cross-task reports, and `run_energy_reconciliation.md` (run counts and energy totals).
- `data/budget_out/`, `data/frontier_out/`, `data/isoflop_out/`, `data/scale_out/`,
  `data/converge_out/`, `data/maze_out/`, `data/arc_smoke_out/`: **pilot (toy, under-trained)**; kept as
  a feasibility record only, superseded by the faithful-recipe regime above. Nine replication runs in
  `budget_out` (seeds 0–2) have accuracy but no attributable per-seed energy, because their power logs
  were written without a seed suffix and overwrote one another; they are excluded from energy totals.

## 10. Real-accuracy regime (faithful recipe: the manuscript results)
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
BEFORE reading any effect from a proxy metric, score a trivial predictor on the same subset
(`code/trivial_baselines.py`): majority-class and copy-input. On Maze-Hard copy-input scores 87.51% token
accuracy, above every trained configuration, so that metric supports no depth conclusion at this budget.

Metric: exact accuracy for Sudoku (discriminating, 31.7–62.4%); **token accuracy for Maze and ARC** (exact = 0
there); every configuration is scored by its BEST evaluation checkpoint (`best_exact_pct` / `best_token_pct`),
the same rule on every task. Energy is idle-corrected and cross-validated CodeCarbon vs `nvidia-smi` (98.82–99.95% on all 72
faithful-recipe runs).

## 11. Pilot frontier (end-to-end pipeline validation)
```bash
# 1) small test set (removes the evaluation bottleneck)
python code/make_small_eval.py data/sudoku-cal data/sudoku-pilot 512
# 2) make pretrain.py print eval metrics (upstream reports only to W&B)
python code/patch_pretrain_print_metrics.py pretrain.py
# 3) sweep recursion depth at a fixed budget, recording accuracy/loss + net energy
bash code/run_frontier.sh
# 4) summarise into a table + report
python code/analyze_frontier.py
```
The pilot (hidden = 256, 1562 steps/config, 512-instance eval subset) validated the pipeline and
reproduced **net energy ∝ recursion depth** in real training runs (D_eff 9/18/36 → 3.34/5.94/11.40 Wh).
Pilot accuracy is ≈ 0 (under-trained); the real frontier raises the budget until τ is reached. Keep this
regime in view when reading §5.1 pitfall 2: the pilot ordering does not survive the faithful recipe, so a
cheap screen decides nothing.

## 12. Pre-registered batches BM and BS (completed)
Adding runs after seeing a result requires pre-registration first (§7). The phase BM batch, 18 runs
pre-registered in `prereg_BM.md` with amendment 1 before any launch, was run by `code/run_BM_chain.sh` and
analysed once by `code/analyze_BM.py`. Amendments 2-5 were each committed before the runs or analysis they
govern. Outcome:

1. Sudoku D36 rerun with gradient accumulation (A2, 3 runs): 31.71 ± 0.63% exact, so the D9-versus-D36 gap
   widens from 26 to 30.73 points; the half batch had favoured the deepest setting.
2. ARC D36 with accumulation (A1, 5 runs) and ARC D9 with per-instance records (C, 5 runs) completed on the
   defective ARC subset and support no claim.
3. The ARC non-recursive control (B, 5 runs) failed on a configuration error (RoPE needs an even per-head
   dimension; 256/12 gives 21) and was not rerun (amendment 5).
4. Amendment 5 (phase BS) fixed the ARC subset (§5.2) and reran the decisive contrast, ARC D9 against D36
   at equal effective batch, 5 seeds each (`code/run_arc_g400.sh`, `code/analyze_BS.py`): 63.72 ± 0.19%
   against 62.52 ± 0.16% token accuracy, Welch p = 6.1e-6, exact permutation p = 0.0079, no overlap
   between seeds. The pre-registration bound the reporting in both directions: had the contrast vanished,
   the ARC claim would have been withdrawn.
