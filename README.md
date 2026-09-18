# Reproducibility package: the energy cost of recursion depth

Reproducibility artifact for the manuscript *"The energy cost of recursion depth: half the joules for the same accuracy"* (target:
*Sustainable Computing: Informatics and Systems*).

The package prices **recursion depth in joules** for Tiny Recursive Models on symbolic reasoning, under
a fixed compute budget, using **Joules-to-target-accuracy** as the primary metric on a single consumer
GPU (NVIDIA RTX 5060 Ti, 16 GB). Framing is Green AI (what a design knob costs in joules), not a
forecasting scaling law. Every empirical number in the manuscript traces to a run artifact in `data/`
(per-step learning curves, 1 Hz power series, CodeCarbon emissions).

Two things are reported: the measured energy price of depth on the two tasks whose accuracy metric can
be priced at all (Sudoku-Extreme, ARC-AGI-1), and **three measurement pitfalls** that generalise beyond
recursion:

1. **Evaluation cost grows with depth** and must be pinned, or the comparison silently charges deeper
   configurations for their own evaluation.
2. **A reduced-scale screen ranked the eventual winner last**, so a cheap screen is not a substitute for
   the faithful recipe.
3. **A proxy metric must be scored against a trivial baseline** before any effect is read from it. On
   Maze-Hard this check voids the metric (see below).

Three tasks were measured, and nothing about depth is generalised across them: two carry a priced
accuracy axis, the third carries only an energy comparison.

**Scale of the study.** 116 runs carry a per-run energy record, totalling **25.3 kWh** of measured GPU
energy. The **72 faithful-recipe runs** account for **22.7 kWh** (about 15.4 kg CO2e at the 675.9 g/kWh
grid intensity CodeCarbon records); the remaining 44 are pilot, calibration and patch-verification runs.
Of the 72, **25 are earlier ARC-AGI-1 runs scored on a defective evaluation subset** (see "What changed in
v6"): their energy is counted, because it was spent, but their accuracies support no claim. Regenerate this breakdown with
`python3 code/reconcile_totals.py --data-root data`, which writes
`data/ablation_reports/run_energy_reconciliation.md`.

**Energy cross-validation.** CodeCarbon versus integrated `nvidia-smi` agree to **98.82–99.95% on every
one of the 72 faithful-recipe runs**. Read that agreement narrowly. Both readers poll the **same NVML
sensor**, so they are not two independent instruments and the number is not a sensor validation. What the
agreement bounds is the 1 Hz integration error and the bookkeeping between the two paths, NOT the accuracy
of the sensor itself: absolute energies carry the sensor's own uncertainty, and only the ordering of
configurations is protected, by holding the iso-environment fixed.

The band has two structures worth knowing: agreement depends on the measurement window (the only readings below 98.68% come from two aborted 31 s pilot runs, at 98.6% and
95.9%), and it depends on the driver stack (the last 35 runs, executed after an upgrade from driver 595.71.05 to
595.84 on the same card, namely 12 Maze-Hard and ARC-AGI-1 seed extensions and 23 runs of phases BM and
BS, sit systematically at 98.82-98.97% against 99.11-99.95% for the 37 runs before them). Scope matters: those earlier pilot runs contribute no reported
number. Per-run agreement is the `gross_agree_pct` column of each summary CSV; the reconciliation table
gives its range per regime.

## What changed in v6 (relative to v5)
- **ARC-AGI-1 evaluation subset replaced.** The subset used by every earlier ARC run,
  `arc1-aug1k-e512`, turned out to hold augmentations of a **single** ARC task: the ARC test split is
  augmented to about 1001 examples per task, and `make_small_eval.py` took the first 512 rows, all of
  which fall in the first task (512 examples, only 72 distinct output grids). We found this in an internal
  audit. It is replaced by `arc1-aug1k-g400`, built by `code/make_group_eval.py`: the original,
  unaugmented test examples of all **400** ARC-AGI-1 evaluation tasks (**419** examples). The training
  split is unchanged. Sudoku-Extreme and Maze-Hard are not augmented at test time and are unaffected.
- **ARC results rerun on the valid subset** (pre-registered as amendment 5 in `prereg_BM.md`): D9 and D36,
  five seeds each, D36 at the same effective batch as D9 through gradient accumulation. Every v5 ARC
  number (36.3/33.3/30.6%, 5.7 points, p = 0.012, 155 vs 278 Wh, trivial baseline 25.00%) is withdrawn.
- **Phase BM batch finished** and is reported: the Sudoku deepest cell rerun at its neighbours' batch
  size (the depth gap widens, so the 26-point headline is conservative). The ARC non-recursive control
  (batch B) failed on a configuration error and was not rerun; the non-recursive control therefore
  covers Sudoku only.
- **Totals** updated from 91 runs / 18.6 kWh (49 faithful, 16.1 kWh) to 116 runs / 25.3 kWh
  (72 faithful, 22.7 kWh).

## Contents
```
README.md               This file: what the package contains and how to read the results
LICENSE                 MIT (our code)
THIRD_PARTY_LICENSES.md Licences of the vendored third-party works
.gitignore              Standard Python ignore rules (build artefacts, caches, virtualenvs)
requirements.txt        Python dependencies (torch cu128 for Blackwell GPUs)
PROTOCOL.md             Full experimental + energy-measurement protocol
PREREGISTRATION.md      Pre-registration records: what was fixed before each added batch was launched
prereg_BM.md            Pre-registration of the 18-run phase BM batch with amendments 1-5 (amendment 5
                        fixes the ARC evaluation subset and locks the 10-run rerun); a verbatim journal
                        entry, in Indonesian, every part committed before the runs it governs
EXPERIMENT_LOG.md       Chronological working journal, every phase from A onward, in Indonesian.
                        APPEND-ONLY: old entries keep framings and numbers that were later withdrawn
                        (the "regime map" framing, an earlier manuscript title, the Maze-Hard accuracy
                        claim retracted after the trivial-baseline check, the ARC results scored on the
                        defective single-task subset). Corrections are appended as new entries and
                        marked, never applied by rewriting an old one. It records how the study
                        developed; for what the study CLAIMS, the manuscript is the reference
vendor/
  README.md             what is vendored here, and the benchmark data that is not
  TinyRecursiveModels/  upstream TRM source, pinned at c011037, PRISTINE (MIT, Samsung)
  patches/              the THREE changes we make to it, kept separate from the upstream tree:
    0001-emit-per-step-progress-and-eval-metrics.patch   per-step progress log + eval metrics
    adam_atan2.py                                        AdamW shim (compiled adam-atan2 fails on sm_120)
    0003-accumulation-and-per-instance-eval.py           gradient accumulation (TRM_ACCUM) and
                                                         per-instance eval logging (TRM_EVAL_PREDS),
                                                         both off by default; used only by the phase BM
                                                         and BS runs (see the data folders below)
  arc-agi-1-raw/        raw ARC-AGI-1 tasks (Apache-2.0, fchollet/ARC-AGI @ 3990304)
code/
  -- runners --
  run_recipe.py            faithful-recipe runner; every reported run came from it
  run_BM_chain.sh          the phase BM run chain (18 pre-registered runs, see prereg_BM.md)
  run_arc_g400.sh          the phase BS chain: ARC D9 and D36 (accumulated) on the valid subset, 10 runs
  run_arcdepth_s34.sh      the pre-registered ARC seed 3/4 chain (phase AW)
  watch_arcdepth_s34.sh    watcher that reports progress of that chain while it runs
  run_accum_check.sh       equivalence check for the accumulation patch: ARC D9 at batch 48 x accum 1
                           against batch 24 x accum 2, whose train-loss curves must coincide
  run_frontier.sh  run_isoflop.py  run_budget.py  run_scale_sweep.py  run_converge.py
  run_maze_sweep.py  run_replicate.py  run_xval_seeds.py    pilot/calibration sweeps
  -- analysis --
  stats_table.py           Welch t-tests, one-way ANOVA, Bonferroni from the summary CSVs
  sig_test.py              single-contrast significance helper
  trivial_baselines.py     majority-class and copy-input baselines on each task's evaluation subset
  add_best_token.py        derives best_token_pct from the per-step logs (best-checkpoint rule)
  estimator_sensitivity.py depth contrast under four checkpoint rules + exact permutation test
                           (its ARC part still reads the old-subset grid and supports no claim)
  validate_costmodel.py    prediction error of the microbenchmark cost model against the real runs
  analyze_BM.py            the pre-specified phase BM analysis, written before any result existed
  analyze_BS.py            the pre-specified analysis of amendment 5 (ARC on the valid subset), committed
                           before any model accuracy on that subset was read
  reconcile_totals.py      run counts, energy totals and cross-validation ranges from the CSVs
  analyze_crosstask.py  analyze_frontier.py  consolidate_ablation.py  energy_xval_report.py
  fit_extrapolation.py  joules_to_target.py  joules_to_target_faithful.py
                                               report and ablation-table builders
  -- figures --
  make_manuscript_figures.py  regenerates every manuscript figure from the summary CSVs
  plot_frontier.py  plot_isoflop.py  plot_budget.py          pilot-regime plots
  -- environment, calibration, patches --
  rebuild_env.sh           rebuilds the TRM tree + venv from vendor/ and applies patches 1 and 2
  microbench.py  calibrate_pipeline.sh                       single-GPU calibration
  make_small_eval.py       builds the fixed 512-instance evaluation subset by taking the first N test
                           rows; valid ONLY for test splits that are not augmented (Sudoku, Maze)
  make_group_eval.py       builds the ARC subset: the first (unaugmented) example of every task group
  adam_atan2_fallback.py   the AdamW shim as installed by rebuild_env.sh
  patch_pretrain_print_metrics.py  pilot-path equivalent of patch 0001
  patch_pretrain_accum_preds.py    third TRM patch: gradient accumulation (TRM_ACCUM) and
                           per-instance evaluation logging (TRM_EVAL_PREDS), both off by default
  resolve_doi.py  resolve_arxiv.py   CrossRef / DataCite reference resolvers
data/
  microbench_results.csv   calibration (params/throughput/energy per step), RTX 5060 Ti, 14 configs

  -- faithful-recipe regime (72 runs) --
  recipe_out/              Sudoku-Extreme: depth axis h512, width axis h256/h512/h768,
                           non-recursive transformer baseline (all 3 seeds)
  sudoku_d36_accum_out/    Sudoku D36 rerun at effective batch 192 (96 x 2 accumulation), 3 seeds (BM A2)
  maze_depth_out/          Maze-Hard depth grid (h256, 5 seeds)
  maze_real_out/           Maze faithful-recipe probe
  arc_d9_g400_out/         ARC-AGI-1 D9 on the valid subset arc1-aug1k-g400 (h256, 5 seeds, phase BS)
  arc_d36_g400_out/        ARC-AGI-1 D36 on the valid subset, effective batch 48 (24 x 2), 5 seeds (BS)
                           -> every ARC result in the manuscript comes from these two folders
  arc_depth_out/           ARC-AGI-1 depth grid D9/D18/D36 (5 seeds) on the DEFECTIVE subset
  arc_d36_accum_out/       ARC D36 with accumulation (BM A1), 5 seeds, DEFECTIVE subset
  arc_d9_preds_out/        ARC D9 with per-instance eval records (BM C), 5 seeds, DEFECTIVE subset
                           -> these three are kept as a record and count in the energy totals;
                              their accuracies support no claim
  (preds/ subfolders hold per-instance evaluation records where SAVE_PREDS was set)

  -- patch verification --
  accum_check_out/         ARC D9 at batch 48 x accum 1 against 24 x accum 2 (accumulation check)

  -- pilot / calibration regime (42 runs, feasibility record only) --
  frontier_out/  isoflop_out/  budget_out/  scale_out/  converge_out/
  maze_out/      arc_smoke_out/

  ablation_reports/        analysis reports, consolidated ablation table, learning curves,
                           per-seed accuracies, run/energy reconciliation
```

## Self-contained
This archive is a complete, self-contained distribution of the study's code, data, and protocol.
Everything needed to audit or rerun the work is inside it: no companion repository has to be fetched, and
no link outside this record has to resolve. A public code mirror of the same package additionally carries
the append-only working journal `EXPERIMENT_LOG.md` (see `PREREGISTRATION.md` for how to read it);
nothing in this archive depends on that mirror. The Tiny Recursive Models codebase the runner builds on is vendored
at a pinned commit under `vendor/`, together with the raw ARC-AGI-1 tasks, so `code/rebuild_env.sh`
performs no network clone. Licences for the vendored works are listed in `THIRD_PARTY_LICENSES.md`.

The one thing still fetched at build time is the Sudoku-Extreme and Maze-Hard benchmark data, which
TRM's own dataset builders pull from Hugging Face. We do not redistribute those: neither declares a
licence, and Sudoku-Extreme alone is ~762 MB. See `vendor/README.md`.

## Two regimes (read this before using the data)
- **Faithful-recipe regime** (`recipe_out/`, `sudoku_d36_accum_out/`, `maze_depth_out/`, `maze_real_out/`,
  `arc_d9_g400_out/`, `arc_d36_g400_out/`, plus the three old-subset ARC folders kept as a record):
  the faithful TRM recipe (hidden >= 256, EMA, ~1M augmented examples, thousands of optimizer steps).
  Sudoku exact accuracy reaches 31.7–62.4%; Maze token accuracy 86–87%; ARC token accuracy 62.5–63.7% on the
  valid subset (best checkpoint). **These are the manuscript results**, except the ARC folders on the
  defective subset, which contribute energy only.
- **Pilot/toy regime** (the remaining `*_out/` folders): early under-trained runs (small width, no EMA)
  with exact accuracy 0–15%. Retained only as a pilot/feasibility record; do **not** read final claims
  from these. Superseded by the faithful-recipe regime.

## Main results (faithful-recipe regime, iso-compute: params x D_eff x batch x steps constant)
All uncertainties are the **sample** standard deviation (ddof=1) over the seeds of the grid in question:
three seeds on Sudoku-Extreme (depth axis, width axis, non-recursive baseline, batch-restored D36), five
seeds on Maze-Hard and on ARC-AGI-1.

- **Sudoku-Extreme, depth axis (h512, 3 seeds).** Exact accuracy is monotone in depth and shallow wins:
  D_eff 9 = **62.4 ± 0.3 %** > 18 = 50.1 ± 1.7 % > 36 = 36.3 ± 0.8 % (Welch t ≈ 12.5 / 12.9; ordering
  holds in every seed). Joules-to-exact-50%, computed per seed then averaged over the seeds that reach
  the target: D9 = 268 +/- 19 Wh (3/3 seeds), D18 = 329 +/- 17 Wh (2/3), D36 never reaches it (0/3).
  At matched accuracy the gap is larger still: whatever D36 attains with its entire 340 Wh, D9 attains on
  161 +/- 1 Wh in every seed, a 53% energy saving (109 vs 230 g CO2e).
- **Sudoku-Extreme, batch confound tested (phase BM, batch A2, 3 seeds).** The deepest cell originally ran
  at half its neighbours' batch (96 vs 192) because the 16 GB card cannot hold the full batch. Rerun with
  gradient accumulation (micro-batch 96 x 2 = effective batch 192), D36 reaches **31.71 ± 0.63 %**, 4.56
  points below the original D36, so the D9-versus-D36 gap widens to **30.73 points**. The smaller batch had
  favoured the deepest setting: the 26-point headline is conservative, and with batch matched the cost of
  depth is about 31 points.
- **Sudoku-Extreme, width axis (D18, 3 seeds).** Interior optimum: h256 = 38.0 ± 1.1 -> **h512 = 50.1 ± 1.7**
  -> h768 = 43.0 ± 1.0 %, significant (Welch h512 vs h256 p = 0.001; h512 vs h768 p = 0.007). The frontier
  is anisotropic: depth monotone-decreasing, width single-peaked.
- **Non-recursive baseline (Sudoku, matched energy, 3 seeds).** An 8-layer non-recursive transformer at
  h512 reaches 49.7 ± 2.1 % exact at 384 Wh. The shallowest TRM beats it significantly (D9 = 62.4 %,
  Welch p = 0.008, reaching 50 % in every seed at 268 +/- 19 Wh against the baseline's 353 +/- 43 Wh in
  two seeds of three); mid-depth TRM ties it (D18 = 50.1 %, p = 0.82) and
  deep TRM is significantly worse (D36 = 36.3 %, p = 0.004). Recursion earns its keep only when shallow;
  deep recursion is significantly worse than no recursion at all.
- **Energy cost model.** From the calibration microbenchmark (direct hardware measurement, regime-independent):
  per-step energy scales as (params x D_eff)^0.84 (R^2 = 0.98, bootstrap 95% CI [0.75, 0.92]).
- **Checkpoint rule (all tasks).** A configuration is scored by its BEST evaluation checkpoint
  (`best_exact_pct` on Sudoku, `best_token_pct` on Maze/ARC), as in Algorithm 1 of the manuscript.
  `best_token_pct` was derived post hoc from `progress_<tag>.jsonl` by `code/add_best_token.py`
  (2026-09-15); `final_token_pct` is kept alongside. The final-checkpoint reading is reported in the
  manuscript where it differs (Maze: 83.2/83.6/84.9 %, deepest degrades least after an early peak).
- **ARC-AGI-1, depth end points on the valid subset (h256, FIVE seeds), token accuracy, best checkpoint**
  (exact at most 0.24%): **D9 = 63.72 ± 0.19 %**, D36 = 62.52 ± 0.16 %, a gap of **+1.20 points**. D36 runs at
  the same effective batch as D9 (48, via 24 x 2 accumulation), so the contrast tests depth alone. Welch
  t = 10.81, df = 7.7, p = 6.1e-6, d = 6.84; exact permutation p = 0.0079 (the floor for five against five):
  the two depths **separate completely**, the worst D9 seed (63.55) above the best D36 seed (62.77). It
  clears the per-axis Bonferroni threshold (0.0167) and the whole-paper threshold (0.05/13 = 0.0038). Both
  arms sit above the copy-input baseline of 60.75% (D9 by 2.97 points, D36 by 1.77), so depth consumes
  about 40% of what the shallow model learns beyond the trivial baseline. Net energy per run: D9
  284.3 ± 0.6 Wh, D36 267.8 ± 0.3 Wh. **D9 reaches D36's mean best accuracy (62.52%) on 98 ± 70 Wh (5 of 5
  seeds) against the 268 Wh D36 spends in full, a 63% saving.** Only D9 and D36 were run on the valid
  subset, so ARC establishes the extreme contrast, not a graded depth axis. Pre-registered as amendment 5
  in `prereg_BM.md`; analysis `code/analyze_BS.py`, run once after all ten runs.
- **Maze-Hard, depth axis (h256, FIVE seeds), token accuracy, best checkpoint** (exact = 0):
  D9 = 86.66 ± 0.18, D18 = 86.76 ± 0.09, D36 = 86.54 ± 0.29 %. **Null measurement, not a null effect.**
  On the same 512-instance subset, copying the input already scores **87.51%** token accuracy (path cells
  are 12.5% of a 30x30 maze), so every Maze run sits BELOW the trivial baseline and the metric does not
  measure task learning at this budget. No contrast approaches significance (p = 0.42, 0.16, 0.31; ANOVA
  F(2,12) = 1.59, p = 0.24). What the grid does establish: the deepest setting spends 3% more energy than
  the shallowest and 7% more than the mid-depth setting (324 vs 314 vs 303 Wh) for nothing measurable.
- **Trivial baselines** (`code/trivial_baselines.py`, same per-sequence metric as the training loop):
  Sudoku exact 0.00% (majority and copy-input), ARC token 41.82% / 60.75% on the valid subset, Maze token
  50.03% / **87.51%**. The ARC figure of 25.00% / 25.00% reported in v5 came from the defective subset and
  is withdrawn.
- **What the three tasks together support.** Added recursion depth never bought accuracy on any task
  measured. Where accuracy could be priced (Sudoku exact, ARC token above its trivial baseline) shallow
  recursion wins by 26 and 1.20 points (about 31 on Sudoku once the batch is matched) and reaches a given
  accuracy on 161 vs 340 Wh and 98 vs 268 Wh.
  Where it could not be priced (Maze token, below its trivial baseline) only the energy side is reported:
  the deepest setting costs 3% more for nothing measurable. Two measured points and one null measurement
  are not a cross-task generalisation, and the manuscript draws none from them.
- **Checkpoint-estimator sensitivity** (`code/estimator_sensitivity.py`): recomputes a depth contrast under
  the best, final, last-five-mean and median rules with an exact permutation test. The ARC figures it gave
  in v5 were computed on the defective subset and are withdrawn; the script's ARC part still reads that
  grid. On the valid subset the ARC depths do not overlap in any seed, which is what protects the ordering
  from the optimistic bias of best-checkpoint selection.
- **Cost-model validation** (`code/validate_costmodel.py`): the microbenchmark fit predicts realised
  per-step energy within +15..+24% on Sudoku but is off by more than 90% on Maze and ARC (sequence length
  900 vs the microbenchmark's 81); b rises to 0.94 on the sub-50 configurations; two-exponent fit
  J ~ P^0.82 D^0.92. Use it only within the regime it was measured in.

## Pre-registered batches BM and BS: what ran and what came of it
Two simulated peer reviews asked for evidence that existing artifacts could not give, so an 18-run batch
was pre-registered in `prereg_BM.md` (amendment 1 written before any run was launched; amendments 2-5 each
committed before the runs or analysis they govern). `code/run_BM_chain.sh` ran it and `code/analyze_BM.py`
holds its pre-specified analysis.

- **A1, ARC D36 with gradient accumulation (5 runs)** and **C, ARC D9 with per-instance evaluation records
  (5 runs)** completed, but on the defective ARC subset. They are kept (`arc_d36_accum_out/`,
  `arc_d9_preds_out/`) as a record and count in the energy totals; their accuracies support no claim.
- **A2, Sudoku D36 with gradient accumulation (3 runs)** completed and is reported above: the depth gap
  widens from 26 to about 31 points.
- **B, non-recursive control on ARC (5 runs)** failed on a configuration error (RoPE needs an even
  per-head dimension; 256/12 gives 21) and was **not rerun**; its folder is not shipped because the runs
  died within seconds and hold no data. The non-recursive control therefore exists on Sudoku only.
- **Amendment 5 (phase BS, 10 runs)** replaced the ARC evaluation subset and reran the decisive ARC
  contrast on it (`code/run_arc_g400.sh`, `code/analyze_BS.py`), with a commitment written in advance to
  withdraw the ARC claim if the contrast vanished. It did not vanish; the result is reported above.

Operational note recorded with the batch: `GROUPS` is a special bash variable and is not exported, so the
runner must be given it as `env GROUPS=... cmd`. An `export GROUPS=3080` is silently ignored and the
runner falls back to its default of 1000, which changes the epoch count.

## Known limitations recorded in the data
- The h768 x D_eff=36 microbenchmark cell ran out of memory on the 16 GB card (Table 1 of the manuscript);
  it has no row in `microbench_results.csv`.
- The first 25 ARC-AGI-1 runs (`arc_depth_out/`, `arc_d36_accum_out/`, `arc_d9_preds_out/`) were scored on
  `arc1-aug1k-e512`, which holds augmentations of a single task. Their energy records are valid; their
  accuracy columns do not measure ARC-AGI-1 and support no claim.
- The non-recursive baseline exists on Sudoku-Extreme only; the ARC counterpart failed and was not rerun.
- Baseline rows in `recipe_out/recipe_summary.csv` carry `D_eff=8` (the runner's DEPTH env variable was
  reused to pass the 8-layer setting); the baseline has no recursion.
- The compiled `adam-atan2` optimizer does not build on sm_120; all runs use the AdamW shim in
  `code/adam_atan2_fallback.py`. Applied identically to every configuration, so comparisons remain matched.
- Nine pilot replication runs (`budget_out/replication_summary.csv`, seeds 0–2) recorded accuracy but
  their power logs were written without a seed suffix and overwrote one another, so per-seed energy is
  not attributable. Those runs are excluded from the energy totals; the exclusion is stated in
  `ablation_reports/run_energy_reconciliation.md`.

## Citing
Cite the manuscript and this package. Use the concept DOI **10.5281/zenodo.21181342**, which always
resolves to the latest version; cite the version DOI shown on the record page instead if you need to
pin an exact snapshot.
Built on the Tiny Recursive Models codebase; please also cite Jolicoeur-Martineau (2025), *Less is More:
Recursive Reasoning with Tiny Networks* (arXiv:2510.04871).

## Funding
Ministry of Higher Education, Science, and Technology of the Republic of Indonesia (Kementerian
Pendidikan Tinggi, Sains, dan Teknologi), *Penelitian Fundamental - Reguler* scheme. Master contract
no. 283/C3/DT.05.00/PL-BARU/2026 (30 January 2026), devolved through LLDIKTI Region IV contract
no. 1650/LL4/PG/2026 (21 April 2026) and institutional contract no. PKS.003/WAREKIII-ULBI/V/2026
(11 May 2026).
