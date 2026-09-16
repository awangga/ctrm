# Reproducibility package: the energy cost of recursion depth

Reproducibility artifact for the manuscript *"The energy cost of recursion depth: more joules, no extra accuracy at a fixed budget"* (target:
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

**Scale of the study.** 91 runs carry a per-run energy record, totalling **18.6 kWh** of measured GPU
energy. The **49 faithful-recipe runs** that produce every reported result account for **16.1 kWh**; the
remaining 42 are pilot and calibration runs. Regenerate this breakdown with
`python3 code/reconcile_totals.py --data-root data`, which writes
`data/ablation_reports/run_energy_reconciliation.md`.

**Energy cross-validation.** CodeCarbon versus integrated `nvidia-smi` agree to **98.8–99.95% on every
one of the 49 faithful-recipe runs**. The band has two structures worth knowing: agreement depends on the
measurement window (the only readings below 98.68% come from two aborted 31 s pilot runs, at 98.6% and
95.9%), and it depends on the driver stack (the last twelve runs, the Maze-Hard and ARC-AGI-1 seed
extensions executed after a driver upgrade on the same card, sit systematically at 98.82-98.93% against
99.11-99.95% for the 37 runs before them). Scope matters: those earlier pilot runs contribute no reported
number. Per-run agreement is the `gross_agree_pct` column of each summary CSV; the reconciliation table
gives its range per regime.

## Contents
```
.zenodo.json            Zenodo deposition metadata
LICENSE                 MIT (our code)
THIRD_PARTY_LICENSES.md Licences of the vendored third-party works
requirements.txt        Python dependencies (torch cu128 for Blackwell GPUs)
PROTOCOL.md             Full experimental + energy-measurement protocol
PREREGISTRATION.md      Pre-registration records: what was fixed before each added batch was launched
prereg_BM.md            Pre-registration of the 18-run phase BM batch, with amendment 1; a verbatim
                        journal entry, in Indonesian. That batch was still running when this version
                        was assembled and no result from it is reported here
vendor/
  TinyRecursiveModels/  upstream TRM source, pinned at c011037, PRISTINE (MIT, Samsung)
  patches/              the THREE changes we make to it, kept separate from the upstream tree:
    0001-emit-per-step-progress-and-eval-metrics.patch   per-step progress log + eval metrics
    adam_atan2.py                                        AdamW shim (compiled adam-atan2 fails on sm_120)
    0003-accumulation-and-per-instance-eval.py           gradient accumulation (TRM_ACCUM) and
                                                         per-instance eval logging (TRM_EVAL_PREDS),
                                                         both off by default; no released run used them
  arc-agi-1-raw/        raw ARC-AGI-1 tasks (Apache-2.0, fchollet/ARC-AGI @ 3990304)
code/
  -- runners --
  run_recipe.py            faithful-recipe runner; every reported run came from it
  run_BM_chain.sh          the phase BM run chain (18 pre-registered runs, see prereg_BM.md)
  run_arcdepth_s34.sh      the pre-registered ARC seed 3/4 chain (phase AW)
  run_accum_check.sh       equivalence check for the accumulation patch: ARC D9 at batch 48 x accum 1
                           against batch 24 x accum 2, whose train-loss curves must coincide
  run_frontier.sh  run_isoflop.py  run_budget.py  run_scale_sweep.py  run_converge.py
  run_maze_sweep.py  run_replicate.py  run_xval_seeds.py    pilot/calibration sweeps
  -- analysis --
  stats_table.py           Welch t-tests, one-way ANOVA, Bonferroni from the summary CSVs
  sig_test.py              single-contrast significance helper
  trivial_baselines.py     majority-class and copy-input baselines on the 512-instance subset
  add_best_token.py        derives best_token_pct from the per-step logs (best-checkpoint rule)
  estimator_sensitivity.py depth contrast under four checkpoint rules + exact permutation test
  validate_costmodel.py    prediction error of the microbenchmark cost model against the real runs
  analyze_BM.py            the pre-specified phase BM analysis, written before any result existed
  reconcile_totals.py      run counts, energy totals and cross-validation ranges from the CSVs
  analyze_crosstask.py  analyze_frontier.py  consolidate_ablation.py  energy_xval_report.py
  fit_extrapolation.py  joules_to_target.py                 report and ablation-table builders
  -- figures --
  make_manuscript_figures.py  regenerates every manuscript figure from the summary CSVs
  plot_frontier.py  plot_isoflop.py  plot_budget.py          pilot-regime plots
  -- environment, calibration, patches --
  rebuild_env.sh           rebuilds the TRM tree + venv from vendor/ and applies patches 1 and 2
  microbench.py  calibrate_pipeline.sh                       single-GPU calibration
  make_small_eval.py       builds the fixed 512-instance evaluation subset
  adam_atan2_fallback.py   the AdamW shim as installed by rebuild_env.sh
  patch_pretrain_print_metrics.py  pilot-path equivalent of patch 0001
  patch_pretrain_accum_preds.py    third TRM patch: gradient accumulation (TRM_ACCUM) and
                           per-instance evaluation logging (TRM_EVAL_PREDS), both off by default
  resolve_doi.py  resolve_arxiv.py   CrossRef / DataCite reference resolvers
data/
  microbench_results.csv   calibration (params/throughput/energy per step), RTX 5060 Ti, 14 configs

  -- faithful-recipe regime (the manuscript results, 49 runs) --
  recipe_out/              Sudoku-Extreme: depth axis h512, width axis h256/h512/h768,
                           non-recursive transformer baseline (all 3 seeds)
  maze_depth_out/          Maze-Hard depth grid (h256, 5 seeds)
  arc_depth_out/           ARC-AGI-1 depth grid (h256, 5 seeds)
  maze_real_out/           Maze faithful-recipe probe

  -- pilot / calibration regime (42 runs, feasibility record only) --
  frontier_out/  isoflop_out/  budget_out/  scale_out/  converge_out/
  maze_out/      arc_smoke_out/

  ablation_reports/        analysis reports, consolidated ablation table, learning curves,
                           per-seed accuracies, run/energy reconciliation
```

## Self-contained
This archive is the complete and only distribution of the study's code, data, and protocol. Everything
needed to audit or rerun the work is inside it: no companion repository has to be fetched, and no link
outside this record has to resolve. The Tiny Recursive Models codebase the runner builds on is vendored
at a pinned commit under `vendor/`, together with the raw ARC-AGI-1 tasks, so `code/rebuild_env.sh`
performs no network clone. Licences for the vendored works are listed in `THIRD_PARTY_LICENSES.md`.

The one thing still fetched at build time is the Sudoku-Extreme and Maze-Hard benchmark data, which
TRM's own dataset builders pull from Hugging Face. We do not redistribute those: neither declares a
licence, and Sudoku-Extreme alone is ~762 MB. See `vendor/README.md`.

## Two regimes (read this before using the data)
- **Faithful-recipe regime** (`recipe_out/`, `maze_depth_out/`, `arc_depth_out/`, `maze_real_out/`):
  the faithful TRM recipe (hidden >= 256, EMA, ~1M augmented examples, thousands of optimizer steps).
  Sudoku exact accuracy reaches 36–62%; Maze token accuracy 86–87%, ARC 31–36% (best checkpoint). **These are the manuscript results.**
- **Pilot/toy regime** (the remaining `*_out/` folders): early under-trained runs (small width, no EMA)
  with exact accuracy 0–15%. Retained only as a pilot/feasibility record; do **not** read final claims
  from these. Superseded by the faithful-recipe regime.

## Main results (faithful-recipe regime, iso-compute: params x D_eff x batch x steps constant)
All uncertainties are the **sample** standard deviation (ddof=1) over three seeds.

- **Sudoku-Extreme, depth axis (h512, 3 seeds).** Exact accuracy is monotone in depth and shallow wins:
  D_eff 9 = **62.4 ± 0.3 %** > 18 = 50.1 ± 1.7 % > 36 = 36.3 ± 0.8 % (Welch t ≈ 12.5 / 12.9; ordering
  holds in every seed). Joules-to-exact-50%, computed per seed then averaged over the seeds that reach
  the target: D9 = 268 +/- 19 Wh (3/3 seeds), D18 = 329 +/- 17 Wh (2/3), D36 never reaches it (0/3).
  At matched accuracy the gap is larger still: whatever D36 attains with its entire 340 Wh, D9 attains on
  161 +/- 1 Wh in every seed, a 53% energy saving (109 vs 230 g CO2e).
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
- **ARC-AGI-1, depth axis (h256, FIVE seeds), token accuracy, best checkpoint** (exact = 0):
  **D9 = 36.27 ± 3.18**, D18 = 33.34 ± 3.37, D36 = 30.59 ± 1.78 %. The shallow-versus-deepest contrast
  survives Bonferroni (Welch t = 3.48, p = 0.012, d = 2.20; threshold 0.0167); adjacent contrasts do not
  (p = 0.20, 0.16); ANOVA F(2,12) = 4.90, p = 0.028. The grid was pre-registered and extended from three
  seeds (p = 0.14 at n = 3) to five (EXPERIMENT_LOG phase AW). D9 matches D36's best accuracy on
  155 ± 82 Wh against D36's 278 Wh (every seed reaches it; four of five below D36's budget, 81-160 Wh, one needs 291 Wh; 44% saving on average).
- **Maze-Hard, depth axis (h256, FIVE seeds), token accuracy, best checkpoint** (exact = 0):
  D9 = 86.66 ± 0.18, D18 = 86.76 ± 0.09, D36 = 86.54 ± 0.29 %. **Null measurement, not a null effect.**
  On the same 512-instance subset, copying the input already scores **87.51%** token accuracy (path cells
  are 12.5% of a 30x30 maze), so every Maze run sits BELOW the trivial baseline and the metric does not
  measure task learning at this budget. No contrast approaches significance (p = 0.42, 0.16, 0.31; ANOVA
  F(2,12) = 1.59, p = 0.24). What the grid does establish: the deepest setting spends 3% more energy than
  the shallowest and 7% more than the mid-depth setting (324 vs 314 vs 303 Wh) for nothing measurable.
- **Trivial baselines** (`code/trivial_baselines.py`, same per-sequence metric as the training loop):
  Sudoku exact 0.00% (majority and copy-input), ARC token 25.00% / 25.00%, Maze token 50.03% / **87.51%**.
- **What the three tasks together support.** Added recursion depth never bought accuracy on any task
  measured. Where accuracy could be priced (Sudoku exact, ARC token above its trivial baseline) shallow
  recursion wins by 26 and 5.7 points and reaches a given accuracy on 161 vs 340 Wh and 155 vs 278 Wh.
  Where it could not be priced (Maze token, below its trivial baseline) only the energy side is reported:
  the deepest setting costs 3% more for nothing measurable. Two measured points and one null measurement
  are not a cross-task generalisation, and the manuscript draws none from them.
- **Checkpoint-estimator sensitivity** (`code/estimator_sensitivity.py`): the ARC D9-vs-D36 gap is 5.7,
  5.4, 5.5 and 3.3 points under the best, final, last-five-mean and median rules, with exact permutation
  p between 0.008 and 0.016 (resolution floor 0.008 at n = 5).
- **Cost-model validation** (`code/validate_costmodel.py`): the microbenchmark fit predicts realised
  per-step energy within +15..+24% on Sudoku but is off by more than 90% on Maze and ARC (sequence length
  900 vs the microbenchmark's 81); b rises to 0.94 on the sub-50 configurations; two-exponent fit
  J ~ P^0.82 D^0.92. Use it only within the regime it was measured in.

## Phase BM: pre-registered runs in progress (no results in this version)
Two simulated peer reviews asked for three things that existing artifacts cannot answer, so an 18-run
batch was pre-registered in `prereg_BM.md` (with amendment 1, both written and committed before any run
was launched) and is running at the time this version was assembled:

- the deepest cell of the ARC-AGI-1 and Sudoku-Extreme grids rerun with **gradient accumulation**, so its
  effective batch matches its neighbours at the same number of epochs (identical compute, reallocated);
- a **non-recursive control on ARC-AGI-1**, matched on epochs with the ARC `D9` cell;
- **per-instance evaluation logging**, which splits the 512-instance subset into a selection half and a
  reporting half and so bounds best-checkpoint selection bias.

`code/run_BM_chain.sh` launches the chain and `code/analyze_BM.py` holds the analysis, written and
committed before any outcome was visible; it runs once, after all 18 runs finish. **No result from this
batch is reported anywhere in this package or in the manuscript version it accompanies.** The
pre-registration binds the reporting: if the ARC gap loses significance once the batch is restored, the
`p = 0.012` claim is withdrawn.

Operational note recorded with the batch: `GROUPS` is a special bash variable and is not exported, so the
runner must be given it as `env GROUPS=... cmd`. An `export GROUPS=3080` is silently ignored and the
runner falls back to its default of 1000, which changes the epoch count.

## Known limitations recorded in the data
- The h768 x D_eff=36 microbenchmark cell ran out of memory on the 16 GB card (Table 1 of the manuscript);
  it has no row in `microbench_results.csv`.
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
