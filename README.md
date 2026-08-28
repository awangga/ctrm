# ctrm — a cross-task regime map for tiny recursive models

Code and analysis for the manuscript *"When recursion depth pays for its energy: a cross-task regime map for tiny recursive models"* (target:
*Sustainable Computing: Informatics and Systems*).

The package maps the **energy–accuracy frontier of recursion depth versus parameter count** for Tiny
Recursive Models on symbolic reasoning, under a fixed compute/energy budget, using
**Joules-to-target-accuracy** as the primary metric on a single consumer GPU (NVIDIA RTX 5060 Ti,
16 GB). Framing is Green AI (efficiency frontier and cross-task regime map), not a forecasting scaling
law. Every empirical number in the manuscript traces to a run artifact (per-step learning
curves, 1 Hz power series, CodeCarbon emissions) archived on Zenodo; the compact subset is in `data/`
here.

**Scale of the study.** 91 runs carry a per-run energy record, totalling **18.6 kWh** of measured GPU
energy. The **49 faithful-recipe runs** that produce every reported result account for **16.1 kWh**; the
remaining 42 are pilot and calibration runs. Regenerate this breakdown with
`python3 code/reconcile_totals.py --data-root data`, which writes
`data/ablation_reports/run_energy_reconciliation.md`.

**Energy cross-validation.** CodeCarbon versus integrated `nvidia-smi` agree to **98.8–99.95% on every
one of the 49 faithful-recipe runs**. The band has two structures worth knowing: agreement depends on the
measurement window (an aborted 31 s pilot run gives the single 95.9% outlier), and it depends on the
driver stack (the last six runs, executed after a driver upgrade on the same card, sit systematically at
98.82–98.88% against 99.36–99.95% before it). Scope matters: the earlier pilot runs, which contribute no
reported number, fall as low as 95.9%. Per-run agreement for both regimes is in the reconciliation table.


> **Where the data lives.** This repository carries the code, protocol, and the compact run artifacts
> (summary CSVs, 1 Hz power series, CodeCarbon emissions, consolidated ablation reports). The
> **complete archive**, including the per-step learning curves (`progress_*.jsonl`, 649 MB) and raw
> training logs, is deposited on Zenodo under the concept DOI
> **[10.5281/zenodo.21181342](https://doi.org/10.5281/zenodo.21181342)**, which always resolves to the
> latest version. Cite the Zenodo DOI, not this repository, when referring to the data.

## Contents
```
LICENSE                 MIT (our code); third-party works keep their own
EXPERIMENT_LOG.md       append-only chronological journal of every phase
THIRD_PARTY_LICENSES.md Licences of the vendored third-party works
requirements.txt        Python dependencies (torch cu128 for Blackwell GPUs)
PROTOCOL.md             Full experimental + energy-measurement protocol
vendor/
  TinyRecursiveModels/  upstream TRM source, pinned at c011037, PRISTINE (MIT, Samsung)
  patches/              our 2 changes to it, kept separate (progress logging + AdamW shim)
  arc-agi-1-raw/        raw ARC-AGI-1 tasks (Apache-2.0, fchollet/ARC-AGI @ 3990304)
code/               run_recipe.py (faithful-recipe runner), stats_table.py (Welch/ANOVA),
                    microbench.py, reconcile_totals.py, energy sampler, analysis/plot
                    scripts, optimizer shim, DOI resolvers
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
  Sudoku exact accuracy reaches 36–62%; Maze token accuracy 83–85%. **These are the manuscript results.**
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
- **Maze-Hard, depth axis (h256, FIVE seeds), token accuracy** (exact = 0, uninformative on Maze):
  D9 = 83.17 ± 0.53, D18 = 83.58 ± 0.51, **D36 = 84.91 ± 0.58 %**. The deepest setting is
  **significantly the best**: Welch D36 vs D9 t = 4.96, p = 0.0011; D36 vs D18 t = 3.88, p = 0.0049,
  both surviving Bonferroni; ANOVA F(2,12) = 14.23. D18 vs D9 is not significant (p = 0.25). The grid was
  extended from three to five seeds because at n = 3 the gap (1.56 pt, p = 0.044) missed the corrected
  threshold while its effect size made that null untrustworthy; reporting the n = 3 null would have been
  an artefact of stopping early. The margin is small (1.74 pt) and costs 3% more energy, but no D9 seed
  reaches 83.85%, below the D36 mean, so shallow cannot get there by running longer.
- **ARC-AGI-1, depth axis (h256, 5 seeds), token accuracy** (exact = 0): D9 = 31.2 ± 3.3,
  D18 = 30.7 ± 3.6, D36 = 25.8 ± 0.6 %. Deepest is worst (D9 ≈ D18 > D36). Run at 3 seeds first
  (D9 vs D36 p = 0.063), then **pre-registered** to 5 seeds because the effect size was large; see
  EXPERIMENT_LOG phase AW, committed before the runs. At n = 5 the omnibus test is significant
  (ANOVA F(2,12) = 5.52, p = 0.020) but **the decisive pairwise contrast misses Bonferroni**
  (D9 vs D36 p = 0.020 against α = 0.0167, d = 2.29; D9 vs D18 p = 0.82, genuinely flat). Built from ARC-AGI-1 training+evaluation (no ConceptARC),
  test truncated to 512; absolute scores are far below the multi-GPU state of the art, so we compare the
  depth ordering at matched compute, not headline scores.
- **Cross-task verdict.** The depth effect changes **sign**, not merely magnitude, across tasks. Where
  the metric still separates configurations (Sudoku exact, 38 points of headroom left), depth is expensive and
  shallow recursion wins decisively. Where the metric has saturated (Maze token, everything within two
  points near 85%), the deepest setting wins significantly, by 1.74 points for 3% more energy. On ARC the omnibus effect is
  significant at five seeds but the decisive pairwise contrast fails the same Bonferroni rule applied on
  the other two axes, so it is reported as **unresolved**: not a third regime, and not evidence of no
  effect either. The contribution is therefore a cross-task
  energy–accuracy **regime map** whose organising condition is where a task sits on its own metric, not a
  universal law about recursion.

## Reproducing
1. Build the environment: `code/rebuild_env.sh` (copies the vendored TRM source from `vendor/`, applies
   the two patches in `vendor/patches/`, creates a `cu128` PyTorch venv, and builds the augmented
   Sudoku/Maze datasets with the test split truncated to 512). Install `requirements.txt`. Run it from
   the package root, or set `VENDOR=/path/to/vendor`.
2. Faithful-recipe runs: `run_recipe.py` reads env vars `HIDDEN/DEPTH/BATCH/STEPS/SEED/DATA/OUT_DIR/
   NEVAL/EMA` and emits, per run, `progress_<tag>.jsonl` (train loss/step + eval/checkpoint),
   `pw_<tag>.csv` (1 Hz power), `emissions_<tag>.csv` (CodeCarbon), and a self-describing summary row.
3. Significance: `stats_table.py` reproduces every Welch t / one-way ANOVA / Bonferroni number in the
   paper from the per-task summary CSVs, and writes `stats_table_contrasts.csv`,
   `stats_table_anova.csv` and the LaTeX table. (`sig_test.py` is the older pilot-only script: it reads
   `budget_out/` and does not reproduce the reported figures.)
4. Run and energy totals: `reconcile_totals.py --data-root data` regenerates the 49/91-run and
   16.1/18.6 kWh aggregates and the per-run cross-validation ranges.
5. Calibration and pilot: see `PROTOCOL.md` §8 and §11.

## Known limitations recorded in the data
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
