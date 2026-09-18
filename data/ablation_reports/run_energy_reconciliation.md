# Rekonsiliasi run & energi (auto-generated)

Dihasilkan oleh `eksperimen/frontier/reconcile_totals.py` dari summary CSV yang di-commit.
Jangan diedit tangan; jalankan ulang skripnya.

| summary CSV | rezim | isi | n run | net Wh | xval min | xval max |
|---|---|---|---:|---:|---:|---:|
| `arc_d36_accum_out/recipe_summary.csv` | faithful | ARC D36 batch efektif dipulihkan (BM A1) | 5 | 1451.2 | 98.86 | 98.89 |
| `arc_d36_g400_out/recipe_summary.csv` | faithful | ARC D36 accum subset sah 400 task (BS) | 5 | 1339.2 | 98.91 | 98.95 |
| `arc_d9_g400_out/recipe_summary.csv` | faithful | ARC D9 subset sah 400 task (BS) | 5 | 1421.7 | 98.95 | 98.97 |
| `arc_d9_preds_out/recipe_summary.csv` | faithful | ARC D9 dgn prediksi per-instance (BM C) | 5 | 1451.6 | 98.93 | 98.95 |
| `arc_depth_out/recipe_summary.csv` | faithful | ARC-AGI-1 depth grid | 15 | 4237.4 | 98.88 | 99.43 |
| `maze_depth_out/recipe_summary.csv` | faithful | Maze-Hard depth grid | 15 | 4707.2 | 98.82 | 99.95 |
| `maze_real_out/recipe_summary.csv` | faithful | Maze faithful-recipe probe | 1 | 209.4 | 99.66 | 99.66 |
| `recipe_out/recipe_summary.csv` | faithful | Sudoku depth+width+baseline | 18 | 6903.1 | 99.22 | 99.80 |
| `sudoku_d36_accum_out/recipe_summary.csv` | faithful | Sudoku D36 batch efektif dipulihkan (BM A2) | 3 | 1011.8 | 98.88 | 98.89 |
| `accum_check_out/recipe_summary.csv` | verifikasi | verifikasi patch akumulasi gradien | 2 | 41.2 | 98.79 | 98.83 |
| `arc_smoke_out/recipe_summary.csv` | pilot | ARC smoke test | 3 | 42.1 | 99.38 | 99.61 |
| `budget_out/energy_xval.csv` | pilot | budget replication seeds 3-4 | 6 | 408.3 | 98.86 | 99.00 |
| `converge_out/converge_summary.csv` | pilot | convergence probe D36 | 1 | 292.1 | 98.91 | 98.91 |
| `converge_out/converge_summary_h128_d18_aug.csv` | pilot | convergence probe D18 (aug) | 1 | 157.9 | 98.96 | 98.96 |
| `frontier_out/frontier_summary.csv` | pilot | frontier pilot | 3 | 104.3 | - | - |
| `isoflop_out/isoflop_summary.csv` | pilot | iso-FLOP pilot | 9 | 94.5 | - | - |
| `maze_out/maze_summary.csv` | pilot | Maze toy sweep | 9 | 1195.5 | 95.89 | 99.14 |
| `scale_out/scale_summary.csv` | pilot | scale sweep pilot | 10 | 254.2 | 98.68 | 99.23 |

## Agregat

| rezim | n run | energi | cross-val CodeCarbon vs nvidia-smi |
|---|---:|---:|---|
| faithful | 72 | 22732.3 Wh = 22.73 kWh | 98.82–99.95% |
| pilot | 42 | 2548.9 Wh = 2.55 kWh | 95.89–99.61% |
| verifikasi | 2 | 41.2 Wh = 0.04 kWh | 98.79–98.83% |
| **TOTAL** | **116** | **25322.5 Wh = 25.32 kWh** | 95.89–99.95% |

## Jejak karbon (CO2e)

Intensitas grid dari CodeCarbon: **675.93 g CO2e/kWh** (Indonesia), konstan di 102 run (rentang 0.0000 g/kWh). Faktor yang sama diterapkan ke energi **net GPU** (idle-corrected) agar konsisten dengan metrik utama paper.

| rezim | energi net | CO2e |
|---|---:|---:|
| faithful | 22.73 kWh | 15.37 kg |
| pilot | 2.55 kWh | 1.72 kg |
| **TOTAL** | **25.32 kWh** | **17.12 kg** |

Sebagai pembanding, angka CO2e mentah CodeCarbon (CPU+GPU+RAM, tanpa koreksi idle) lebih tinggi; nilai di atas sengaja memakai basis net-GPU yang sama dengan seluruh metrik energi paper.


## Angka yang dipakai manuscript

- Rezim faithful-recipe (semua hasil yang dilaporkan): **72 run, 22.7 kWh**, cross-val **98.82–99.95%** pada setiap run.
- Seluruh studi (faithful + pilot): **116 run, 25.3 kWh**.
- Cross-val pilot turun sampai **95.89%**, jadi klaim "99.1–99.95% pada setiap run" HANYA sah bila di-scope ke rezim faithful-recipe.

## Dikecualikan dari total energi

budget_out/replication_summary.csv memuat akurasi 3 seed (0,1,2) x 3 depth = 9 run pilot, tetapi power log-nya ditulis ke nama file tanpa sufiks seed (pw_h128_d{9,18,36}.csv) sehingga saling menimpa dan energi per-seed tidak dapat diatribusikan. Sembilan run itu dikecualikan dari total energi; hanya seed 3-4 (energy_xval.csv) yang punya cross-val per-run.

