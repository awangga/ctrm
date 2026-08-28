> **[REGIME: TOY/PILOT — SUPERSEDED]** Report ini memakai data rezim-toy lama (h128/no-EMA, exact 0-15%). Hasil final manuscript memakai rezim akurasi nyata: lihat `isoflop_real_report.md`, `pd_plane_report.md`, `maze_feasibility_report.md`, `maze_depth_report.md`. Disimpan sebagai jejak historis (ablasi), bukan sumber klaim.

# Dataset ablasi terkonsolidasi + learning curves

Dibangun dari 3 sumber run (49 run total, 616 eval-checkpoint). Tiap baris `ablation_master.csv` self-describing (config penuh + metrik + energi).

## Artefak
- `ablation_master.csv` — 1 baris/run (source, hidden, D_eff, L/H_cycles, seed, params, epochs, eval_interval, wall, exact/token/lm_loss/q_halt, net/gross energy, mean_W, cc_gpu_Wh, %agree).
- `learning_curves_all.csv` — long-form semua eval-checkpoint (plot-ready).
- `fig_learning_budget_exact.png`, `fig_energy_accuracy_frontier.png`, `fig_isoflop_depth.png`.

## Cakupan
- **frontier**: 3 run, 3 sel (hidden×D). budget hidden=128 seed unik: 5.
- **isoflop**: 9 run, 9 sel (hidden×D). budget hidden=128 seed unik: 5.
- **budget**: 18 run, 3 sel (hidden×D). budget hidden=128 seed unik: 5.
- **scale**: 10 run, 10 sel (hidden×D). budget hidden=128 seed unik: 5.
- **maze**: 8 run, 2 sel (hidden×D). budget hidden=128 seed unik: 5.
- **converge**: 1 run, 1 sel (hidden×D). budget hidden=128 seed unik: 5.

## Tabel ringkas (per run)

| source | hidden | D_eff | seed | exact% | token% | lm_loss | net Wh | gross Wh | %agree |
|---|---|---|---|---|---|---|---|---|---|
| budget | 128 | 9 | 0 | 5.469 | 67.819 | 0.7418 | 33.701 | 34.674 | — |
| budget | 128 | 9 | 0 | 5.469 | 67.819 | 0.7418 | — | — | — |
| budget | 128 | 9 | 1 | 10.938 | 69.066 | 0.7178 | — | — | — |
| budget | 128 | 9 | 2 | 5.273 | 67.409 | 0.7468 | — | — | — |
| budget | 128 | 9 | 3 | 5.859 | 67.822 | 0.7426 | 33.919 | 34.895 | 99.00 |
| budget | 128 | 9 | 4 | 3.320 | 67.508 | 0.7398 | 34.480 | 35.459 | 98.87 |
| budget | 128 | 18 | 0 | 13.281 | 68.764 | 0.7235 | 59.603 | 61.231 | — |
| budget | 128 | 18 | 0 | 13.281 | 68.764 | 0.7235 | — | — | — |
| budget | 128 | 18 | 1 | 7.812 | 67.950 | 0.7434 | — | — | — |
| budget | 128 | 18 | 2 | 6.445 | 67.998 | 0.7403 | — | — | — |
| budget | 128 | 18 | 3 | 8.984 | 67.947 | 0.7658 | 60.132 | 61.763 | 98.86 |
| budget | 128 | 18 | 4 | 10.547 | 68.253 | 0.7428 | 59.942 | 61.572 | 98.89 |
| budget | 128 | 36 | 0 | 15.234 | 69.459 | 0.7162 | 109.845 | 112.811 | — |
| budget | 128 | 36 | 0 | 15.234 | 69.459 | 0.7162 | — | — | — |
| budget | 128 | 36 | 1 | 6.055 | 66.872 | 0.7735 | — | — | — |
| budget | 128 | 36 | 2 | 11.133 | 68.844 | 0.7285 | — | — | — |
| budget | 128 | 36 | 3 | 6.641 | 68.374 | 0.7443 | 109.926 | 112.894 | 98.90 |
| budget | 128 | 36 | 4 | 5.078 | 66.498 | 0.7751 | 109.927 | 112.895 | 98.88 |
| converge | 128 | 36 | 0 | 10.352 | 68.545 | 0.7267 | 292.110 | 299.977 | — |
| frontier | 256 | 9 | 0 | 5.078 | 67.009 | 0.7584 | 17.139 | 17.634 | — |
| frontier | 256 | 18 | 0 | 4.297 | 66.942 | 0.7837 | 30.500 | 31.390 | — |
| frontier | 256 | 36 | 0 | 3.516 | 67.658 | 0.7364 | 56.640 | 58.251 | — |
| isoflop | 128 | 9 | 0 | 5.469 | 67.441 | 0.7627 | 10.050 | 64.704 | — |
| isoflop | 128 | 18 | 0 | 2.734 | 64.849 | 0.8682 | 10.348 | 53.434 | — |
| isoflop | 128 | 36 | 0 | 0.195 | 62.977 | 0.8963 | 11.263 | 41.515 | — |
| isoflop | 256 | 9 | 0 | 2.539 | 65.664 | 0.7999 | 10.415 | 28.942 | — |
| isoflop | 256 | 18 | 0 | 0.000 | 61.097 | 0.8977 | 10.629 | 17.665 | — |
| isoflop | 256 | 36 | 0 | 0.000 | 56.298 | 1.1017 | 11.482 | 12.283 | — |
| isoflop | 384 | 9 | 0 | 0.586 | 62.493 | 0.8614 | 9.786 | 11.344 | — |
| isoflop | 384 | 18 | 0 | 0.000 | 58.439 | 0.9753 | 10.065 | 11.666 | — |
| isoflop | 384 | 36 | 0 | 0.000 | 42.407 | 1.5466 | 10.490 | 12.707 | — |
| maze | 128 | 9 | 0 | 0.000 | 81.771 | 0.8760 | 66.897 | 68.959 | — |
| maze | 128 | 9 | 0 | 0.000 | 84.148 | 0.7709 | 145.517 | 149.919 | — |
| maze | 128 | 9 | 1 | 0.000 | 83.691 | 0.9229 | 145.293 | 149.694 | — |
| maze | 128 | 9 | 2 | 0.000 | 85.799 | 0.7518 | 144.755 | 149.156 | — |
| maze | 128 | 18 | 0 | 0.000 | 82.948 | 0.8321 | 117.367 | 120.865 | — |
| maze | 128 | 18 | 0 | 0.000 | 82.909 | 0.9301 | 253.660 | 261.252 | — |
| maze | 128 | 18 | 1 | 0.000 | 83.744 | 0.8590 | 253.256 | 260.848 | — |
| maze | 128 | 18 | 2 | 0.000 | 83.485 | 0.8728 | 252.573 | 260.163 | — |
| scale | 128 | 9 | 0 | 1.172 | 63.898 | 0.8406 | 4.715 | 4.867 | — |
| scale | 128 | 18 | 0 | 1.953 | 63.624 | 0.8633 | 8.551 | 8.800 | — |
| scale | 256 | 9 | 0 | 0.977 | 63.964 | 0.8263 | 9.727 | 10.020 | — |
| scale | 256 | 18 | 0 | 1.953 | 64.022 | 0.8335 | 16.986 | 17.480 | — |
| scale | 384 | 9 | 0 | 2.148 | 64.224 | 0.8273 | 15.191 | 15.667 | — |
| scale | 384 | 18 | 0 | 2.344 | 65.041 | 0.8105 | 26.410 | 27.225 | — |
| scale | 512 | 9 | 0 | 5.273 | 64.634 | 1.0125 | 23.712 | 24.490 | — |
| scale | 512 | 18 | 0 | 2.344 | 65.034 | 0.7984 | 40.831 | 42.136 | — |
| scale | 768 | 9 | 0 | 0.391 | 64.808 | 0.8041 | 39.583 | 40.850 | — |
| scale | 768 | 18 | 0 | 2.148 | 64.518 | 0.8404 | 68.477 | 70.628 | — |

