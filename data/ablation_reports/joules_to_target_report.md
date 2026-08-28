> **[REGIME: TOY/PILOT — SUPERSEDED]** Report ini memakai data rezim-toy lama (h128/no-EMA, exact 0-15%). Hasil final manuscript memakai rezim akurasi nyata: lihat `isoflop_real_report.md`, `pd_plane_report.md`, `maze_feasibility_report.md`, `maze_depth_report.md`. Disimpan sebagai jejak historis (ablasi), bukan sumber klaim.

# Joules-to-target-accuracy (metrik utama) + Pareto frontier

Energi kumulatif net (idle-corrected) untuk MENCAPAI target akurasi tau, dihitung dari deret-waktu daya (pw_*.csv) di-join ke checkpoint eval (progress). Checkpoint pertama yang menembus tau menentukan energi. 'tak tercapai' = config tak pernah lewati tau pada budget ini (bukti saturasi).

Idle=4.7W. Sumber: budget_out, frontier_out, scale_out, maze_out, converge_out (hanya run berdaya-terukur).

## Budget_out (h128, ~30k step, 5 seed): energi rata-rata untuk token-acc >=65%

| D_eff | Wh-to-token65 (mean) | n seed capai | Wh-to-exact5 (mean) | n capai |
|---|---|---|---|---|
| 9 | 7.32 | 3/5 | 13.39 | 3/5 |
| 18 | 11.92 | 3/5 | 19.91 | 3/5 |
| 36 | 21.97 | 3/5 | 43.94 | 3/5 |

## Interpretasi

- Config TERMURAH mencapai token-acc 65%: **h128_d9** (D=9) pada **6.43 Wh**.
- Rata-rata Wh-to-token65: D=9 -> 9.54 Wh vs D=36 -> 21.88 Wh (**2.3x** lebih boros utk target sama). Saturasi energi terukur eksplisit.
- Target exact-acc 10% dicapai oleh 8/31 run (akurasi rendah = model under-trained; metrik token lebih informatif pada budget ini).

