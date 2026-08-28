> **[REGIME: TOY/PILOT — SUPERSEDED]** Report ini memakai data rezim-toy lama (h128/no-EMA, exact 0-15%). Hasil final manuscript memakai rezim akurasi nyata: lihat `isoflop_real_report.md`, `pd_plane_report.md`, `maze_feasibility_report.md`, `maze_depth_report.md`. Disimpan sebagai jejak historis (ablasi), bukan sumber klaim.

# Cross-validasi energi: CodeCarbon vs nvidia-smi

Sumber: `budget_out/energy_xval.csv` (6 run, hidden=128, D=9/18/36, seed 3-4). Kedua metode mengukur energi GPU pada window training fisik yang sama (CodeCarbon `gpu_energy` via pynvml; nvidia-smi `power.draw` 1 Hz diintegral trapezoid). Pembanding: energi GROSS (total draw), bukan net, karena keduanya mengukur draw total.

## Per-run

| run | D_eff | seed | wall s | nvidia-smi gross (Wh) | CodeCarbon GPU (Wh) | setuju % |
|---|---|---|---|---|---|---|
| h128_d9_s3 | 9 | 3 | 747 | 34.895 | 34.547 | 99.00 |
| h128_d18_s3 | 18 | 3 | 1250 | 61.763 | 61.064 | 98.86 |
| h128_d36_s3 | 36 | 3 | 2274 | 112.894 | 111.656 | 98.90 |
| h128_d9_s4 | 9 | 4 | 749 | 35.459 | 35.061 | 98.87 |
| h128_d18_s4 | 18 | 4 | 1249 | 61.572 | 60.892 | 98.89 |
| h128_d36_s4 | 36 | 4 | 2274 | 112.895 | 111.641 | 98.88 |

## Ringkas

- Rerata kesepakatan: **98.90%** (min 98.86%).
- Selisih relatif absolut: rata-rata **1.10%**, maksimum **1.14%**.
- Run lolos ambang >95%: **6/6**.

## Vonis

**LOLOS** target integritas #3 (>95% setuju): 6/6 run, rerata 98.90%. Pengukuran energi tervalidasi silang; metrik Joule/Wh-untuk-target dapat dipercaya.

Catatan: CodeCarbon `gpu_energy` konsisten sedikit LEBIH RENDAH (~1%) dari integral nvidia-smi; selisih sistematis kecil ini wajar (beda interval sampling: CodeCarbon 5 s vs nvidia-smi 1 s, dan penanganan tepi window). Keduanya membaca sensor pynvml yang sama, jadi ini validasi konsistensi pembukuan, bukan dua sensor fisik independen.

