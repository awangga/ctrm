# Feasibility task-2 Maze (rezim akurasi nyata) — fase P

Tujuan: konfirmasi pipeline resep-setia TRM jalan di task **kedua** (Maze-Hard, seq900) di GPU lokal,
sebagai prasyarat generalitas lintas-task untuk klaim compute-optimal recursion frontier (jangan
single-task). Artefak: `maze_real_out/{progress,pw,emissions}_h256_d18_recipe_b48_s0.* + recipe_summary.csv`.

## Config
resep setia (EMA, augmentasi maze-aug), HIDDEN=256, D_eff=18, batch=48, ~7916 step (epochs=380),
seed=0, eval_interval=19 epoch (20 eval). Dataset maze-aug (train), test truncated 512.

## Hasil terukur (telusur ke artefak)

| metrik | nilai |
|---|---|
| wall | 4829 s (~1.34 jam) |
| token-accuracy (first / max / last) | **23.7% / 86.7% / 85.5%** |
| exact-match (full path) | **0.0%** sepanjang 20 eval |
| net energi (idle-corrected) | 209.4 Wh |
| cross-val CodeCarbon vs nvidia-smi | **99.66%** |
| GPU mem puncak | ~7.3 GB (muat 16GB nyaman; batch48 tanpa OOM) |

Kurva token (20 eval): 23.7 -> 53.7 -> 67.9 -> 83.2 -> 86.7 -> plateau 84-86 (stabil sejak ~step 2000).

## Verdict feasibility

1. **FEASIBLE & BELAJAR.** Token-accuracy melonjak 23.7 -> 86.7% lalu plateau ~85% -> pipeline Maze
   resep-setia jalan benar di GPU lokal. Tidak ada OOM pada seq900 batch48 (~7.3GB).
2. **exact-match = 0 sepanjang run.** Pada h256/D18, model menguasai struktur token tetapi tak pernah
   menghasilkan path maze yang benar 100%. exact tidak dapat membedakan antar-config di rezim ini.
3. **KEPUTUSAN METRIK lintas-task: gunakan TOKEN-ACCURACY untuk Maze**, bukan exact. Ini jujur dan
   discriminating (token bervariasi & sensitif), sedangkan exact akan rata-0 (uninformative).
   Untuk Sudoku metrik utama tetap exact (di sana exact 36-62%, sangat discriminating).

## Catatan kejujuran
- Run gagal-2-detik pertama (PATH venv belum di-export, ModuleNotFoundError torch) sudah diperbaiki
  (export PATH venv) dan baris cacatnya (wall=1.7s, net=-0.0001) DIHAPUS dari recipe_summary.csv.
- Feasibility ini n=1, h256/D18 saja. BELUM menjawab pertanyaan paper (depth earns-its-keep di Maze?).
  Itu butuh grid depth-comparison (usulan di bawah).

## Usulan langkah berikut (BUTUH KONFIRMASI USER sebelum jalan)
Grid depth-comparison Maze iso-compute, paralel dgn desain Sudoku:
- Config: D_eff in {9, 18, 36} @ h512 (atau h256 utk hemat), EMA, iso-compute (params x D x batch x steps konstan).
- Metrik: **token-accuracy** (exact diperkirakan 0). 3 seed/config utk error bar (spt Sudoku).
- Estimasi durasi: seq900 ~0.7 step/s. Per config ~3-4 jam; 3 config x 1 seed ~10-12 jam;
  3 config x 3 seed ~30-36 jam. Bisa dipangkas (h256, step lebih sedikit, atau 1 seed dulu lalu replikasi).
- Hipotesis (diuji, bukan diklaim): bila Maze juga dangkal-menang -> generalitas frontier menguat;
  bila berbeda -> peta rezim lintas-task (tetap temuan sah, less-is-more vs task-dependent).
