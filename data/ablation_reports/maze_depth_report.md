# Depth-comparison Maze iso-compute (rezim akurasi nyata) — fase Q + Q-bis (3 seed)

Pertanyaan: apakah temuan Sudoku (recursion dangkal menang, depth tak earns-its-keep) **general** ke
task kedua (Maze-Hard, seq900)? Grid depth iso-compute @h256, metrik **token-accuracy**
(exact=0 di semua config Maze, uninformative — lihat maze_feasibility_report.md fase P).
Fase Q = 1 seed (seed0); **fase Q-bis menambah seed1+seed2** untuk uji signifikansi pola.

Artefak: `maze_depth_out/{progress,pw,emissions}_h256_d{9,18,36}_*_s{0,1,2}.* + recipe_summary.csv`.
Iso-compute: D_eff x batch x steps konstan (9x48x24000 = 18x48x12000 = 36x24x12000 = 10.368M).

## Hasil 3 seed (token-accuracy final, telusur ke artefak)

| D_eff | batch | steps | token% s0 | s1 | s2 | **mean±sd** | net Wh (mean) | cross-val % |
|---|---|---|---|---|---|---|---|---|
| 9  | 48 | 24000 | 83.85 | 82.96 | 82.54 | **83.12 ± 0.67** | 316.6 | 99.4–99.6 |
| 18 | 48 | 12000 | 82.98 | 83.61 | 83.48 | **83.36 ± 0.34** | 306.1 | 99.4–99.8 |
| 36 | 24 | 12000 | 85.28 | 84.76 | 83.99 | **84.68 ± 0.65** | 327.7 | 99.4–99.95 |

Rentang mean: 83.12–84.68 → **gap terbesar 1.56 poin** (D36 vs D9). Cross-val energi 99.4–99.95% sembilan run.

## Uji signifikansi (Welch t, Student-t exact; ANOVA satu-arah)

| kontras | diff (poin) | t | df | p (2-arah) | vs Bonferroni 0.0167 |
|---|---|---|---|---|---|
| D36 vs D9  | +1.56 | 2.90 | 4.0 | 0.044 | **NS** |
| D36 vs D18 | +1.32 | 3.12 | 3.0 | 0.052 | **NS** |
| D18 vs D9  | +0.24 | 0.55 | 2.9 | 0.619 | NS |

ANOVA: **F(2,6)=6.46, p=0.032** (marginal). **Tak satu pun kontras pasangan lolos koreksi
multiple-comparison.** Ada **tren lemah naik-terhadap-depth** (D36 tertinggi), arah **berlawanan**
Sudoku, tetapi tidak kokoh.

## Temuan lintas-task (Sudoku vs Maze)

| | Sudoku (exact, n=3) | Maze (token, n=3) |
|---|---|---|
| D9  | 62.4 ± 0.24 | 83.12 ± 0.67 |
| D18 | 50.1 ± 1.37 | 83.36 ± 0.34 |
| D36 | 36.3 ± 0.64 | 84.68 ± 0.65 |
| pola | **monoton dangkal-menang** | **nyaris datar; tren-lemah dalam-unggul** |
| gap D-terbaik vs D-terburuk | **26 poin** (Welch t>12, p<1e-3) | **1.56 poin** (p=0.044, NS koreksi) |
| rasio gap | — | **~17× lebih kecil dari Sudoku** |

**Efek recursion depth TASK-DEPENDENT dalam arah maupun besaran.** Di Sudoku-Extreme, iso-compute,
dangkal menang telak dan monoton (gap 26 poin, sangat signifikan). Di Maze-Hard, depth nyaris tak
menggerakkan token-accuracy (gap 1.56 poin, ~17× lebih kecil, tak lolos koreksi); tren tipisnya
malah condong dalam-unggul (D36), kebalikan Sudoku. Ini **bukan** hukum universal "dangkal selalu
menang".

## Bingkai jujur (anti-overclaim)

1. **Yang berlaku di KEDUA task — depth tak pernah jadi alokasi energy-optimal.** Di Sudoku recursion
   dalam kalah telak (akurasi turun) sekaligus boros. Di Maze, keunggulan D36 (+1.3 poin atas D18)
   (a) tak signifikan setelah koreksi, dan (b) menuntut energi **+7%** (328 vs 306 Wh) → **tidak
   energy-optimal**. Untuk target token ~83%, **D18 paling hemat**. Kesimpulan *less-is-more*
   **bertahan**: menambah depth tak pernah *earns its keep* pada anggaran compute tetap di kedua task.
2. **Yang TIDAK general:** besar dan **arah** efek. "Frontier compute-optimal condong dangkal" kuat di
   Sudoku, hilang/berbalik-lemah di Maze. Kontribusi paper = **peta rezim** (kapan & seberapa depth
   penting), bukan klaim tunggal universal. Ini justru memperkuat "diagnosis → generalisasi" (CLAUDE.md).
3. **Status uji:** dengan 3 seed, gap Maze terbesar **p=0.044** (gagal Bonferroni 3-kontras); ANOVA
   marginal p=0.032. Laporkan sebagai **efek-depth lemah/near-flat pada Maze-token**, bukan
   "dangkal menang" maupun "dalam menang". n=3 seed setara Sudoku; keterbatasan tersisa: exact=0
   memaksa metrik token (proxy lebih longgar), rentang skala/params terbatas.
4. **Energi:** metrik utama Joule-to-target tetap valid & cross-val 99.4–99.95%. Di Maze, depth ekstra
   (D36) menambah energi tanpa akurasi berarti; D18 titik hemat. Konsisten arah Green-AI di kedua task.

## Implikasi manuscript
Bingkai kontribusi sebagai **peta rezim energi-akurasi lintas-task**: (i) Sudoku → frontier tajam,
monoton condong dangkal; (ii) Maze → depth ~netral pada token (efek ~17× lebih kecil, tak lolos
koreksi), energi-optimal di depth sedang (D18). Benang merah yang **tahan di kedua task**: **recursion
depth tak pernah menjadi alokasi compute/energy-optimal**; alokasikan anggaran ke optimasi/lebar sedang.
Pembeda terukur dari klaim TRM (depth substitusi parameter) yang **tidak compute-optimal** pada anggaran
tetap di kedua task yang diuji.
