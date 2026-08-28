# Depth-comparison ARC-AGI-1 iso-compute (rezim akurasi nyata) — fase R (task-3), 3 seed

Pertanyaan: apakah temuan depth (Sudoku dangkal-menang monoton; Maze near-flat) berlaku di task
**ketiga**, ARC-AGI-1? Grid depth iso-compute @h256, metrik **token-accuracy** (exact=0 di semua config
ARC pada anggaran ini, sama seperti Maze). **3 seed** (0,1,2).

Artefak: `arc_depth_out/{progress,pw,emissions}_h256_d{9,18,36}_recipe_*_s{0,1,2}.* + recipe_summary.csv`.
Data: `arc1-aug1k-e512` (ARC-AGI-1 training+evaluation, augmentasi 1000x, test di-truncate 512).
Iso-compute: D_eff x batch x steps konstan. GROUPS=3080 (mean_puzzle_examples=3.85).

## KOREKSI KEJUJURAN (seed0 -> 3-seed)
Laporan seed0 (n=1) sempat menyimpulkan **"monoton dangkal-menang, gap 10.35 poin"** (D9=36.46 >
D18=29.83 > D36=26.11). **Replikasi 3-seed MEMBATALKAN klaim itu.** Gap D9-vs-D18 seed0 ternyata
**di-inflate satu seed**: pada seed2, D18=36.41% (outlier tinggi) sehingga rata-rata D18 melonjak dan
mendekati D9. Variansi antar-seed ARC jauh lebih besar (sd 3-5 poin) daripada Sudoku/Maze (sd 0.2-1.7).
Verdict yang benar ada di bawah.

## Hasil 3-seed (token-accuracy final, telusur ke artefak)

| D_eff | batch | token% (mean±std) | 3 seed (s0/s1/s2) | net Wh (mean) | cross-val % |
|---|---|---|---|---|---|
| 9  | 48 | **32.78 ± 3.23** | 36.46 / 30.40 / 31.49 | 291.9 | 99.1-99.7 |
| 18 | 48 | **31.19 ± 4.68** | 29.83 / 27.34 / 36.41 | 283.6 | 99.2-99.7 |
| 36 | 24 | **25.98 ± 0.61** | 26.11 / 25.31 / 26.51 | 280.6 | 99.2-99.8 |

exact=0 semua (ARC sangat sulit di anggaran 1-GPU) -> metrik token-acc.

## Uji signifikansi (Welch t, Student-t exact; ANOVA satu-arah)

| kontras | diff (poin) | t | df | p (2-arah) | vs Bonferroni 0.0167 |
|---|---|---|---|---|---|
| D9 vs D18  | +1.59 | 0.48 | 3.6 | 0.657 | **NS (seri)** |
| D9 vs D36  | +6.81 | 3.59 | 2.1 | 0.063 | **NS** |
| D18 vs D36 | +5.22 | 1.91 | 2.1 | 0.192 | **NS** |

ANOVA: **F(2,6)=3.49, p=0.099** (NS). **Tak satu pun kontras lolos koreksi multiple-comparison.**

## Verdict jujur (anti-overclaim)
1. **D9 ≈ D18 (seri, p=0.66); keduanya cenderung di atas D36** (selisih ~5-7 poin) **tetapi TIDAK
   signifikan di n=3** (p=0.06-0.19; ANOVA p=0.099). Bukan "monoton dangkal-menang" seperti kesan seed0.
2. **Arah konsisten dengan hipotesis "depth dalam tak membantu":** D36 (terdalam) selalu numerik
   TERENDAH dan paling rapat (sd 0.61), sementara D9/D18 lebih tinggi tapi berisik. Namun besarnya
   efek tak melewati derau seed -> laporkan sebagai **efek-depth lemah/tak-signifikan pada ARC-token**,
   mirip Maze, bukan seperti Sudoku.
3. **Energi:** D36 termurah (280.6 Wh) tapi juga terendah akurasi; D9 tertinggi akurasi (32.8%) pada
   energi ~4% lebih besar (291.9 Wh). Untuk mencapai token tertinggi, dangkal lebih efektif; deep tak
   pernah memberi keunggulan akurasi yang membayar energinya. Cross-val CodeCarbon vs nvidia-smi 99.1-99.8%.
4. **Keterbatasan:** n=3, exact=0 memaksa token (proxy), anggaran 1-GPU << 4xH100 (absolut 26-33% jauh
   di bawah 45% SOTA; yang dibandingkan ORDERING iso-compute lintas-depth, bukan SOTA). Tanpa ConceptARC.

## Temuan lintas-task (3 task, rezim akurasi nyata, 3 seed masing-masing)

| Task | metrik | D9 | D18 | D36 | pola | signifikansi |
|---|---|---|---|---|---|---|
| Sudoku-Extreme | exact | 62.4 | 50.1 | 36.3 | **monoton dangkal-menang** | **kuat** (gap 26, Welch t>12) |
| ARC-AGI-1 | token | 32.78 | 31.19 | 25.98 | D9≈D18 > D36 (tren) | **lemah/NS** (ANOVA p=0.099) |
| Maze-Hard | token | 83.12 | 83.36 | 84.68 | near-flat | NS (gap 1.56) |

**Verdict lintas-task (dikoreksi):** Hanya **Sudoku** menunjukkan efek depth kuat & signifikan (dangkal
menang telak). Di **ARC dan Maze** efek depth **lemah/tak-signifikan di n=3**. Benang merah yang bertahan
di KETIGA task: **menambah recursion depth tak pernah MENANG secara signifikan** pada anggaran tetap;
pada ARC/Maze ia netral-atau-merugikan-tak-signifikan, pada Sudoku ia merugikan telak. Deep (D36) tak
pernah jadi alokasi compute/energy-optimal di mana pun. Kontribusi = **peta rezim energi-akurasi
lintas-task**: kekuatan keuntungan-dangkal bergantung task (kuat di Sudoku diskriminatif; lemah di ARC &
Maze). Ini pembingkaian jujur "diagnosis -> generalisasi", bukan klaim hukum tunggal universal.

## Implikasi manuscript
ARC MEMPERKUAT (bukan melemahkan) tesis "depth tak pernah earns-its-keep": di ketiga task deep tak
pernah menang signifikan. Tapi ARC juga menegaskan **peta rezim**, bukan hukum monoton universal: efek
depth tegas hanya di Sudoku. Sajikan ARC apa adanya (D9≈D18>D36 tren, NS n=3), JANGAN sebagai "dangkal
menang" kedua. Ini justru lebih tahan reviewer Q1 (mengakui variansi & keterbatasan n).
