# Iso-FLOP depth frontier di REZIM AKURASI NYATA (h512+EMA, Sudoku-aug1k)

Pertanyaan inti paper: pada anggaran compute/energi setara, apakah recursion depth *earns its keep*?
Dijawab di rezim akurasi nyata (bukan toy 18%): resep setia TRM (h512, EMA, sudoku augmentasi ~1M),
tiga config **iso-compute** (params×D_eff×batch×steps konstan; ratio 1.00).

## Hasil (artefak: recipe_out/progress_h512_d{9,18,36}_*.jsonl + recipe_summary.csv)

| D_eff | batch | steps | best_exact | token% | net Wh | cross-val % | Wh-to-exact50 |
|---|---|---|---|---|---|---|---|
| **9**  | 192 | 50k | **62.7** | 86.3 | 405 | 99.46 | **259** |
| 18 | 192 | 25k | 51.6 | 81.9 | 360 | 99.27 | 317 |
| 36 | 96  | 25k | 35.5 | 76.8 | 341 | 99.63 | tak tercapai |

## Temuan

1. **Monoton & besar: depth dangkal MENANG.** D9 (62.7%) > D18 (51.6%) > D36 (35.5%) pada
   iso-compute. Selisih **27 poin** D9 vs D36 — jauh di luar noise (kontras tajam dgn rezim toy
   di mana beda depth tak signifikan, lihat significance_report).
2. **Depth AKTIF MERUGIKAN, bukan sekadar "tak berbayar".** Pada anggaran tetap, tiap penggandaan
   depth memangkas akurasi tajam. Recursion bukan pengganti efisien parameter/step di sini.
3. **Efisiensi energi (Joules-to-target):** untuk capai exact 50%, D9 butuh **259 Wh**, D18 317 Wh,
   D36 **tak pernah** mencapainya. D9 ~18% lebih hemat energi dari D18 untuk akurasi sama, lalu
   melanjut ke 62.7%.
4. **Cross-val energi 99.3-99.6%** di ketiga run (CodeCarbon vs nvidia-smi).

Catatan kejujuran: energi tidak persis iso (D9 405 vs D36 341 Wh) karena per-step-time tak persis
∝ params×D (ada overhead tetap + eval); D9 menjalankan 2× step → wall lebih lama. Namun arah dan
besar efek tak berubah: bahkan dgn energi ~19% lebih banyak, D9 unggul +27 poin → kesimpulan kokoh.

## Implikasi paper (compute-optimal recursion frontier)
Di rezim akurasi nyata pada Sudoku-Extreme, **frontier compute-optimal condong ke recursion DANGKAL**:
alokasikan anggaran ke lebih banyak langkah optimasi (depth kecil), bukan ke kedalaman recursion.
Ini *less-is-more* yang dinyatakan kuat & terukur energinya — bukan null/ambigu seperti rezim toy.
Pembeda dari klaim TRM (depth menggantikan parameter): substitusi itu **tidak compute-optimal**
pada anggaran tetap untuk task ini.

## Replikasi 3-seed (celah n=1 DITUTUP) — fase N-bis
Tiap config diulang seed 0,1,2 (9 run total). best_exact (%) mean±std:

| D_eff | mean±std exact | 3 seed (s0/s1/s2) | net Wh | cross-val % |
|---|---|---|---|---|
| 9  | **62.4 ± 0.30** | 62.7/62.5/62.1 | ~402 | 99.6 |
| 18 | 50.1 ± 1.68 | 51.6/50.4/48.2 | ~358 | 99.5 |
| 36 | 36.3 ± 0.79 | 35.5/37.1/36.1 | ~340 | 99.7 |

**Stabilitas:** ordering D9>D18>D36 berlaku di SETIAP seed; rentang tak tumpang-tindih
(min D9=62.1 > max D18=51.6 > ... > max D36=37.1). Welch t: D9-vs-D18=12.5, D18-vs-D36=12.9
(efek jauh di luar variansi seed). **Celah n=1 TERTUTUP: frontier compute-optimal-dangkal
KOKOH & signifikan.** Std seed kecil (0.3-1.7) vs gap antar-depth ~12-14 poin.
