# Baseline non-rekursif: apakah recursion itu sendiri sepadan? — fase T/T-bis (rek #2, 3 seed)

Audit Q1 (risiko #3): perbandingan hanya depth-vs-width dalam TRM; tak ada baseline non-rekursif. Ini
menjawabnya: **transformer non-rekursif** (arch `transformers_baseline`, H_cycles=1, 8 layer, h512)
dilatih pada Sudoku-Extreme resep setia, pada anggaran **energi ~setara** TRM terbaik, **3 seed**.

## Setup (matched-energy)
- Baseline: h512, 8 layer transformer non-rekursif, batch 192, **100000 step**, EMA, sudoku-aug1k,
  pos_encodings=rope (default arch; baseline tak dukung `none`). Deviasi jujur: TRM pakai pos none.
- Target: samakan energi dgn TRM D9 (402 Wh). Baseline mencapai **384.0 Wh** (dalam ~5% dari target).

## Hasil 3 seed (telusur: recipe_out/recipe_summary.csv + progress_/pw_/emissions_)

| model | exact% (mean±std) | net Wh | Wh-to-exact50 | cross-val % |
|---|---|---|---|---|
| **TRM D9 (dangkal)** | **62.4 ± 0.30** | 402 | **259** | 99.6 |
| non-recursive baseline | 49.7 ± 2.14 | 384 | ~384 | 99.6 |
| TRM D18 (sedang) | 50.1 ± 1.68 | 358 | 317 | 99.5 |
| TRM D36 (dalam) | 36.3 ± 0.79 | 340 | tak tercapai | 99.7 |

Baseline 3 seed: 51.37 / 50.39 / 47.27 (mean 49.67 ± 2.14). Semua TRM & baseline n=3.

## Uji signifikansi (Welch t, Student-t exact) — TRM vs baseline

| kontras | diff (poin) | t | df | p | verdict |
|---|---|---|---|---|---|
| TRM D9 vs baseline  | **+12.76** | 10.22 | 2.1 | **0.008** | dangkal MENANG signifikan |
| TRM D18 vs baseline | +0.39 | 0.25 | 3.8 | 0.817 | **seri** (NS) |
| TRM D36 vs baseline | **-13.41** | -10.17 | 2.5 | **0.004** | dalam KALAH signifikan |

## Verdict: recursion sepadan HANYA bila dangkal (kini terbukti dua-arah, n=3)
1. **TRM dangkal (D9) mengalahkan baseline SIGNIFIKAN**: 62.4 vs 49.7% (+12.8 poin, p=0.008), DAN
   mencapai exact-50% pada **259 Wh vs ~384 Wh** (1.5x lebih hemat ke target). Recursion dangkal earns
   its keep.
2. **TRM sedang (D18) SERI dengan baseline**: 50.1 vs 49.7% (p=0.82). Keuntungan recursion sudah lenyap.
3. **TRM dalam (D36) KALAH SIGNIFIKAN dari baseline**: 36.3 vs 49.7% (-13.4 poin, p=0.004). Recursion
   dalam **lebih buruk daripada tanpa recursion sama sekali** pada anggaran setara.
4. **Sintesis:** nilai recursion nyata tapi **terkonsentrasi penuh di depth dangkal**. Kurva depth
   melintasi baseline: dangkal jauh di atas, sedang menyentuh, dalam jauh di bawah. Ini menyatukan tesis
   paper: bukan "recursion tak berguna" (D9 menang telak & signifikan), melainkan **"recursion hanya
   membayar bila dangkal; depth ekstra bukan hanya tak berbayar tapi kalah dari non-rekursif"**.

## Catatan kejujuran
- TRM & baseline sama-sama n=3. Kedua arah (D9>baseline, D36<baseline) signifikan setelah mempertimbangkan
  variansi seed. D18~baseline dalam derau.
- Perbedaan arsitektur: baseline pakai rope, TRM pakai pos none; puzzle_emb sama. Perbandingan = "recursion
  vs tidak" pada energi setara, bukan ablasi satu-variabel murni. Energi cross-val 99.6%.
