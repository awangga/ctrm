# Hasil fase BM (analisis pra-spesifikasi, dijalankan sekali)

Ambang Bonferroni per sumbu: 0.0167.


Higienis data: tidak ada baris yang dibuang, tidak ada duplikat tag, jumlah run tiap batch sesuai pra-registrasi.


## A1. ARC-AGI-1: sel D36 dengan batch efektif 48 (compute sama, epoch sama)

| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |
|---|---|---|---|---|---|
| D9 vs D36 (alokasi LAMA, batch 24) | 36.27±3.18 | 30.59±1.78 | +5.67 | p=0.0122 (perm 0.0159, lantai 0.0079) | LOLOS |
| D9 vs D36 (alokasi BARU, batch efektif 48) | 36.27±3.18 | 28.46±0.98 | +7.80 | p=0.0039 (perm 0.0079, lantai 0.0079) | LOLOS |
| D18 vs D36 (alokasi BARU) | 33.34±3.37 | 28.46±0.98 | +4.88 | p=0.0292 (perm 0.0079, lantai 0.0079) | gagal |
| D36 lama vs D36 baru (efek alokasi saja) | 30.59±1.78 | 28.46±0.98 | +2.13 | p=0.0555 (perm 0.0397, lantai 0.0079) | gagal |

Energi net D36: lama 278 Wh, baru 290 Wh (+5%). Langkah optimizer baru kira-kira separuh lama, contoh yang dikonsumsi sama.

## A2. Sudoku-Extreme: sel D36 dengan batch efektif 192

| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |
|---|---|---|---|---|---|
| D9 vs D36 (alokasi LAMA, batch 96) | 62.43±0.30 | 36.26±0.79 | +26.17 | p=0.0001 (perm 0.1000, lantai 0.1000) | LOLOS |
| D9 vs D36 (alokasi BARU, batch efektif 192) | 62.43±0.30 | 31.71±0.63 | +30.73 | p=0.0000 (perm 0.1000, lantai 0.1000) | LOLOS |
| D18 vs D36 (alokasi BARU) | 50.07±1.68 | 31.71±0.63 | +18.36 | p=0.0010 (perm 0.1000, lantai 0.1000) | LOLOS |
| D36 lama vs D36 baru (efek alokasi saja) | 36.26±0.79 | 31.71±0.63 | +4.56 | p=0.0018 (perm 0.1000, lantai 0.1000) | LOLOS |

## B. Ringkasan run belum lengkap (`arc_baseline_out/`); batch B belum selesai.


## C. ARC-AGI-1: checkpoint dipilih pada separuh subset, dilaporkan pada separuh lain

| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |
|---|---|---|---|---|---|
| D9 vs D36 (held-out half) | 39.29±5.03 | 28.44±1.00 | +10.85 | p=0.0075 (perm 0.0079, lantai 0.0079) | LOLOS |

Per run D9: h256_d9_recipe_b48_s0=39.68, h256_d9_recipe_b48_s1=47.71, h256_d9_recipe_b48_s2=35.91, h256_d9_recipe_b48_s3=35.16, h256_d9_recipe_b48_s4=38.00

Per run D36: h256_d36_recipe_b24_s0=29.26, h256_d36_recipe_b24_s1=28.74, h256_d36_recipe_b24_s2=27.28, h256_d36_recipe_b24_s3=29.45, h256_d36_recipe_b24_s4=27.49
