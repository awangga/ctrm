# Hasil AMANDEMEN 5: kontras kedalaman ARC pada subset sah (dijalankan sekali)

Subset `arc1-aug1k-g400`: 400 task ARC berbeda, 419 contoh. D36 memakai akumulasi gradien
(micro 24 x 2 = batch efektif 48, sama dengan D9).

## 1. Pemeriksaan baseline sepele (wajib sebelum kontras dibaca)

Baseline token pada subset ini: majority 41.82%, copy-input 60.75%.

| lengan | akurasi token terbaik (per seed) | rerata | di atas copy-input? |
|---|---|---:|---|
| D9 | 63.74, 63.66, 63.63, 63.55, 64.04 | 63.72±0.19 | ya |
| D36 | 62.53, 62.77, 62.55, 62.37, 62.39 | 62.52±0.16 | ya |

## 2. Kontras D9 lawan D36

| kontras | D9 | D36 | selisih | Welch | permutasi | Bonferroni 0,0167 |
|---|---:|---:|---:|---|---|---|
| D9 lawan D36 | 63.72±0.19 | 62.52±0.16 | +1.20 | t=10.81, df=7.7, p=0.0000, d=6.84 | p=0.0079 (lantai 0.0079) | LOLOS |

Energi net: D9 284.3±0.6 Wh, D36 267.8±0.3 Wh.

## 3. Vonis

Kedua lengan berada di atas baseline copy-input dan kontras lolos ambang Bonferroni sumbunya.
