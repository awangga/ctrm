# Analisis pelengkap fase BS (auto-generated oleh analyze_BS_supplement.py)


## J. Pembelahan separuh-seleksi / separuh-pelaporan (pra-registrasi batch C)

Seleksi checkpoint pada separuh genap (210 contoh), pelaporan pada separuh ganjil (209 contoh).

| kontras | grup 1 | grup 2 | selisih | Welch | permutasi | pemisahan total |
|---|---|---|---|---|---|---|
| ARC D9 lawan D36, held-out | 65.42+-0.36 | 64.70+-0.15 | +0.73 | t=4.17 df=5.4 p=7.41e-03 d=2.64 | perm 0.0079 (lantai 0.0079) | min1 64.95 > maks2 64.86: True |

## K. ARC, checkpoint AKHIR

| kontras | grup 1 | grup 2 | selisih | Welch | permutasi | pemisahan total |
|---|---|---|---|---|---|---|
| ARC D9 lawan D36, final | 63.38+-0.48 | 62.08+-0.53 | +1.30 | t=4.04 df=7.9 p=3.79e-03 d=2.56 | perm 0.0079 (lantai 0.0079) | min1 62.81 > maks2 62.62: True |

## L. Energi ke target yang SAMA untuk kedua lengan

**Sudoku**, target 36.26% (rerata akurasi terbaik lengan dalam)
- dangkal ke target: [162, 161, 160] -> rerata 161+-1 Wh, median 161 (3/3 seed)
- dalam ke target: [None, 323, None] -> rerata 323 Wh (1/3 seed)
- dalam, energi total run: rerata 340 Wh
- rasio dangkal/total 47%, dangkal/dalam-ke-target 50%

**ARC g400**, target 62.52% (rerata akurasi terbaik lengan dalam)
- dangkal ke target: [68, 217, 57, 45, 102] -> rerata 98+-70 Wh, median 68 (5/5 seed)
- dalam ke target: [214, 204, 247, None, None] -> rerata 221+-22 Wh (3/5 seed)
- dalam, energi total run: rerata 268 Wh
- rasio dangkal/total 37%, dangkal/dalam-ke-target 44%


## M. Batch A2 Sudoku (pra-registrasi butir 2)

| kontras | grup 1 | grup 2 | selisih | Welch | permutasi | pemisahan total |
|---|---|---|---|---|---|---|
| D9 lawan D36 batch disamakan | 62.43+-0.30 | 31.71+-0.63 | +30.73 | t=76.57 df=2.9 p=7.97e-06 d=62.52 | perm 0.1000 (lantai 0.1000) | min1 62.11 > maks2 32.42: True |
| D36 lama lawan D36 baru | 36.26+-0.79 | 31.71+-0.63 | +4.56 | t=7.83 df=3.8 p=1.76e-03 d=6.39 | perm 0.1000 (lantai 0.1000) | min1 35.55 > maks2 32.42: True |

## N. Tumpang tindih uji-latih ARC

Pasangan (input, label) uji yang identik dengan suatu pasangan latih: **3 dari 419**. Contoh uji task evaluasi ditahan oleh protokol upstream (`dataset/build_arc_dataset.py`); demonstrasinya dilatih. Kecocokan ini kebetulan dan mengenai kedua kedalaman setara.
