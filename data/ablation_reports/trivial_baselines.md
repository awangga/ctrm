# Baseline sepele pada subset evaluasi 512 (auto-generated)

Metrik identik dengan `models/losses.py` upstream (mask label, rata-rata per urutan).

| task | n | posisi berlabel/contoh | majority token | copy-input token | majority exact | copy-input exact |
|---|---:|---:|---:|---:|---:|---:|
| Sudoku-Extreme | 512 | 81 | 11.11% | 30.92% | 0.00% | 0.00% |
| Maze-Hard | 512 | 900 | 50.03% | 87.51% | 0.00% | 0.00% |
| ARC-AGI-1 | 419 | 259 | 41.82% | 60.75% | 0.00% | 0.00% |
| ARC-AGI-1 subset lama (1 task + augmentasi, TIDAK SAH) | 512 | 48 | 25.00% | 25.00% | 0.00% | 0.00% |

Catatan: pada Maze-Hard seluruh 900 posisi berlabel dan sel lintasan hanya 12,5% dari grid,
sehingga menyalin input sudah mencapai 87,51%. Akurasi token model (86,5-86,8%) berada DI BAWAH
angka itu, jadi metrik token Maze tidak mengukur penguasaan task pada anggaran ini.
Pada ARC-AGI-1 (subset sah, 419 contoh dari 400 task) baseline token majority 41.82% dan copy-input 60.75%. Angka ini dihitung ulang pada fase BS; angka 25,00% lama berasal dari subset yang ternyata satu task beserta augmentasinya.
Untuk akurasi EXACT (metrik yang dilaporkan pada Sudoku) baseline sepele adalah 0% di ketiga task,
sehingga 62,4% Sudoku tidak dapat dibandingkan dengan baseline token 30,92% di sumbu yang sama.
