> **[REGIME: TOY/PILOT — SUPERSEDED]** Report ini memakai data rezim-toy lama (h128/no-EMA, exact 0-15%). Hasil final manuscript memakai rezim akurasi nyata: lihat `isoflop_real_report.md`, `pd_plane_report.md`, `maze_feasibility_report.md`, `maze_depth_report.md`. Disimpan sebagai jejak historis (ablasi), bukan sumber klaim.

# Ringkasan lintas-task: generalisasi saturasi energi (Sudoku vs Maze)

Pertanyaan: (1) apakah pola Sudoku --- depth naik => energi naik tanpa akurasi naik --- berlaku juga di Maze? (2) apakah run konvergensi menembus plateau exact>15%?

## Sudoku-pilot (h128, ~30k step, 5 seed)

| D_eff | token% (mean) | exact% (mean) | Wh-to-token65 (mean) |
|---|---|---|---|
| 9 | 67.9 | 6.2 | 7.8 |
| 18 | 68.2 | 9.4 | 11.9 |
| 36 | 68.0 | 8.8 | 22.0 |

## Maze-30x30-hard (h128, ~8k step, 3 seed; D=36 OOM 16GB)

| D_eff | token% (mean) | exact% (mean) | Wh-to-token60 (mean) |
|---|---|---|---|
| 9 | 84.5 | 0.0 | 9.0 |
| 18 | 83.4 | 0.0 | 15.8 |

## Generalisasi

- **Sudoku**: energi-untuk-token65 naik 2.8x dari D=9 ke D=36 sementara token-acc datar (~68%) dan exact tak naik signifikan (lihat significance_report). Saturasi energi: depth lebih dalam, tak berbayar.
- **Maze**: energi-untuk-token60 naik 1.8x dari D=9 ke D=18; token-acc d9=85% >= d18=83% (depth dalam tak menaikkan akurasi). Pola IDENTIK dengan Sudoku => **saturasi energi tergeneralisasi lintas-task**.
- **Batas hardware**: Maze D=36 OOM pada 16GB (seq_len 900 x depth 36); konsisten dgn ceiling kalibrasi. Dilaporkan sebagai batasan, bukan kegagalan.

## Konvergensi (Sudoku h128 D=36, 80k step)

- best_exact=17.1875% (vs ~15.2% pada 30k step), final_exact=10.3516%, net=292.1095 Wh, energi-setuju 98.91%.
- Verdict: sedikit di atas plateau 15% (hanya +2.0 poin untuk 2.7x step & ~2.7x energi); exact-acc berosilasi tanpa tren naik. **Menguatkan saturasi**: budget jauh lebih besar nyaris tak membeli akurasi.

## Kesimpulan untuk paper

Tesis *less-is-more* / saturasi energi kini didukung di **dua task** (Sudoku, Maze) plus uji konvergensi: pada anggaran tetap, menambah recursion depth menaikkan energi (1.7--3.0x) tanpa menaikkan akurasi. Energi tervalidasi silang 98.9--99.0% di semua run.

