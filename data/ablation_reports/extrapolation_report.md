> **[REGIME: TOY/PILOT — SUPERSEDED]** Report ini memakai data rezim-toy lama (h128/no-EMA, exact 0-15%). Hasil final manuscript memakai rezim akurasi nyata: lihat `isoflop_real_report.md`, `pd_plane_report.md`, `maze_feasibility_report.md`, `maze_depth_report.md`. Disimpan sebagai jejak historis (ablasi), bukan sumber klaim.

# Ekstrapolasi hold-out: fit skala-kecil -> prediksi skala-besar

Sumber: `scale_summary.csv` (10 run, width 128-768 x depth 9/18, iso-step). Sumbu: compute = params_M x D_eff. Hold-out = 3 titik compute TERBESAR (tak dilihat saat fit). Bootstrap eksponen N=10000, RNG tetap. Galat = MAPE pada titik hold-out.

### Energi net (Wh)

- Power-law: Energi net (Wh) = A · compute^b ; **b = 0.825** (95% CI bootstrap [0.740, 0.879]), R²(fit, log-log) = 0.9952
- Fit pada 7 titik compute terkecil; hold-out 3 titik terbesar.

| held-out | compute (P·D) | aktual (Wh) | prediksi (Wh) | galat % |
|---|---|---|---|---|
| h768_d9 | 87.79 | 39.5826 | 43.4595 | 9.79 |
| h512_d18 | 90.52 | 40.8309 | 44.5739 | 9.17 |
| h768_d18 | 175.57 | 68.4771 | 76.9861 | 12.43 |

- **MAPE hold-out = 10.46%**

### lm_loss

- Power-law: lm_loss = A · compute^b ; **b = 0.032** (95% CI bootstrap [-0.038, 0.147]), R²(fit, log-log) = 0.0998
- Fit pada 7 titik compute terkecil; hold-out 3 titik terbesar.

| held-out | compute (P·D) | aktual (loss) | prediksi (loss) | galat % |
|---|---|---|---|---|
| h768_d9 | 87.79 | 0.8041 | 0.8986 | 11.75 |
| h512_d18 | 90.52 | 0.7984 | 0.8995 | 12.67 |
| h768_d18 | 175.57 | 0.8404 | 0.9190 | 9.35 |

- **MAPE hold-out = 11.26%**

## Vonis

- **Hukum energi** TERVERIFIKASI lewat ekstrapolasi: eksponen b=0.825 (CI [0.740, 0.879]), MAPE hold-out 10.46% (>=10% — perlu hati-hati). Cost-model energi dapat dipakai memprediksi konsumsi skala lebih besar.
- **lm_loss vs compute**: eksponen b=0.032 (CI [-0.038, 0.147]), MAPE hold-out 11.26%. Dilaporkan apa adanya; galat lebih besar (rentang/sampel terbatas).

