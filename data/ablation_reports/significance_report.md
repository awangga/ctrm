> **[REGIME: TOY/PILOT — SUPERSEDED]** Report ini memakai data rezim-toy lama (h128/no-EMA, exact 0-15%). Hasil final manuscript memakai rezim akurasi nyata: lihat `isoflop_real_report.md`, `pd_plane_report.md`, `maze_feasibility_report.md`, `maze_depth_report.md`. Disimpan sebagai jejak historis (ablasi), bukan sumber klaim.

# Uji signifikansi: exact-accuracy vs recursion depth (hidden=128)

Sumber: `eksperimen/frontier/budget_out/progress_h128_d*_s*.jsonl` (15 run = 3 depth x 5 seed). Metrik: exact (puzzle) accuracy final, %. Bootstrap/permutation N=10000, seed RNG tetap.

## Ringkasan per-depth (mean +- sd antar seed)

| D_eff | n_seed | mean exact % | sd | min | max |
|---|---|---|---|---|---|
| 9 | 5 | 6.172 | 2.839 | 3.320 | 10.938 |
| 18 | 5 | 9.414 | 2.636 | 6.445 | 13.281 |
| 36 | 5 | 8.828 | 4.269 | 5.078 | 15.234 |

## 1. Bootstrap CI slope (exact% per oktaf depth, basis log2 D_eff)

- Slope teramati: **1.328 %/oktaf**
- 95% CI bootstrap: **[-0.831, 3.565]**
- Fraksi bootstrap dengan slope>0: **88.3%**
- CI MEMUAT 0 (tren TIDAK signifikan pada 0.05)

## 2. Permutation test (H0: depth tak berpengaruh)

- p (two-sided, |slope|): **0.2463** (gagal tolak H0 pada 0.05)

## 3. Welch t-test D=9 vs D=36

- t = -1.159, df ~ 7.0, mean(9)=6.172%, mean(36)=8.828%
- |t| <= ~2.0 -> beda TIDAK terdeteksi (n kecil; perlakukan sebagai indikatif)

## Vonis

Tren depth->exact **TIDAK signifikan**: selisih antar-depth masih dalam noise seed. Sesuai aturan integritas repo, dilaporkan sebagai **null/indikatif**, bukan hukum. Butuh lebih banyak seed untuk mempersempit CI.

