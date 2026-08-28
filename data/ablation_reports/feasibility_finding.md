# Uji kelayakan akurasi nyata — TIDAK KONKLUSIF (run tidak setia ke resep TRM)

> **KOREKSI (penting):** judul awal "GATE GAGAL" ditarik. Setelah membaca resep kanonik
> TRM (`config/arch/trm.yaml` + README `pretrain_mlp_t_sudoku`), run uji ini ternyata
> **menyimpang besar** dari resep yang mencapai ~87%:
>
> | Param | Resep TRM | Run uji ini | Gap |
> |---|---|---|---|
> | hidden_size | **512** (~5M) | 128 (0.69M) | 7x |
> | epochs | 50000 | 10240 | ~5x |
> | ema | True | False | beda |
> | global_batch_size | 768 | 128 | 6x |
>
> Jadi plateau ~18% **TIDAK** membuktikan "augmentasi tak cukup" atau "pipeline tak bisa
> akurasi tinggi". Yang terbukti hanya: **h128 kecil + under-trained + tanpa EMA** mentok ~18%.
> Uji kelayakan yang sah harus memakai resep TRM yang setia (h512, 50k epoch, EMA, batch besar).

## Pertanyaan
Apakah keluar dari rezim toy (exact ~17%) cukup dengan menambah data via augmentasi
(pilot `num_aug=0` -> augmented `num_aug=1000`, ~1M contoh, seq_len tetap 81)?

## Hasil (run nyata, artefak: converge_out/progress_h128_d18_converge_s0.jsonl)
- Config: h128, D_eff=18, 80k step, data `sudoku-aug1k` (~1M contoh augmentasi).
- **best_exact = 18.4%** (final 4.7%, osilasi 5--18% sepanjang step 14k--80k).
- token-acc plateau ~68--69%. net energi 157.9 Wh, cross-val 98.96%.

## Vonis
**Augmentasi 1000x TIDAK menggeser plateau.** Trajektori h128_d18 di ~1M contoh
**identik** dengan pilot 1000 contoh (best 18.4% vs 17.2%). Bottleneck **bukan jumlah data**.

## Diagnosis (kandidat, belum diuji)
1. **Kapasitas**: h128 (0.69M) jauh di bawah TRM ~7M (≈h512); model terlalu kecil utk Sudoku-Extreme.
2. **Resep/optimizer**: kalibrasi memakai shim AdamW (adam-atan2 gagal compile di sm_120),
   lr=1e-4 tetap, `ema=False`. TRM asli (capai ~87%) pakai adam-atan2 + EMA + jadwal lr + ACT.
3. **Recursion config**: H_cycles/L_cycles & halting mungkin perlu mengikuti resep Sudoku TRM.

## Implikasi
Klaim apa pun tentang "akurasi" pada setup ini terbatas pada rezim ~18% exact. Untuk Q1
yang kredibel, harus dulu **mereproduksi angka headline TRM (akurasi tinggi)** pada SATU config,
membuktikan pipeline benar; baru frontier energi-akurasi bermakna. Augmentasi sudah dicoret
sebagai penyebab. Langkah berikut perlu konfirmasi user (config besar + resep TRM = mahal).
