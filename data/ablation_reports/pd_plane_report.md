# Peta P↔D plane di REZIM AKURASI NYATA (Sudoku-Extreme, resep setia TRM h*+EMA)

Pertanyaan inti paper (compute-optimal recursion frontier): pada anggaran compute/energi setara,
ke mana sebaiknya anggaran dialokasikan, ke **lebar (P, hidden size)** atau ke **kedalaman recursion
(D_eff = H_cycles x L_cycles)**? Dijawab di rezim akurasi nyata (bukan toy 18%), resep setia TRM
(EMA, sudoku augmentasi ~1M), iso-compute (params x D_eff x batch x steps konstan).

Artefak: `recipe_out/recipe_summary.csv` + `progress_*.jsonl` + `pw_*.csv` + `emissions_*.csv`.
Semua angka telusur ke run nyata (CLAUDE.md aturan integritas #1).

## Dua sumbu, dua perilaku berbeda

### Sumbu DEPTH (h512 tetap, D_eff bervariasi) — 3 seed/config

| D_eff | batch | steps | best_exact (mean±std) | 3 seed (s0/s1/s2) | net Wh |
|---|---|---|---|---|---|
| **9**  | 192 | 50k | **62.4 ± 0.24** | 62.7/62.5/62.1 | 402 |
| 18 | 192 | 25k | 50.1 ± 1.37 | 51.6/50.4/48.2 | 358 |
| 36 | 96  | 25k | 36.3 ± 0.64 | 35.5/37.1/36.1 | 340 |

**Monoton: depth dangkal MENANG.** D9 > D18 > D36, selisih ~26 poin D9 vs D36. Tiap penggandaan
depth memangkas akurasi. Welch t: D9-vs-D18=12.5, D18-vs-D36=12.9 (jauh di luar variansi seed).

### Sumbu WIDTH (D18 tetap, hidden bervariasi) — iso-compute

| hidden | params | batch | steps | best_exact (mean±std) | seed | net Wh |
|---|---|---|---|---|---|---|
| 256 | ~kecil | 192 | 84.7k | 38.0 ± 1.11 | s0,s1,s2 | 503 |
| **512** | ~sedang | 192 | 25k | **50.1 ± 1.68** | s0,s1,s2 | 358 |
| 768 | ~besar | 128 | 19.3k | 43.0 ± 0.98 | s0,s1,s2 | 313 |

**Non-monoton: optimum di TENGAH (h512).** 38.0 -> 50.1 -> 43.0. Lebar terlalu kecil (h256)
kekurangan kapasitas; lebar terlalu besar (h768) under-trained pada anggaran tetap (step lebih
sedikit untuk muat param lebih banyak). h512 unggul +12.0 poin atas h256 dan +7.1 poin atas h768.
Rentang antar-seed tak tumpang-tindih (min h512=48.2 > max h768=43.9 > max h256=39.3).

## Sintesis: peta P↔D plane (best_exact %, mean)

```
              D_eff=9      D_eff=18     D_eff=36
  h256          -           38.0           -
  h512        62.4         50.1          36.3
  h768          -           43.0           -
```

Titik terbaik teramati: **(h512, D9) = 62.4%**. Dari titik ini:
- **Tambah depth (D9->D18->D36): akurasi TURUN tajam** (62->50->36). Depth aktif merugikan.
- **Geser width dari optimum (h512->h256 atau h512->h768): akurasi TURUN** (50->38 / 50->43 @D18).
  Width punya optimum interior, bukan "makin besar makin baik" maupun "makin kecil makin baik".

## Temuan utama (untuk manuscript)

1. **Anisotropi frontier: depth dan width TIDAK simetris.** Pada anggaran tetap, sumbu depth
   monoton-menurun (dangkal selalu menang), sedangkan sumbu width ber-optimum-interior. Ini
   membedakan recursion depth dari kapasitas lebar: keduanya bukan substitut setara.
2. **Compute-optimal condong ke recursion DANGKAL + width SEDANG.** Alokasi terbaik: depth kecil
   (lebih banyak langkah optimasi) pada lebar moderat (h512), bukan recursion dalam atau model lebar.
3. **Pembeda dari klaim TRM** (depth menggantikan parameter): substitusi itu **tidak compute-optimal**
   pada anggaran tetap untuk task ini. Depth tambahan tidak *earn its keep*.
4. **Energi (cross-val CodeCarbon vs nvidia-smi 99.2-99.8% di SEMUA run).** Catatan kejujuran:
   energi tidak persis iso antar-sel (314-505 Wh) karena per-step-time tak persis proporsional dan
   ada overhead tetap + eval; arah & besar efek tak berubah meski energi dinormalisasi.

## Catatan kejujuran (anti-overclaim)
- Sumbu width kini **3 seed penuh** (h256/h512/h768; lihat UPDATE fase T di bawah). Std kecil (1.0-1.7)
  dan gap antar-sel besar (7-12 poin) -> ordering kokoh & optimum-interior signifikan (Welch p=0.001/0.007);
  tidak diklaim sebagai hukum kuantitatif eksak.
- Plane masih jarang (5 sel terisi dari 3x3). Cukup untuk klaim kualitatif anisotropi +
  lokasi optimum; bukan permukaan respons penuh.

---

## UPDATE fase T: width axis dilengkapi ke 3-seed (rek #3) + baseline (rek #2)

Sumbu width @D18 kini **3 seed penuh** (sebelumnya h256/h768 hanya n=2):

| hidden | best_exact mean±std | 3 seed (s0/s1/s2) |
|---|---|---|
| 256 | **38.02 ± 1.11** | 39.26/37.70/37.11 |
| **512** | **50.07 ± 1.68** | 51.56/50.39/48.24 |
| 768 | **42.97 ± 0.98** | 43.95/42.97/41.99 |

**Optimum-interior h512 KINI SIGNIFIKAN** (n=3): Welch h512-vs-h256 t=10.34 p=**0.0010**; h512-vs-h768
t=6.31 p=**0.0065** (dua-duanya lolos Bonferroni 0.025). Anisotropi frontier (depth monoton, width
ber-optimum) kini se-rigor klaim depth. Baseline non-rekursif (51.4% @384Wh) lihat `baseline_report.md`:
recursion sepadan hanya bila dangkal (D9 62.4% menang; D18 seri; D36 36.3% kalah dari baseline).
