# Cost-model energi: sumber yang benar (microbench hardware), bukan hukum training toy — fase R (rek #1)

Audit Q1 (risiko #2) menandai: hukum energi b=0.825 di manuscript berasal dari `scale_out` (run training
rezim-TOY, iso-step). Ini memperbaikinya.

## Temuan jujur: iso-compute TAK BISA memfit hukum energi
Run rezim-nyata (32 run) dirancang **iso-compute**: dalam tiap task, params x D_eff x batch x steps
ditahan ~konstan. Karena energi total ~ compute total, energi juga ~konstan di dalam task -> tak ada
variasi compute untuk memfit hukum. Fit global energi-vs-compute lemah (R^2=0.57) dan within-task
tak bermakna (recipe_out b=-223). **Kesimpulan:** hukum energi-vs-compute TIDAK dapat diturunkan dari
data iso-compute; itu wajar dan bukan kegagalan.

## Cost-model yang benar: microbench kalibrasi (pengukuran hardware langsung)
Cost-model energi yang sahih = **energi per-langkah (J/step) sebagai fungsi FLOPs-per-langkah**, diukur
langsung di hardware pada batch & seq terkontrol (batch 128, seq 81), 14 config (`microbench_results.csv`).
Ini **regime-independent**: properti hardware/FLOP, bukan hasil-training.

- **J/step = 0.99 x (params_M x D_eff)^0.842**
- log-log **R^2 = 0.982**, bootstrap 95% CI eksponen **[0.75, 0.93]** (N=5000, RNG tetap)
- rentang: J/step 3.5–113, params_M x D_eff 6.2–181

Eksponen 0.84 **konsisten** dengan fit pilot `scale_out` (b=0.825, CI[0.74,0.88]) — dua sumber berbeda
(pengukuran per-step hardware vs run training pilot) memberi eksponen yang sama, memperkuat model. Tapi
sumber yang dilaporkan di manuscript kini = **microbench hardware**, bukan training toy.

## Implikasi manuscript
- Ganti klaim "\emph{from the calibration sweep} ... b=0.825 (R^2=0.995, MAPE 10.5%)" (yang sebenarnya
  angka scale_out toy) menjadi cost-model per-step dari microbench: **b=0.84, R^2=0.98, CI[0.75,0.93]**,
  ditegaskan sebagai pengukuran hardware terkontrol (regime-independent), dengan catatan bahwa hukum
  energi TIDAK difit ulang pada run iso-compute (by design compute konstan). Pilot sweep disebut sebagai
  cross-check yang konsisten (b=0.82), bukan sumber utama.
- Ini menutup risiko audit #2: tak ada lagi data toy yang menyangga klaim cost-model; sumbernya
  pengukuran hardware nyata.

---

## KOREKSI (2026-08-25, fase Z)

Batas atas CI bootstrap yang tertulis di atas, **[0.75, 0.93]**, tidak dapat direproduksi dari
`microbench_results.csv`. Dihitung ulang dengan tiga konvensi (RNG `random` seed 20260627 dengan
persentil-indeks, RNG yang sama dengan `np.percentile`, dan `np.random.default_rng(0)`), ketiganya
memberi batas atas **0.921--0.923**, yang membulat ke **0.92**. Batas bawah 0.7525 dan eksponen 0.8423
serta R^2 0.982 terkonfirmasi.

Nilai kanonik yang benar: **b = 0.842, 95% CI [0.75, 0.92], R^2 = 0.982, n = 14**.
Manuscript dan `zenodo/README.md` sudah dikoreksi. Angka ini kini diregenerasi otomatis oleh
`make_manuscript_figures.py`, yang mencetaknya bersama figur `fig_cost_model.pdf`.
