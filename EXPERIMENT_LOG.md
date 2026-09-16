# Jurnal Eksperimen — Frontier Energi-Akurasi Recursion (TRM)

> Catatan kronologis tiap langkah eksperimen: tujuan, config, hasil terukur, keputusan +
> alasan, commit, dan artefak. **Bahan langsung penulisan manuscript & tabel ablasi.**
> Aturan: tiap entri baru ditambahkan di bawah; jangan menulis ulang sejarah, koreksi
> ditandai eksplisit. Angka HARUS telusur ke artefak (aturan integritas #1).
> Lingkungan: RTX 5060 Ti 16GB, torch 2.11+cu128, optimizer shim AdamW (adam-atan2 gagal
> compile di sm_120). Data Sudoku via TRM `build_sudoku_dataset.py`.

## Ikhtisar fase

| Fase | Tujuan | Artefak | Commit |
|---|---|---|---|
| A. Kalibrasi | microbench biaya/step per (hidden,D) | `kalibrasi/microbench_results.csv` | (awal) |
| B. Iso-FLOP 2D | akurasi vs (P,D) pada compute setara | `isoflop_out/` | 105d2b1 |
| C. Budget-resolved | depth sweep 30k step, h128 | `budget_out/` | 00f531c |
| D. Replikasi seed | CI atas crossover saturasi | `budget_out/*_s*` | 1193296, 6f47b0f |
| E. Signifikansi+konsolidasi | bootstrap CI, dataset ablasi | `ablation/` | 49a436a |
| F. 5-seed + cross-val energi | CodeCarbon vs nvidia-smi | `budget_out/`, `energy_xval.csv` | 1c3156a |
| G. Scale sweep + ekstrapolasi | rentang 28x, hold-out + CI eksponen | `scale_out/`, `extrapolation_report.md` | 55fef20 |
| H. Metrik utama | Joules-to-target + Pareto | `joules_to_target.*` | 11bebdd |
| I. Maze + konvergensi | generalitas lintas-task | `maze_out/`, `converge_out/` | 2add787 |
| J. Audit | bongkar kelemahan hasil | (analisis) | — |
| K. Uji kelayakan augmentasi | keluar plateau 17%? | `sudoku-aug1k`, `feasibility_finding.md` | 0caf4b4, 1c5a90c |
| L. Smoke resep-setia | h512+EMA bisa akurasi tinggi? | `recipe_out/` | (berjalan) |

---

## Langkah detail

### A. Kalibrasi (microbench)
Ukur ms/step, J/step, peak-mem per (hidden∈{128..768}, D∈{9,18,36}). Temuan: params∝hidden²,
invariant terhadap depth; J/step naik bersih dgn width & depth; ceiling 16GB ≈ 10M params.
Artefak: `eksperimen/kalibrasi/microbench_results.csv`.

### B. Iso-FLOP 2D (preliminary)
Grid 3×3 (h128/256/384 × D9/18/36), compute~setara via step ∝ 1/biaya. Net energi ~10-11.5 Wh.
**Hasil:** pada budget pendek, config terkecil-terdangkal (h128 D9) terbaik (5.47% exact, 67.4% token);
menambah depth/width menurunkan akurasi. Artefak: `isoflop_out/isoflop_summary.csv`. Commit 105d2b1.
Catatan audit (fase J): energi iso-FLOP kurang reliabel (trace daya lintas-config).

### C. Budget-resolved depth (h128, ~30k step)
D9/18/36, 15 checkpoint, histori per-step penuh. Energi naik ~linear dgn depth (33.7/59.6/109.8 Wh).
Single-seed tak stabil. Commit 00f531c.

### D. Replikasi 3-seed
**Hasil:** exact 7.2±2.6 / 9.2±2.9 / 10.8±3.8% (D9/18/36). Mean naik lemah dgn depth tapi rentang
seed tumpang tindih. **Koreksi kejujuran (6f47b0f):** crossover TIDAK signifikan; manuscript di-soften.

### E. Signifikansi + konsolidasi (commit 49a436a)
`sig_test.py`: slope +1.79%/oktaf, 95% CI bootstrap [-1.19, 4.61] (memuat 0), permutation p=0.26.
**Vonis: tren depth→exact TIDAK signifikan.** `consolidate_ablation.py`: dataset ablasi + learning curves.

### F. 5-seed lengkap + cross-val energi (commit 1c3156a)
Tambah seed 3,4 (`run_xval_seeds.py`) dgn CodeCarbon+nvidia-smi serentak. 5-seed: 6.2/9.4/8.8%
(D9/18/36) — **non-monoton**, puncak D18, tren tetap tak signifikan (CI [-0.83,3.57]).
**Cross-val energi: 6/6 LOLOS, rerata 98.90% setuju** (`energy_xval_report.md`).
Catatan audit: kedua metode baca pynvml → konsistensi pembukuan, bukan sensor independen.

### G. Scale sweep 2D + ekstrapolasi hold-out (commit 55fef20)
`run_scale_sweep.py`: width {128..768} × D{9,18}, iso-step 4000. Compute P×D 6.25→175.6 (**28×**).
`fit_extrapolation.py` (fit 7 titik kecil → prediksi 3 besar yg ditahan):
- **Hukum energi:** b=0.825, 95% CI [0.740,0.879], R²=0.995, **MAPE hold-out 10.5%** → prediktif.
- **lm_loss:** b=0.032, R²=0.10, MAPE 11.3% → DATAR (iso-step: model besar under-trained).
Cross-val energi 10/10 = 98.89%.

### H. Metrik utama Joules-to-target (commit 11bebdd)
`joules_to_target.py` dari histori existing. **Wh-to-token65 Sudoku: D9=7.8, D18=11.9, D36=22.0**
(depth 2.8× lebih boros utk target sama). Termurah h128_d9 @6.43 Wh.

### I. Maze (task-2) + konvergensi (commit 2add787)
Maze-30x30-hard, h128, 3 seed, 8k step. **d9 token=84.5% ≥ d18=83.4%** (depth tak naikkan akurasi);
energi d9=9.0→d18=15.8 Wh (1.8×). **d36 OOM 16GB** (seq_len 900×depth 36). exact=0 (maze sulit).
Konvergensi h128_d36 80k step: best_exact 17.2% (vs 15.2%@30k), osilasi tanpa tren → saturasi.
**Kesimpulan saat itu: saturasi energi tergeneralisasi 2 task.**

### J. AUDIT hasil (atas permintaan user — pembongkaran skeptis)
Temuan kritis:
1. **Joules-to-target sebagian MEKANIS:** semua depth capai target di step SAMA (~6000 Sudoku,
   ~500 Maze) → selisih energi murni biaya per-step, bukan trade-off akurasi.
2. **Target τ nyaris setara baseline:** token Sudoku mulai 59%, target 65 = +6 poin.
3. **Rezim near-floor:** sudoku-pilot 1000 puzzle, exact mentok 15-17% (TRM asli ~87%).
4. **Maze nyaris tak berkontribusi:** exact=0 selalu; token 84% = +20 atas floor majority-class 64%.
5. **Cross-val energi ~tautologis** (pynvml sama).
**Vonis audit: hasil bersih & jujur tapi tipis/mekanis; BELUM Q1. Akar: data toy + perbandingan
equal-step yg tak adil bagi depth.**

### K. Uji kelayakan augmentasi (commit 0caf4b4 → koreksi 1c5a90c)
Hipotesis: pilot `num_aug=0` penyebab plateau. Build `sudoku-aug1k` (subsample 1000 × aug 1000 ≈
1M contoh, seq tetap 81). Run h128_d18 80k step. **best_exact 18.4% = sama dgn pilot (17.2%).**
- Vonis awal: "GATE GAGAL, augmentasi tak cukup."
- **KOREKSI KEJUJURAN (1c5a90c):** run ini TIDAK setia resep TRM (h128 vs **h512**, 10240 vs
  **50000** epoch, ema False vs **True**, batch 128 vs 768). Plateau 18% hanya membuktikan
  model-kecil+under-trained mentok, BUKAN bahwa pipeline tak bisa akurasi tinggi. Klaim ditarik.

### L-bis. Smoke resep-setia REPRODUKSI — TERVERIFIKASI dari artefak (commit di repo)
Setelah rebuild environment durable (fase M), smoke diulang dgn runner menulis artefak LANGSUNG
ke repo `eksperimen/frontier/recipe_out/` (di-commit berkala). Hasil **terverifikasi dari artefak**:
- Config: h512, D_eff=18 (H3/L6), EMA=True, batch=192, data sudoku-aug1k (~1M), 25k step (wall 8428s).
- **best_exact = 51.6%** (step ~23k), final_exact 49.0%, token 81.9%, **net 360.5 Wh**,
  cross-val energi (smi vs CodeCarbon) **99.27%**. Artefak: `recipe_out/progress_h512_d18_recipe_b192_s0.jsonl`
  + pw + emissions + recipe_summary.csv (commit 1614863 dst).
- Trajektori naik monoton menembus plateau h128: 6k=12.5%, 10k=24.6%, 15k=38.5%, 20k=47.5%, 23k=51.6%.
- **MELAMPAUI observasi live yg hilang (44%)** → reproduksi sukses & konsisten.

**Vonis fase L (final):** resep setia (h512+EMA, ~5M params) JELAS keluar dari rezim toy 18% dan
mencapai akurasi nyata (~50% exact pada eval-512). Pipeline TERBUKTI benar; h128 sebelumnya
under-capacity. **Jalur ke rezim akurasi-nyata layak.** Estimasi resep PENUH (50k epoch, batch 768
via grad-accum) jauh lebih panjang (~1-2 hari) dan diharap mendekati ~87% TRM; perlu konfirmasi
sebelum dijalankan. Catatan biaya: 360 Wh/run h512 25k-step (≈10× h128) → iso-FLOP grid penuh mahal.

### L. Smoke resep-setia (TERPUTUS — scratchpad terhapus, artefak HILANG)
`run_recipe.py`: h512, D18, EMA=True, batch 192, data sudoku-aug1k, target 25k step.
**Observasi LIVE selama run (terpantau di terminal, BUKAN dari artefak tersimpan):** exact-acc
memanjat monoton menembus plateau 18% h128 — step 6k=14.5%, 10k=29.5%, 15k=40.4%, **18k=44.1%**
(token 80.3%), masih naik. Gate (>40%) efektif LOLOS: resep setia (h512+EMA) jelas mencapai
rezim akurasi nyata, mengonfirmasi h128 sebelumnya under-capacity (pipeline tidak rusak).

**INSIDEN (dicatat jujur):** sebelum run selesai & artefak sempat disalin ke repo, **scratchpad
sesi di `/tmp` terhapus** (cleanup saat sesi berganti `63edeaef`→`1abe76a1`). HILANG: seluruh
environment TRM (kode, venv torch 2.11+cu128, dataset `sudoku-aug1k`) + artefak `recipe_out/`
(progress/pw/emissions). Repo git aman (di disk).

**Konsekuensi integritas (#1):** angka 44% di atas adalah observasi live yang TIDAK terverifikasi
artefak → **tidak boleh disitir sebagai hasil** sampai di-run ulang. Status: sinyal kuat bahwa
jalur akurasi-nyata layak, tetapi WAJIB direproduksi.

**Pelajaran → tindakan:** runner harus menyalin artefak ke repo **secara berkala saat berjalan**
(bukan hanya di akhir), karena scratchpad `/tmp` fana. Lihat [[trm-experiment-env]].

### M. Rebuild environment (DIPERLUKAN sebelum lanjut)
Untuk melanjutkan run apa pun: clone ulang TRM, buat venv (torch cu128; adam-atan2 gagal compile
→ shim AdamW), rebuild dataset (`sudoku-aug1k` via build_sudoku_dataset.py subsample 1000 aug 1000).
Lalu reproduksi smoke resep-setia agar 44% menjadi hasil tersitir, baru lanjut ke iso-FLOP rezim
akurasi nyata.

### N. Iso-FLOP depth frontier REZIM AKURASI NYATA (h512+EMA) — temuan Q1 utama
Tiga config iso-compute (ratio 1.00) di sudoku-aug1k, resep setia. Artefak repo recipe_out + commit berkala.
| D_eff | batch×steps | best_exact | token | net Wh | cross-val | Wh-to-exact50 |
|---|---|---|---|---|---|---|
| 9  | 192×50k | **62.7%** | 86.3 | 405 | 99.46% | 259 |
| 18 | 192×25k | 51.6% | 81.9 | 360 | 99.27% | 317 |
| 36 | 96×25k  | 35.5% | 76.8 | 341 | 99.63% | tak tercapai |
**Vonis:** monoton, depth DANGKAL menang telak (D9 +27 poin atas D36). Di rezim akurasi nyata depth
AKTIF MERUGIKAN pada iso-compute (kontras tajam dgn rezim toy yg tak signifikan). Frontier compute-optimal
condong ke recursion dangkal + lebih banyak step. Energi tak persis iso (D9 405 vs D36 341 Wh) tapi arah
& besar efek kokoh. Report: ablation/isoflop_real_report.md. Ini hasil headline paper.

### N-bis. Replikasi 3-seed frontier (celah n=1 ditutup)
9 run (D9/D18/D36 × seed0/1/2), h512+EMA iso-compute, artefak repo recipe_out + commit berkala.
best_exact mean±std: D9=62.4±0.30, D18=50.1±1.68, D36=36.3±0.79 (%). Ordering D9>D18>D36 di
SETIAP seed (rentang tak overlap); Welch t=12.5 & 12.9. Energi cross-val 99.5-99.7%. **Frontier
compute-optimal-dangkal KOKOH & signifikan antar-seed — celah replikasi (fase J/audit) tertutup.**
Report: ablation/isoflop_real_report.md.

---

## Fase O — Width-axis Sudoku + peta P↔D plane (rezim akurasi nyata)
**Tanggal:** 2026-06-30. **Tujuan:** lengkapi sumbu width (hidden 256/512/768 @D18 iso-compute)
untuk melengkapi sumbu depth (fase N/N-bis), lalu sintesis jadi peta P↔D plane akurasi nyata.

**Config (resep setia TRM, EMA, sudoku-aug1k, run_recipe.py):**
- Width @D18: h256 (b192, 84.7k step), h512 (b192, 25k step, dari depth-axis), h768 (b128, 19.3k step).
  Iso-compute: params x D_eff x batch x steps konstan. Seed 0,1 (h512: seed 0,1,2).

**Hasil terukur (telusur: recipe_out/recipe_summary.csv + progress_/pw_/emissions_):**
- WIDTH @D18 best_exact mean±std: h256=38.5±0.78 (n=2, 505Wh) | h512=50.1±1.37 (n=3, 358Wh) |
  h768=43.5±0.49 (n=2, 314Wh). **Optimum interior di h512** (non-monoton 38->50->43).
- DEPTH @h512 (rekap fase N-bis): D9=62.4±0.24 > D18=50.1±1.37 > D36=36.3±0.64. Monoton dangkal-menang.
- Cross-val energi 99.2-99.8% di semua run.
- Titik plane terbaik: (h512,D9)=62.4%.

**Temuan:** anisotropi frontier — depth monoton-menurun, width ber-optimum-interior. Compute-optimal
= recursion dangkal + width sedang. Memperkuat klaim: depth tambahan tidak earn-its-keep; substitusi
depth<->params (klaim TRM) tidak compute-optimal pada anggaran tetap untuk task ini.

**Insiden teratasi (lihat fase sebelumnya):** git corrupt (objek kosong dari commit bersamaan) pulih
dari remote + flock pada watcher; chain width sempat mati saat recovery, di-relaunch via run_widthaxis2.sh.

**Keputusan:** sumbu width n=2 (h256/h768) cukup untuk klaim kualitatif optimum-interior (std kecil,
gap besar); tidak diklaim hukum kuantitatif eksak. **Lanjut:** task-2 Maze (feasibility) untuk
generalitas lintas-task sebelum integrasi manuscript.

**Artefak:** ablation/pd_plane_report.md, recipe_out/{progress,pw,emissions}_h{256,768}_d18_*_s1.*,
recipe_summary.csv (baris h256_d18_s1, h768_d18_s1).

---

## Fase P — Feasibility task-2 Maze (rezim akurasi nyata)
**Tanggal:** 2026-06-30. **Tujuan:** konfirmasi pipeline resep-setia TRM jalan di task KEDUA (Maze,
seq900) sebagai prasyarat generalitas lintas-task (hindari single-task).

**Config:** resep setia EMA, HIDDEN=256 D_eff=18 batch=48 ~7916 step seed=0, data=maze-aug (run_recipe.py).

**Hasil terukur (telusur: maze_real_out/recipe_summary.csv + progress_/pw_/emissions_):**
- token-accuracy first/max/last = 23.7%/86.7%/85.5% (plateau ~85% sejak ~step 2000).
- exact-match = 0.0% sepanjang 20 eval. net=209.4Wh, cross-val 99.66%, GPU ~7.3GB (muat 16GB, no OOM).
- wall=4829s (~1.34 jam).

**Verdict:** FEASIBLE & belajar (token 24->87%). exact=0 -> metrik discriminating lintas-task untuk
Maze = TOKEN-ACCURACY (exact rata-0, uninformative). Sudoku tetap pakai exact (36-62%, discriminating).

**Insiden minor:** run pertama gagal 2 detik (PATH venv belum di-export -> ModuleNotFoundError torch);
diperbaiki (export PATH venv/bin), baris cacat (wall=1.7s) dihapus dari CSV.

**Keputusan:** feasibility cukup utk lanjut. Grid depth-comparison Maze (D9/D18/D36 iso-compute, metrik
token-accuracy) DIUSULKAN, menunggu konfirmasi user (durasi ~10-36 jam tergantung seed/scale).

**Artefak:** ablation/maze_feasibility_report.md, maze_real_out/{progress,pw,emissions}_h256_d18_recipe_b48_s0.*,
recipe_summary.csv.

---

## Fase Q — Depth-comparison Maze iso-compute + verdict generalitas lintas-task
**Tanggal:** 2026-07-01. **Tujuan:** uji apakah temuan Sudoku (depth tak earns-its-keep) general ke
task kedua (Maze-Hard). Grid depth iso-compute @h256, 1 seed, metrik token-accuracy (exact=0 di Maze).

**Config (iso-compute D x batch x steps = 10.368M):** D9 b48 24k step, D18 b48 12k, D36 b24 12k.
resep setia EMA, data=maze-aug (run_mazedepth.sh).

**Hasil terukur (telusur: maze_depth_out/recipe_summary.csv + progress_/pw_/emissions_):**
- token% final: D9=83.85 (313Wh), D18=82.98 (302Wh), D36=85.28 (326Wh). exact=0 semua. cross-val 99.4%.
- Rentang token 82.98-85.28 -> gap 2.3 poin (dalam derau 1-seed), tak monoton signifikan.
- Bandingkan Sudoku exact: D9=62.4>D18=50.1>D36=36.3 (gap 26 poin, monoton dangkal-menang).

**Verdict:** efek recursion depth TASK-DEPENDENT. Sudoku: depth sangat menentukan (dangkal menang
telak). Maze: depth ~netral pada token-accuracy. Yang bertahan di KEDUA task: depth dalam tak pernah
MENANG signifikan -> less-is-more/depth-tak-earns-its-keep tetap sah. Bingkai kontribusi = PETA REZIM
lintas-task, bukan hukum universal. Keterbatasan jujur: Maze n=1, metrik token (proxy), gap 2.3 poin
belum diuji signifikansi -> laporkan sbg null-effect depth pd Maze-token.

**Keputusan:** grid Maze 1-seed cukup utk klaim kualitatif generalitas (task-dependent). Replikasi
seed Maze opsional bila reviewer minta uji signifikansi "datar". Lanjut: integrasi ke manuscript.

**Artefak:** ablation/maze_depth_report.md, maze_depth_out/{progress,pw,emissions}_h256_d{9,18,36}_*.*,
recipe_summary.csv.

---

## Fase Q-bis — Replikasi seed Maze depth (n=1 → n=3): uji signifikansi pola (2026-07-02)

**Tujuan:** menaikkan Maze depth-grid dari 1 seed ke 3 seed (setara Sudoku), menguji apakah pola
"datar" fase Q signifikan. Runner: run_mazedepth_seeds.sh (seed1,2) + run_mazedepth_seeds_resume.sh.

**Catatan operasional (kejujuran):** chain seed pertama MATI saat mesin reboot (Jul-1 20:57, D18_s2
baru step 1770/11979). 4/6 baris selamat (D9/D18/D36 s1 + D9 s2, artefak durable di repo, tak hilang).
Sisa D18_s2 + D36_s2 di-restart pasca-reboot via run_mazedepth_seeds_resume.sh (rm progress parsial
D18_s2, ditulis-ulang tag sama). Semua 6 run replikasi selesai bersih.

**Config (iso-compute D x batch x steps = 10.368M):** D9 b48 24k, D18 b48 12k, D36 b24 12k, @h256 EMA.

**Hasil terukur 3-seed (telusur: maze_depth_out/recipe_summary.csv, 9 run):**
- D9  = **83.12 ± 0.67%** token (net 316.6 Wh)
- D18 = **83.36 ± 0.34%** token (net 306.1 Wh)  <- paling hemat energi
- D36 = **84.68 ± 0.65%** token (net 327.7 Wh)  <- +7% energi vs D18
- cross-val CodeCarbon vs nvidia-smi 99.4–99.95% (9 run).

**Uji signifikansi (Welch, Student-t exact):**
- D36 vs D9:  diff +1.56, t=2.90 df=4.0, **p=0.044** → NS setelah Bonferroni (thr 0.0167).
- D36 vs D18: diff +1.32, t=3.12 df=3.0, **p=0.052** → NS.
- D18 vs D9:  diff +0.24, t=0.55 df=2.9, p=0.619 → NS.
- ANOVA satu-arah: **F(2,6)=6.46, p=0.032** (marginal).

**Verdict (dikoreksi dari fase Q):** efek recursion depth pada Maze-token **lemah/near-flat**: gap
terbesar 1.56 poin = **~17× lebih kecil** dari Sudoku (26 poin, t>12). TAK satu pun kontras pasangan
lolos koreksi multiple-comparison; ANOVA hanya marginal. Ada **tren-lemah dalam-unggul** (D36 tertinggi),
**arah berlawanan** Sudoku, tetapi tidak kokoh. Yang bertahan tegas di KEDUA task: **recursion dalam tak
pernah jadi alokasi energy-optimal** — di Sudoku ia kalah akurasi & boros; di Maze keunggulan D36 tak
signifikan DAN menuntut +7% energi (D18 titik hemat). *less-is-more/depth-tak-earns-its-keep* tetap sah.

**Keputusan:** replikasi seed selesai; klaim Maze kini **efek-depth lemah terukur (n=3, diuji)**, bukan
lagi "null n=1". Kontribusi = **peta rezim energi-akurasi lintas-task** (depth task-dependent dalam arah
& besaran; tak pernah energy-optimal). SIAP integrasi manuscript.

**Artefak:** ablation/maze_depth_report.md (diperbarui 3-seed), maze_depth_out/{progress,pw,emissions}_
h256_d{9,18,36}_recipe_*_s{0,1,2}.*, recipe_summary.csv; runner run_mazedepth_seeds.sh +
run_mazedepth_seeds_resume.sh (di trm-env/).

---

## Fase R — Audit Q1 + pelengkapan zenodo/eksperimen + task-3 ARC-AGI-1 (mulai 2026-07-02/03)

**Tujuan:** (a) audit kelayakan Q1 Elsevier (*Sustainable Computing*); (b) lengkapi paket `zenodo/`
& folder `eksperimen/` agar konsisten rezim-nyata; (c) tambah task-3 **ARC-AGI-1** (keputusan user:
"jalankan ARC dulu" sebelum finalisasi klaim manuscript) di bawah protokol iso-compute yang sama.

**(a) Audit Q1 (ringkas, telusur ke report rezim-nyata):** substansi sudah level-Q1 untuk venue ini
(energi terukur cross-val 99.2-99.95%, klaim tajam depth-tak-pernah-energy-optimal lintas-task,
replikasi 3-seed + uji signifikansi). Celah WAJIB: (1) manuscript masih memuat angka toy + bergaya
proposal -> perlu integrasi rezim-nyata; (2) abstrak menjanjikan ARC yang belum dijalankan; (3) hukum
energi b=0.825 berasal rezim-toy (scale_out) -> pisahkan sbagai validasi cost-model, bukan hukum
akurasi; (4) Maze pakai token-acc (justifikasi); (5) width n=2, baseline HRM tak dijalankan; (6) 2 entri
.bib tanpa DOI (YangJoules2025, OliveiraFilho2025). Keputusan: tutup #2 dgn menjalankan ARC.

**(b) Pelengkapan folder (housekeeping, bukan hasil baru):**
- zenodo/data: disalin `recipe_out/` (Sudoku 3-seed), `maze_depth_out/` (Maze 3-seed), `maze_real_out/`
  (kecuali `log_*.txt` stdout) + `ablation_reports/` (4 report current). Data toy lama (`budget/frontier/
  isoflop_out`) diberi `PILOT_NOTE.txt` (superseded).
- zenodo/code: ditambah runner nyata `run_recipe.py` + `sig_test.py` + analisis (`joules_to_target.py`,
  `fit_extrapolation.py`, `energy_xval_report.py`, `analyze_crosstask.py`, `consolidate_ablation.py`,
  dll) yang sebelumnya hilang. `requirements.txt` +scipy.
- zenodo README/PROTOCOL/.zenodo.json diperbarui: dari "kalibrasi+protokol saja" -> mencakup hasil
  rezim-nyata (angka Sudoku/Maze 3-seed), pemisahan tegas rezim toy vs nyata, PROTOCOL §10 baru (resep
  setia iso-compute). 6 report ablasi toy lama di `eksperimen/.../ablation/` diberi banner
  "[REGIME: TOY/PILOT — SUPERSEDED]" (jejak historis, bukan sumber klaim).

**(c) ARC-AGI-1 setup + smoke (telusur ke artefak):**
- Data mentah: clone `fchollet/ARC-AGI` (400 training + 400 evaluation), dirakit ke format Kaggle-combined
  (`arc_raw/arc-agi_{training,evaluation}_{challenges,solutions}.json`) di `trm-env/` (durable, tak
  di-commit: besar & rebuildable).
- Build augmented: `dataset.build_arc_dataset --subsets training evaluation --num-aug 1000` ->
  `data/arc1-aug1k` (800 puzzle, 730.515 augmented IDs, 5.4GB, seq_len=900, vocab=12).
  Deviasi jujur dari resep 45%: subset `concept` (ConceptARC) TIDAK disertakan (hanya training+evaluation
  ARC-AGI-1) demi kesederhanaan; kita tak mengejar SOTA, hanya perbandingan iso-compute lintas-depth.
- Perbaikan runner: `run_recipe.py` `GROUPS` kini env-overridable (ARC train=800 groups, bukan 1000) agar
  mapping epoch<->step akurat; iso-compute ratio antar-depth tetap terjaga.
- **Smoke (h256 D18 b48 ~1500 step, GROUPS=800): FEASIBLE.** GPU **8.0 GB** (puzzle_emb 730k muat, NO OOM
  di 16GB), lm_loss mulai 2.64 turun, training jalan (~147W). Artefak: `arc_smoke_out/`.
- Rencana grid: mirror desain Maze (iso-compute, h256, token-accuracy krn exact diperkirakan ~0):
  D9 b48 24k / D18 b48 12k / D36 b24 12k, seed 0 dulu lalu replikasi. Metrik utama = token-acc.

**Status:** smoke ARC jalan (konfirmasi "belajar" via eval token-acc pending); folder di-commit.

### Fase R (lanjutan) — KOREKSI eval-cost & step-mapping ARC, grid seed0 diluncurkan (2026-07-03)

Dua kendala tersingkap saat smoke ARC (dicatat untuk kejujuran & ablasi):

1. **Eval-cost bomb.** Test ARC augmented = **368.150 puzzle** (vs Sudoku/Maze 512). Eval penuh 16-step
   recursion atas 368k = tak feasible & merusak akuntansi energi (persis pitfall "eval mendominasi" di
   manuscript §5). **Solusi:** truncate test ARC ke **512** (`data/arc1-aug1k-e512`, train di-symlink,
   test 512, `make_small_eval` gaya). Konsisten aturan eval-subset Sudoku/Maze.
2. **Step-mapping meleset 3.85x.** `pretrain.py:219`: `total_steps = epochs x total_groups x
   mean_puzzle_examples / batch`. ARC `mean_puzzle_examples=3.85` (Sudoku/Maze=1.0), sedangkan heuristik
   `run_recipe` SPE=GROUPS/BATCH mengasumsikan mean_ex=1 -> step aktual 3.85x label. **Solusi:** set
   `GROUPS=3080` (=800 x 3.85) via env, sehingga STEPS = step-aktual. Iso-compute antar-depth tetap sah
   (mean_ex konstan lintas-depth); rasio D9:D18 = 2:1 terjaga.

**Smoke final (h256 D18 b48, e512, GROUPS=3080):** FEASIBLE (GPU ~8GB, no OOM), eval-512 cepat (tak
memblok), token-acc **8.35% @step373** (belajar dari baseline), **exact=0** -> metrik utama = token-acc
(seperti Maze). lm_loss 2.64 -> 1.87 turun.

**Grid seed0 diluncurkan** (`run_arcdepth.sh`, OUT `arc_depth_out/`): D9 b48 ~22.5k step, D18 b48 ~11.2k,
D36 b24 ~9.6k (iso-compute ~9.7M; D36 -14% krn pembulatan ei, dilaporkan apa adanya spt Maze). Metrik
token-acc, energi cross-val CodeCarbon vs nvidia-smi. Watcher cron `watch_arcdepth.sh` tiap 20mnt.
Deviasi resep jujur: subset ConceptARC TIDAK disertakan (hanya ARC-AGI-1 training+evaluation); kita tak
mengejar 45% SOTA, hanya perbandingan iso-compute lintas-depth. Hipotesis (diuji): apakah pola Maze
(near-flat) atau Sudoku (dangkal-menang) yang berlaku di ARC -> memperkuat peta rezim lintas-task.

### Fase R (hasil) — ARC-AGI-1 depth seed0: dangkal-menang MONOTON (2026-07-03)

**Hasil seed0 (telusur: arc_depth_out/recipe_summary.csv):** token-accuracy final iso-compute @h256:
- D9  = **36.46%** (net 292.2 Wh, xval 99.11%, wall 6755s)
- D18 = **29.83%** (net 284.9 Wh, xval 99.15%)
- D36 = **26.11%** (net 283.1 Wh, xval 99.21%)
exact=0 semua (ARC sulit di anggaran 1-GPU) -> metrik token-acc. **Monoton dangkal-menang, gap 10.35
poin** D9 vs D36. Energi ~iso (283-292 Wh), D9 sedikit boros tapi unggul akurasi -> Joules-to-target
memihak dangkal.

**Verdict lintas-task (3 task, rezim nyata):** Sudoku (exact, gap 26, t>12) & ARC (token, gap 10.35)
= **monoton dangkal-menang**; Maze (token, gap 1.56 NS) = near-flat. **2 dari 3 task dangkal-menang**
-> generalitas menguat. Klaim inti bertahan di KETIGA: recursion dalam tak pernah earns-its-keep /
tak pernah compute-energy-optimal pada anggaran tetap. Kontribusi = peta rezim (kuat di task
diskriminatif, datar di task jenuh). Detail: ablation/arc_depth_report.md.

**Replikasi seed1,2 DILUNCURKAN** (run_arcdepth_seeds.sh, GROUPS=3080, e512) untuk uji signifikansi
seperti Maze/Sudoku. Watcher cron watch_arcdepth aktif. Keterbatasan jujur: n=1 seed0 (menunggu 3-seed);
gap 10.35 >> derau seed (0.2-1.7) -> diperkirakan signifikan. Deviasi: tanpa ConceptARC, test-512,
bukan kejar 45% SOTA (anggaran 1-GPU << 4xH100); yang dibandingkan ordering lintas-depth iso-compute.

### Fase R-final — ARC 3-seed: KOREKSI KEJUJURAN verdict (2026-07-04)

**Replikasi seed1,2 selesai (9 run total).** Token% 3-seed (telusur arc_depth_out/recipe_summary.csv):
- D9  = **32.78 ± 3.23** (36.46/30.40/31.49), net 291.9 Wh
- D18 = **31.19 ± 4.68** (29.83/27.34/**36.41** outlier), net 283.6 Wh
- D36 = **25.98 ± 0.61** (26.11/25.31/26.51), net 280.6 Wh

**Uji signifikansi:** D9-vs-D18 p=0.657 (seri), D9-vs-D36 p=0.063, D18-vs-D36 p=0.192, ANOVA
F(2,6)=3.49 p=0.099 -> **SEMUA NS setelah Bonferroni**.

**KOREKSI KEJUJURAN:** klaim seed0 (fase R-hasil) "ARC monoton dangkal-menang, gap 10.35 poin, spt
Sudoku" **DIBATALKAN**. Gap seed0 di-inflate satu seed (D18_s2=36.41 outlier). Verdict 3-seed yang benar:
**D9 ≈ D18 (seri) > D36 (tren), TAK signifikan di n=3**; variansi ARC besar (sd 3-5) vs Sudoku/Maze
(sd 0.2-1.7). ARC ternyata mirip **Maze (efek depth lemah/NS)**, bukan Sudoku.

**Verdict lintas-task DIKOREKSI:** hanya **Sudoku** = efek depth kuat & signifikan (dangkal menang telak
gap26 t>12). **ARC & Maze = lemah/NS di n=3.** Benang merah bertahan di KETIGA: recursion depth **tak
pernah MENANG signifikan** pada anggaran tetap; deep (D36) tak pernah compute/energy-optimal (di ARC
D36 numerik terendah & termurah energi tapi akurasi paling rendah). Kontribusi = **peta rezim** (kekuatan
keuntungan-dangkal task-dependent), bukan hukum monoton universal. Ini pembingkaian jujur diagnosis->
generalisasi, lebih tahan reviewer Q1.

**Status:** ARC 3-seed SELESAI. Cron watch_arcdepth dihapus. Semua 3 task (Sudoku/Maze/ARC) rezim-nyata
tuntas dgn replikasi 3-seed + uji signifikansi. SIAP integrasi manuscript 3-task. Report:
ablation/arc_depth_report.md (bagian 3-seed + KOREKSI).

### Fase S — Integrasi manuscript 3-task (2026-07-04)

Manuscript `main.tex` diintegrasi dari rezim-toy/proposal ke **hasil rezim-nyata 3-task** (semua angka
telusur ke artefak; tak ada lagi angka toy sebagai temuan):
- Abstrak: dari "We propose" -> pelaporan hasil (Sudoku 62.4 vs 36.3 exact t>12; Maze/ARC within-noise n=3).
- Kontribusi: peta rezim lintas-task (depth tak pernah menang signifikan; kekuatan task-dependent).
- Section Results BARU (ganti Iso-FLOP/budget toy): (i) depth Sudoku dangkal-menang monoton + Joules-to-50%;
  (ii) width optimum-interior h512 (anisotropi); (iii) lintas-task Maze near-flat + ARC weak/NS (tabel+uji);
  (iv) sintesis peta rezim; (v) energi cross-val 99.2-99.95% + cost-model b=0.825 (dipisah dari accuracy-law).
- Threats diperluas jujur (skala, metrik exact vs token, n=3, ConceptARC dikecualikan, ARC<<SOTA, iso-compute
  bukan iso-energy, baseline HRM tak dilatih). Reproducibility: tiap angka telusur artefak + Zenodo.
- 3 figur rezim-nyata di-generate dari CSV (fig_sudoku_frontier, fig_width_optimum, fig_crosstask_depth).
- Sitasi: DOI OliveiraFilho2025 (Phoeni6, venue target) diverifikasi CrossRef -> 10.1016/j.suscom.2025.101172
  (vol 47, art 101172); YangJoules2025 (ICML) tak ada di CrossRef -> DOI pending (jujur). 39 sitasi resolved,
  0 undefined. Compile bersih: 16 halaman.

Sisa pra-submit (bukan blocker): email koresponden (\ead placeholder), DOI Zenodo camera-ready, opsional
figur energi tambahan, verifikasi DOI YangJoules2025 bila terbit di PMLR/DOI resmi.

### Fase T — Penguatan Q1: cost-model (rek#1), width n=3 (rek#3), baseline non-rekursif (rek#2) (2026-07-04)

Menutup 3 risiko audit Q1. Semua telusur ke artefak.

**Rek #1 — cost-model energi (tanpa run baru).** iso-compute tak bisa memfit hukum energi (by design
compute konstan; fit global R^2=0.57). Cost-model benar = microbench hardware: **J/step=0.99·(params×D)^0.84,
R^2=0.98, CI[0.75,0.93]** (regime-independent). Konsisten pilot (0.82). Manuscript §energy direframe.
Report: `ablation/energy_costmodel_report.md`.

**Rek #3 — width axis 3-seed.** Tambah h256_d18_s2 (37.11%) & h768_d18_s2 (41.99%). Width @D18 3-seed:
h256=**38.02±1.11** < h512=**50.07±1.68** > h768=**42.97±0.98**. **Optimum-interior h512 SIGNIFIKAN**:
Welch h512-vs-h256 p=0.0010, h512-vs-h768 p=0.0065 (lolos Bonferroni). `pd_plane_report.md` diperbarui.

**Rek #2 — baseline non-rekursif (matched-energy).** transformer 8-layer non-rekursif h512, 100k step,
**51.37% exact @ 383.6 Wh** (token 82.2%, xval 99.59%). Verdict: **recursion sepadan HANYA bila dangkal** —
TRM D9 (62.4%, Wh-to-50%=259) menang telak +11 poin & 1.5x lebih hemat energi; TRM D18 (50.1%) ~seri;
TRM D36 (36.3%) KALAH dari baseline. Nilai recursion terkonsentrasi di depth dangkal. `baseline_report.md`.

**Integrasi manuscript + regen figur + recompile + commit.** Runner: run_recipe.py kini dukung
ARCH=transformers_baseline (cmd non-rekursif). Chain: run_rec3_chain.sh. Semua di recipe_out/.

### Fase T-bis — Replikasi baseline non-rekursif ke n=3 (2026-07-04)

Baseline (transformer non-rekursif h512, matched-energy) diulang seed 1,2 -> **n=3**.
Best_exact (telusur recipe_out/recipe_summary.csv): s0=51.37, s1=50.39, s2=47.27 -> **49.67 ± 2.14%**
@ 384.0 Wh (xval 99.6%). Welch vs baseline: **TRM D9(62.4) MENANG p=0.008**; D18(50.1) seri p=0.82;
**TRM D36(36.3) KALAH p=0.004**. Verdict kini terbukti dua-arah & signifikan (n=3): recursion sepadan
HANYA bila dangkal; recursion dalam SIGNIFIKAN lebih buruk daripada tanpa-rekursi pada energi setara.
Manuscript (Table tab:baseline + subsection + Threats) diperbarui ke mean±std 3-seed. baseline_report.md
jadi n=3. Menutup keterbatasan 'baseline n=1' dari audit.

### Fase U — Rekonsiliasi angka FINAL (konsistensi manuscript↔jurnal) (2026-07-05)

Verifikasi konsistensi manuscript vs jurnal ini. Manuscript, report ablation, dan zenodo memakai angka
FINAL berikut (semua telusur `recipe_summary.csv`, **sample std ddof=1**). Entri fase lebih awal yang
berbeda adalah **historis-tersuperseded** (jurnal append-only), BUKAN koreksi baru — dicatat di sini
agar pembaca tak bingung:

- **Sudoku depth (h512, 3-seed)** [fase N-bis]: D9=62.4±0.3 > D18=50.1±1.7 > D36=36.3±0.8; net
  402/358/340 Wh; Wh-to-50% 259/317; Welch t=12.5/12.9.
  *Superseded:* fase N seed0 (D9=62.7%, 405 Wh); rekap fase O memakai population std (0.24/1.37/0.64) —
  data sama, konvensi std beda; kanonik = sample std.
- **Width (@D18, 3-seed)** [fase T]: h256=38.0±1.1 < h512=50.1±1.7 > h768=43.0±1.0; Welch h512-vs-h256
  p=0.001, h512-vs-h768 p=0.007 (lolos Bonferroni). *Superseded:* fase O n=2 (38.5/43.5).
- **Maze token (3-seed)** [fase Q-bis]: 83.1/83.4/84.7; gap 1.56 (p=0.044); ANOVA F(2,6)=6.46 p=0.032;
  pairwise NS. *Superseded:* fase Q n=1 (gap 2.3).
- **ARC token (3-seed)** [fase R-final]: D9=32.8±3.2, D18=31.2±4.7, D36=26.0±0.6; ANOVA p=0.099, semua
  pairwise NS. *Superseded:* fase R-hasil seed0 (dangkal-menang gap 10.35).
- **Baseline non-rekursif (3-seed)** [fase T-bis]: 49.7±2.1% @384 Wh; Welch vs baseline: TRM D9 p=0.008
  (menang), D18 p=0.82 (seri), D36 p=0.004 (kalah). *Superseded:* fase T n=1 (51.37%).
- **Cost-model** [fase T]: J/step = α·(P·D_eff)^0.84, R²=0.98, CI[0.75,0.93] (microbench hardware).
- **Energi:** cross-val CodeCarbon vs nvidia-smi 99.1–99.95% tiap run; total ~14.6 kWh atas 37 run nyata.

Kesimpulan: manuscript konsisten dengan keadaan FINAL jurnal; tak ada entri final yang bertentangan.

### Fase V — Audit kelayakan submit Elsevier + perbaikan pra-submit (2026-08-24)

Tujuan: menilai apakah manuscript layak dikirim ke *Sustainable Computing: Informatics and Systems*
dan menutup semua celah yang bisa ditutup tanpa keputusan penulis. **Tidak ada run baru**: seluruh
angka dapat direkonstruksi dari artefak yang sudah di-commit.

**Temuan integritas (KOREKSI KEJUJURAN).**
1. Klaim `A = 99.1–99.95% pada SETIAP run` **terlalu luas**. Terukur: benar untuk **37 run
   faithful-recipe** (99.11–99.95), tetapi run pilot turun sampai **95.89%** (`maze_out/h128_d36_s0`);
   sembilan run `scale_out` dan enam `budget_out/energy_xval` juga di bawah 99.1. Manuscript kini
   men-scope klaim itu secara eksplisit ke rezim faithful-recipe di 4 tempat (abstrak, kontribusi,
   §protokol, §hasil) dan menyebut angka pilot apa adanya.
2. Agregat `14.6 kWh atas 70 run` **tidak cocok artefak**. Hitung ulang: **79 run dengan catatan energi
   per-run = 15.08 kWh**; faithful 37 run = 12.53 kWh (klaim 12.5 kWh COCOK). Manuscript diperbarui ke
   79 run / 15.1 kWh. Dikecualikan & dicatat: 9 run replikasi `budget_out` seed 0–2 punya akurasi tapi
   power log-nya ditulis tanpa sufiks seed sehingga saling menimpa, energi per-seed tak dapat
   diatribusikan.
3. `arc_smoke_out` (3 run pilot, 42.1 Wh) ikut dihitung tetapi **di-gitignore**. Baris ignore dihapus,
   folder di-commit, supaya total 79 run benar-benar dapat diaudit dari paket.

Skrip baru **`reconcile_totals.py`** (dukung `--data-root`) menghasilkan
`ablation/run_energy_reconciliation.md`. Diverifikasi memberi angka identik dari repo kerja maupun dari
paket Zenodo, jadi reviewer bisa meregenerasi agregat itu sendiri.

**Temuan integritas sitasi.** Verifikasi ulang 39 sitasi ke CrossRef/DataCite/penerbit:
- `YangJoules2025`: bib mencantumkan **3 penulis, rekaman PMLR hanya 2** (Adámek bukan penulis paper
  ini). Diperbaiki + volume 267, pages 70498--70514, URL PMLR terverifikasi.
- `SharmaKaplan2022`: field `doi` menunjuk preprint arXiv 2020 **berjudul lain** ("A Neural Scaling Law
  from the Dimension of the Data Manifold"). Diganti rekaman JMLR terverifikasi (23(9):1--34, 2022,
  jmlr.org/papers/v23/20-1111.html); JMLR tidak menerbitkan DOI.
- `GemaInverse2025` & `ChenOverthink2024`: `and others` diganti **14 penulis penuh** dari DataCite.
- `YangLooped2024`: dijadikan `@inproceedings` ICLR 2024; DOI arXiv 2023 ditandai sebagai preprint.
Hasil akhir: 36 cocok API, 2 tanpa DOI (JMLR & PMLR, keduanya berURL penerbit terverifikasi), 0 sitasi
menggantung.

**Kelengkapan Elsevier (sebelumnya 0 dari 7).** Ditambahkan ke `main.tex`: CRediT (disusun dari
`proposal/pembagian_tugas_tim.md`, ditandai perlu konfirmasi), Declaration of competing interest,
Funding, Data availability, Declaration of generative AI, Acknowledgements yang diperluas. Berkas baru:
`manuscript/highlights.txt` (5 bullet, semua <=85 karakter) dan `manuscript/cover_letter.md` (termasuk
5 usulan reviewer dari daftar pustaka).

**Perbaikan format.** Tabel `tab:crosstask` overfull **100.8pt** (~3,5 cm keluar margin) -> statistik
uji dipindah ke caption, kolom diringkas, `\small`: kini 0 overfull. `tab:baseline` (9.6pt) dan satu
paragraf (16.9pt) juga bersih. Nomor seksi hardcode `Section~6` (2x) -> `\ref{sec:costmodel}`. Keyword
7 -> 6. Ditambah `lineno` untuk peer review. Compile: 22 halaman, 0 error, 0 undefined, 0 overfull.

**Zenodo (tetap DRAFT sampai accepted, sesuai instruksi).** DOI 10.5281/zenodo.21181343 terverifikasi
**404** (reserved, belum published) dan repo GitHub `bukped/trm` terverifikasi **private**, jadi
manuscript tidak boleh mengklaim keduanya sebagai artefak hidup. §Reproducibility & Data availability
ditulis ulang: DOI reserved, dirilis saat accepted, paket menyertai submisi sebagai supplementary
material untuk audit reviewer. Paket disinkronkan penuh: dari 8 -> 13 folder data (menambah `scale_out`,
`converge_out`, `maze_out`, `arc_smoke_out`, melengkapi sisanya), `ablation_reports` dari 7 -> 25 berkas.
README/PROTOCOL/.zenodo.json/DEPOSITION.md diselaraskan; std populasi (0.24/1.37/0.64) di README
dikoreksi ke sample std kanonik (0.3/1.7/0.8); PROTOCOL §5 "subset 1000–2000" dikoreksi ke **512**
(sesuai `rebuild_env.sh`/`make_small_eval.py` yang benar-benar dipakai). Tarball 61.9 -> 100 MB.

**Kebersihan repo.** 9 figur pilot PNG dipindah dari `manuscript/figures/` ke
`eksperimen/frontier/pilot_figures/` (tidak pernah dirujuk `main.tex`; 6 skrip plot pilot diarahkan
ulang). 12 skrip `scripts/watch_*.sh` untuk run yang sudah rampung dihapus (`auto-commit-push-notify.sh`
dipertahankan: dipakai hook Stop). `main.tex.bak` + intermediate LaTeX dihapus. `manuscript/` kini berisi
tepat berkas submisi.

**Sisa yang butuh keputusan penulis (tidak bisa diselesaikan sendiri):** email koresponden
(`\ead` masih placeholder), konfirmasi daftar penulis & peran CRediT, nomor kontrak hibah Ditjen Dikti,
dan publish Zenodo saat accepted.

### Fase V-bis — Kanal artefak: Zenodo saja, tanpa repo GitHub (2026-08-24)

Keputusan: **satu kanal distribusi, yaitu Zenodo.** Repo GitHub `bukped/trm` tidak dirujuk sama sekali
sebagai artefak paper. Ini juga menutup masalah fase V bahwa repo itu terverifikasi *private* sehingga
tak layak disitasi.

Yang dikerjakan:
- Manuscript diperiksa: **nol rujukan GitHub** (Reproducibility & Data availability memang sudah
  Zenodo-only). Path artefak diselaraskan ke layout paket rilis: `ablation/...` ->
  `data/ablation_reports/run_energy_reconciliation.md`, `reconcile_totals.py` ->
  `code/reconcile_totals.py`, supaya pembaca menemukannya persis di tempat yang disebut.
- Metadata deposisi diverifikasi: `related_identifiers` **kosong** (tautan ke repo GitHub sudah tidak
  ada sejak metadata di-PUT ulang di fase V); notes & description bersih dari kata "github".
- `zenodo/README.md` §"How this package is built" diganti §**"Self-contained"**: arsip ini satu-satunya
  distribusi, tak ada repo pendamping yang perlu diambil, tak ada tautan luar yang harus resolve.
  Satu-satunya dependensi eksternal yang tersisa (dan wajib) = clone upstream
  SamsungSAILMontreal/TinyRecursiveModels oleh `rebuild_env.sh`; itu pihak ketiga, bukan repo kami.
- **`DEPOSITION.md` dikeluarkan dari rekaman publik** (HTTP 204 DELETE). Isinya catatan kerja internal
  dan menyebut lokasi file kredensial (`.env`); tidak layak ada di arsip publik. Bersama
  `sync_package.sh` kini dikecualikan dari tarball oleh `sync_package.sh` sendiri, dan file-nya diberi
  penanda "INTERNAL" di baris pertama.
- Unggahan ulang yang sedang berjalan dihentikan di tengah jalan sebelum menulis versi yang keburu
  basi; PUT yang dibatalkan tidak merusak apa pun (tarball lengkap 99.98 MB dari unggahan sebelumnya
  tetap utuh di bucket), lalu diganti unggahan final 366 berkas.

Deposisi **tetap draft/unsubmitted**; publish tetap manual saat paper accepted.

### Fase V-ter — Vendoring kode TRM ke paket (cek lisensi dulu) (2026-08-24)

Menutup satu-satunya dependensi repo eksternal yang tersisa dari fase V-bis. **Lisensi diperiksa lebih
dulu, dan hasilnya menentukan batas vendoring.**

**Hasil cek lisensi:**
| Komponen | Lisensi | Keputusan |
|---|---|---|
| `SamsungSAILMontreal/TinyRecursiveModels` | **MIT**, (c) 2025 Samsung Electronics Co., Ltd. | VENDOR |
| `fchollet/ARC-AGI` (data mentah ARC-AGI-1) | **Apache-2.0** (tanpa berkas NOTICE) | VENDOR |
| `sapientinc/sudoku-extreme` (HF) | **tidak mendeklarasikan lisensi** | JANGAN redistribusi |
| `sapientinc/maze-30x30-hard-1k` (HF) | **tidak mendeklarasikan lisensi** | JANGAN redistribusi |

Sudoku-Extreme juga 762 MB, tetapi alasan utamanya lisensi: tanpa deklarasi, hak redistribusi tidak
jelas. Keduanya tetap diambil saat build oleh dataset builder TRM sendiri; ini didokumentasikan terbuka
sebagai satu-satunya dependensi jaringan yang tersisa.

**Yang di-vendor** (`zenodo/vendor/`, ~16 MB):
- `TinyRecursiveModels/` — 40 berkas tracked, di-export `git archive` pada commit terpaku
  `c01103738605ba39d1430519b1ee0c62f4c707f8` (2026-03-31). **Pristine**, LICENSE upstream utuh.
- `patches/` — dua modifikasi kami dipisah dari pohon upstream supaya jelas mana milik siapa:
  `0001-emit-per-step-progress-and-eval-metrics.patch` (+23 baris di `pretrain.py`; hanya logging,
  tidak menyentuh model/optimizer/data/loss) dan `adam_atan2.py` (shim AdamW untuk sm_120).
- `arc-agi-1-raw/` — 4 JSON (400 training + 400 evaluation) dari `fchollet/ARC-AGI` commit
  `399030444e0ab0cc8b4e199870fb20b863846f34`, plus LICENSE Apache-2.0.

**Verifikasi kuat (bukan sekadar menyalin):** salinan bersih vendor + kedua patch menghasilkan
`pretrain.py` yang **byte-identik** (`diff -q` bersih) dengan `pretrain.py` di `trm-env/TRM` yang
benar-benar memproduksi seluruh 37 run faithful-recipe. Jadi kode di arsip = kode yang menghasilkan
angka, bukan pendekatan.

**Perubahan lain:** `rebuild_env.sh` tidak lagi meng-clone jaringan (salin dari `$VENDOR`, terapkan
patch vendor; clone upstream tinggal fallback bila vendor hilang). `THIRD_PARTY_LICENSES.md` baru di
akar paket. `vendor/README.md` menjelaskan provenance, isi tiap patch dan alasannya, serta apa yang
sengaja TIDAK di-vendor. README/PROTOCOL/.zenodo.json diselaraskan. Manuscript §Reproducibility menyebut
paket self-contained + patch terpisah. Compile 23 halaman, 0 error/undefined/overfull.

Tarball 100 -> 102 MB, 366 -> 415 berkas. Deposisi tetap draft/unsubmitted.

### Fase V-quater — Email koresponden diisi (2026-08-25)

Keputusan penulis: email koresponden = `awangga@ulbi.ac.id` (konsisten dengan afiliasi ULBI di blok
penulis). `\ead{}` diisi, komentar `%% TODO: lengkapi email koresponden sebelum submit` dihapus, dan
placeholder di `cover_letter.md` diganti. **Blocker keras terakhir untuk Editorial Manager tertutup.**
Compile 23 halaman, 0 error/undefined/overfull; `main.tex` kini nol placeholder.

Dua penanda `%% KONFIRMASI PENULIS` sengaja dipertahankan (peran CRediT dan nomor kontrak hibah): itu
komentar LaTeX, tidak muncul di PDF, dan berfungsi sebagai pengingat bila penulis ingin menyesuaikan.

Sisa keputusan penulis tinggal 3: konfirmasi peran CRediT, nomor kontrak hibah Ditjen Dikti, dan publish
deposisi Zenodo saat paper accepted.

### Fase V-quinquies — Nomor kontrak hibah ditemukan di dokumen PDF/DOCX proposal (2026-08-25)

Pencarian nomor hibah di fase V hanya menyisir berkas `.md` sehingga menyimpulkan "tidak ketemu di
`proposal/`". **Itu kurang teliti**: nomornya ada, di dokumen biner. Ditemukan setelah menyisir
`proposal/dokumen/*.pdf` (pdftotext) dan `*.docx` (unzip `word/document.xml`).

**Sumber `Template Surat Pernyataan Kesanggupan dan Pakta Integritas Penelitian 2026__78636547.docx`:**
- Nomor Kontrak Induk: **283/C3/DT.05.00/PL-BARU/2026**, tanggal 30 Januari 2026
- Nomor Kontrak Turunan: **1650/LL4/PG/2026** (21 April 2026, LLDIKTI Wilayah IV <-> ULBI) dan
  **PKS.003/WAREKIII-ULBI/V/2026** (11 Mei 2026, ULBI <-> peneliti)
- Skema: Penelitian Fundamental - Reguler; tahun pelaksanaan 2026, tahun ke-1 dari 2
- Dana tahun ke-1: Rp 125.130.000 (nilai SBK di proposal: Rp 150.000.000)

**Dari `proposal/dokumen/Hukum Penskalaan ....pdf`:** NO. DOKUMEN proposal BIMA **PP-251000228286**,
kementerian penaung = **Kementerian Pendidikan Tinggi, Sains, dan Teknologi** (bukan lagi "Ditjen Dikti"
seperti yang tertulis sebelumnya di manuscript). `suratpernyataan.pdf` adalah hasil pindai (0 karakter
terekstrak), jadi tidak dipakai sebagai sumber.

**Diterapkan:** §Funding manuscript kini memuat rantai kontrak lengkap dan nama kementerian yang benar;
Acknowledgements diselaraskan; `zenodo/README.md`, `.zenodo.json`, dan `cover_letter.md` ikut diperbarui.
Nominal dana sengaja TIDAK dicantumkan (bukan konvensi Elsevier, tidak menambah nilai).

Catatan tipografi: nomor kontrak adalah string panjang tanpa titik-patah alami dan sempat menyebabkan
2 overfull hbox. Solusi: `\slash` sebagai titik-patah, `\nobreakdash-` melindungi tanda hubung
`WAREKIII-ULBI`, dan paragraf dibungkus `sloppypar`. Compile kembali 0 overfull, 23 halaman; ketiga nomor
terbaca utuh di PDF.

Sisa keputusan penulis tinggal 2: konfirmasi peran CRediT, dan publish deposisi Zenodo saat accepted.

### Fase V-sexies — Konfirmasi penulis atas ketiga nomor kontrak (2026-08-25)

Penulis mengonfirmasi ketiga nomor yang diekstrak di fase V-quinquies, dengan penamaan peran eksplisit:
kontrak induk `283/C3/DT.05.00/PL-BARU/2026`, turunan LLDIKTI4 `1650/LL4/PG/2026`, turunan ULBI
`PKS.003/WAREKIII-ULBI/V/2026`. Diverifikasi karakter-per-karakter (setelah normalisasi `\slash` dan
`\nobreakdash-`): ketiganya sudah cocok persis di `main.tex`, `main.pdf`, `zenodo/README.md`, dan
`.zenodo.json`.

Dua penyelarasan kecil mengikuti penamaan penulis:
- `cover_letter.md` sebelumnya hanya memuat kontrak induk (diringkas); kini memuat ketiganya.
- Label di §Funding: "institutional contract" -> **"ULBI institutional contract"**, supaya pembedaan
  turunan LLDIKTI4 vs turunan ULBI terbaca eksplisit seperti yang penulis maksud.

Compile 23 halaman, 0 error/undefined/overfull. Paket Zenodo tidak perlu diunggah ulang: manuscript
tidak termasuk isi paket, dan `README.md`/`.zenodo.json` sudah memuat ketiga nomor sejak fase
V-quinquies (checksum tarball di Zenodo terverifikasi `febe48ea...`, cocok byte per byte).

### Fase V-septies — CRediT dikonfirmasi penulis; manuscript final (2026-08-25)

Penulis mengonfirmasi peran CRediT sudah benar apa adanya. Blok komentar
`%% KONFIRMASI PENULIS` di atas §CRediT dihapus; `main.tex` kini nol penanda konfirmasi dan nol
placeholder. Compile 23 halaman, 0 error/undefined/overfull.

Dengan ini seluruh butir yang bisa diselesaikan dari sisi repo TUNTAS. Yang tersisa hanya satu tindakan
di luar repo: **publish deposisi Zenodo saat paper accepted** (manual di UI; publish memintakan DOI
secara permanen dan tidak bisa dibatalkan).

Catatan konvensi commit: mulai commit ini, pesan commit tidak lagi memakai trailer `Co-Authored-By`
atas permintaan penulis. Identitas git = `Rolly Maulana Awangga <rolly@awang.ga>` (sudah berlaku sejak
commit f598473).

### Fase W — Audit referensi + naikkan 7 sitasi ke venue terbitnya (2026-08-25)

**Audit profil venue 39 sitasi** (setelah koreksi): ML papan-atas 18 (46%), sistem/HPC/energi 7 (18%),
jurnal keberlanjutan 6 (15%), lain 3, preprint arXiv asli 5 (13%). Artinya **64% basis referensi berada
di irisan ML x sistem**, sementara hanya **satu** sitasi berasal dari jurnal target (OliveiraFilho2025 di
SUSCOM). Klaster keberlanjutan justru bagian paling lemah (Sensors, LNCS book-chapter, MAKE).

**Temuan utama: 7 referensi tampak lebih lemah daripada aslinya** karena disitasi sebagai
"arXiv preprint" padahal sudah terbit. Diverifikasi ke DBLP (dan DataCite untuk daftar penulis):

| Entri | Sebelumnya | Terbit di |
|---|---|---|
| `Schwarzschild2021` | arXiv | NeurIPS 2021, hlm 6695--6706 |
| `You2023` | arXiv (2022) | **NSDI 2023**, hlm 119--139 |
| `Xu2025` | arXiv (2024) | **ICML 2025** |
| `Mirzadeh2025` | arXiv (2024) | ICLR 2025 |
| `Saunshi2025` | arXiv | ICLR 2025 |
| `Brandfonbrener2025` | arXiv (2024) | TMLR 2025, vol 2025 |
| `Geiping2025` | arXiv | NeurIPS 2025 |

Field `doi` tetap menunjuk preprint arXiv (banyak venue ini tak mencetak DOI); tiap entri diberi catatan
eksplisit "DOI is the arXiv preprint; published at X (verified DBLP)" supaya perbedaan tahun bib vs tahun
DOI tidak terbaca sebagai kesalahan.

**Dua koreksi terhadap temuan awal saya sendiri:**
- `Xu2025` sempat saya laporkan ICLR 2025; itu **salah**, akibat query DBLP memakai judul yang keliru.
  Setelah diquery dengan judul persis dari `.bib`: **ICML 2025**.
- `Saunshi2025` sempat disimpulkan masih preprint berdasar OpenAlex; DBLP menunjukkan **ICLR 2025**.
  Untuk venue ilmu komputer, DBLP lebih dapat diandalkan daripada OpenAlex.

Tersisa 5 preprint sejati (terverifikasi hanya ada sebagai CoRR/arXiv): `ChenOverthink2024`,
`Choshen2024`, `Li2025`, `Snell2025`, dan `JolicoeurMartineau2025` (paper TRM itu sendiri).
Verifikasi ulang menyeluruh: 32 entri cocok API, 2 tanpa DOI (JMLR & PMLR, keduanya berURL penerbit),
5 "beda tahun" yang memang disengaja dan terdokumentasi. Compile 23 halaman, 0 error/undefined/overfull.

**Implikasi venue (untuk keputusan submisi):** profil referensi menempatkan paper di irisan ML x sistem
(rumah paling alami: MLSys, TMLR), tetapi keduanya sulit dihitung sebagai jurnal terindeks untuk laporan
Penelitian Fundamental. SUSCOM tetap pilihan utama: preseden venue langsung, Elsevier (terkonfirmasi
CrossRef: penerbit "Elsevier BV", ISSN 2210-5379, prefix 10.1016), dan naskah sudah disesuaikan untuknya.
Cadangan berurutan bila ditolak: Future Generation Computer Systems / Journal of Systems and Software,
IEEE Micro, ACM TOMPECS.

### Fase X — Penyelarasan ke venue SUSCOM (anti desk-reject), butir 2-7 (2026-08-25)

Membandingkan naskah dengan artikel contoh yang diunggah penulis: **Phoeni6** (OliveiraFilho2025,
SUSCOM 47:101172) di `../uploads/1-s2.0-S2210537925000939-main.pdf`. Profil pembanding: 18 halaman,
6 seksi, 52 referensi, 2 case study, Related Work di **§5 (belakang)**, dan *Dataset link* GitLab
**hidup** di header ARTICLE INFO.

**Keputusan pokok: kontribusi kita TIDAK disamakan dengan Phoeni6.** Phoeni6 adalah kontribusi
*artefak/tooling* (containerized tools + DBMS + model data + metodologi, divalidasi 2 case study).
Kita kontribusi *temuan empiris* (peta rezim lintas-task + hasil null + protokol). Membingkai ulang
naskah sebagai "framework" berarti bertanding langsung melawan Phoeni6 di jurnal yang sama dengan
senjata yang lebih lemah, sekaligus melanggar aturan repo sendiri ("kontribusi dibingkai sebagai metode,
bukan framework"). Posisi yang diambil: Phoeni6 menjawab *bagaimana mengukur*, kita menjawab *apa yang
ditemukan ketika diukur*; komplementer, dan dinyatakan eksplisit di naskah.

Butir yang dikerjakan (2-7 dari strategi; butir 1 soal tautan Zenodo mati masih menunggu keputusan):

- **(2) Jembatan eksplisit ke Phoeni6** di akhir Introduction: mengadopsi komitmen fair-comparison dan
  reproducibility, lalu membelanjakannya pada target berbeda (anggaran latih tetap, bukan jaringan
  pra-latih apa adanya).
- **(3) Sitasi SUSCOM 1 -> 6.** Lima entri baru diverifikasi CrossRef dan ditempatkan di tempat yang
  memang relevan, bukan padding: `Desislavov2023` (energi inferensi melampaui hukum performa-vs-parameter,
  masuk ke bagian standar bukti scaling), `He2026` (model prediktif energi latih DNN, sejajar cost-model
  kita), `Yao2025` (trade-off daya vs waktu respons), `Spillo2026` (benchmark karbon vs performa),
  `DiGirolamo2022` (bottleneck energi GPU).
- **(4) Abstrak dipimpin payoff energi.** Dulu membuka dengan arsitektur rekursif; kini kalimat ketiga
  sudah memberi angka keputusan: 259 Wh (175 g CO2e) @D9 vs 317 Wh (214 g CO2e) @D18, D36 tak pernah
  mencapai target. Dirampingkan 260 -> **235 kata** (abstrak Phoeni6 ~173 kata).
- **(5) CO2e dilaporkan, data lama yang belum dipakai.** CodeCarbon sudah merekam emisi di 65 run dengan
  intensitas grid **675,93 g CO2e/kWh (Indonesia), persis konstan**. Diterapkan ke energi net-GPU
  (idle-corrected) agar sebasis dengan seluruh metrik energi: **8,5 kg CO2e** untuk 37 run yang
  dilaporkan, **10,2 kg** seluruh studi. `reconcile_totals.py` diperluas menghitung ini (bagian
  "Jejak karbon") sehingga telusur. Catatan jujur ditulis di naskah: angka gross CodeCarbon
  (CPU+GPU+RAM, tanpa koreksi idle) lebih tinggi. Phoeni6 sendiri menyebut carbon footprint hanya
  sebagai motivasi dan tidak melaporkannya, jadi ini diferensiasi.
- **(6) Subseksi baru "Practical guidance"** di Discussion: tiga aturan operasional (belanjakan anggaran
  ke optimizer step pada rekursi dangkal; width dari optimum interior; rencanakan run dengan cost-model
  terukur, bukan hitungan FLOP karena b=0.84 < 1), plus peringatan biaya evaluasi.
- **(7) Related Work dipindah** dari §2 ke setelah Results (kini §6), mengikuti tata letak Phoeni6.
  Dua kalimat yang jadi salah tense setelah pindah diperbaiki, dan ditambah lead-in.

Hasil: 44 sitasi (0 menggantung, 44 bibitem), 37 cocok API + 2 tanpa DOI (JMLR/PMLR) + 5 beda-tahun yang
disengaja. Compile **25 halaman**, 0 error/undefined/overfull.

### Fase X-bis — Membaca 10 paper SUSCOM: dua koreksi sitasi + tiga sitasi lebih kuat (2026-08-25)

Penulis mengunduh kesepuluh paper dari daftar fase X ke `../uploads`. Dibaca semua. Ini menutup celah
yang saya buat sendiri di fase X: lima sitasi SUSCOM di sana dipilih dari **judul saja** karena CrossRef
tidak menyimpan abstraknya.

**KOREKSI KEJUJURAN atas sitasi sendiri:**
1. `Yao2025` **salah dikutip**. Di Introduction saya memakainya sebagai bukti "energi AI dipetakan di
   sumbu parameter". Isi sebenarnya penjadwalan preemptif + power cap GPU per prioritas job (aproksimasi
   matrix-geometric); sumbu parameter tak disinggung. Kalimat Introduction ditulis ulang menjadi
   pembedaan dua sumbu yang benar: sumbu parameter (`Desislavov2023`) versus sumbu sistem, yaitu knob
   pada model yang sudah dipilih (`He2026`, `Yao2025`, `Castano2026`). Kedalaman rekursi bukan keduanya,
   dan itu justru celah yang kita isi. Koreksi ini malah memperkuat argumen posisi.
2. `DiGirolamo2022` **DICABUT**. Isinya algoritma graf (BFS/SSSP) di GPU dengan Unified Memory
   oversubscription, bukan jaringan saraf. Menahannya di paper energi ML = padding yang kelihatan.

**Tiga sitasi pengganti yang benar-benar on-point (semua diverifikasi CrossRef):**
- `Guo2025` SDEnergy (45:101062) — prediksi energi NN dari encoding struktur (GNN) + perangkat,
  MAPE 5,35% pada NAS-Bench-101. Pembanding langsung cost-model kita.
- `Huynh2026` (50:101329) — enam Transformer, energi & CO2 diukur **CodeCarbon atas tiga run
  independen**. Protokol nyaris identik dengan kita, dan temuannya sebentuk: alokasi intuitif bukan yang
  efisien.
- `Castano2026` (50:101347) — pemilihan hardware inferensi sebagai keputusan Pareto energi-performa-biaya
  atas korpus MLPerf.

**`Spillo2026` dinaikkan dari catatan kaki jadi jangkar.** Ternyata saudara metodologis: pipeline
reproducible, 14 algoritma, 3 dataset, temuan "algoritma sederhana kompetitif dengan emisi jauh lebih
rendah, tuning ekshaustif menambah karbon tanpa akurasi". Bersama `Huynh2026` kini dipakai di Discussion
sebagai korroborasi lintas-domain: temuan kita sebentuk dengan keduanya di sumbu berbeda, sehingga
pelajarannya soal alokasi anggaran, bukan kekhasan satu keluarga arsitektur.

**Dua paper prioritas C tidak disitasi** setelah dibaca: `suscom.2025.101254` sebenarnya perbandingan
akurasi CNN (judulnya menyesatkan), `EXTREM-EDGE` adalah ekstensi ISA RISC-V. Tidak dipaksakan.

**Temuan langsung untuk butir 1 (tautan data mati).** Dari 10 paper, dua memasang *Dataset link* di
header ARTICLE INFO: Phoeni6 memakai GitLab, dan `Castano2026` memakai **DOI Zenodo**
(`10.5281/zenodo.15643705`) yang diverifikasi **resolve HTTP 200**. Artinya kanal Zenodo lazim dan
diterima di jurnal ini, tetapi presedennya rekaman yang sudah hidup. Ini memperkuat rekomendasi bahwa
DOI 404 saat review adalah risiko nyata, bukan kekhawatiran berlebihan.

Hasil: 46 sitasi (0 menggantung, 46 bibitem), **8 sitasi SUSCOM** (dari 1 sebelum fase X). Verifikasi
API: 39 cocok, 2 tanpa DOI (JMLR/PMLR), 5 beda-tahun yang disengaja. Compile 26 halaman, 0 error.

### Fase Y — Zenodo dipublikasikan; manuscript beralih ke concept DOI (2026-08-25)

Keputusan penulis mengubah pendirian fase V-bis ("draft sampai accepted") setelah bukti baru dari
pembacaan fase X-bis: paper `Castano2026` di jurnal target memasang DOI Zenodo sebagai *Dataset link*,
dan rekaman itu punya **dua versi** (v1 2025, v2 2026). Jadi mempublikasikan tidak mengunci apa pun.

**Yang dipahami dan diverifikasi tentang mekanismenya:**
- Deposisi kita punya `conceptrecid` 21181342, sehingga DOI yang selama ini disitasi
  (`...21181343`) ternyata **DOI versi**, bukan DOI konsep. Kalau dibiarkan, DOI di paper akan selamanya
  menunjuk v1 dan revisi apa pun tak terlihat pembaca.
- Yang beku setelah publish hanyalah **berkas pada satu versi**. Versi baru bisa ditambahkan kapan saja,
  dan concept DOI otomatis mengikutinya.

**Dikerjakan:**
1. Sebelum publish, tarball di Zenodo ternyata tertinggal (belum memuat `reconcile_totals.py` versi CO2e).
   Dibangun ulang dan diunggah; checksum diverifikasi cocok byte per byte.
2. **v1 dipublikasikan**: `10.5281/zenodo.21181343`, concept DOI lahir `10.5281/zenodo.21181342`.
   Keduanya diverifikasi resolve HTTP 200 di doi.org dan DataCite.
3. Seluruh sitasi dialihkan ke **concept DOI** di `main.tex` (4x), `cover_letter.md`, `zenodo/README.md`,
   `DEPOSITION.md`. Bahasa "reserved DOI / dirilis saat accepted / ditahan selama review" dihapus dari
   abstrak, §Reproducibility, §Data availability, dan cover letter karena tidak lagi benar.
4. **v2 diterbitkan** (`10.5281/zenodo.22090322`): `README.md` di dalam v1 masih memuat kalimat "reserved
   DOI, resolves once published on acceptance" yang jadi salah begitu rekaman terbit. Sekaligus jadi uji
   nyata mekanisme versi.
5. Verifikasi akhir: concept DOI `21181342` redirect ke `records/22090322` (v2); `21181343` tetap hidup
   sebagai snapshot v1. DataCite mencatat concept DOI `HasVersion` keduanya.

Compile 26 halaman, 0 error/undefined/overfull; nol klaim usang tentang status Zenodo tersisa di PDF.
**Butir 1 strategi anti desk-reject dengan ini tertutup.**

### Fase Z — Audit kesiapan submit vs 11 paper SUSCOM: 2 figur baru + KOREKSI dua angka (2026-08-25)

Membandingkan naskah dengan seluruh 11 paper SUSCOM di `../uploads` (bukan hanya Phoeni6), pada dimensi
yang diminta penulis: gaya bahasa, penyajian, kelengkapan figur/tabel/rumus, dan lampiran.

**Panjang naskah TIDAK bermasalah.** 27 halaman itu artefak satu-kolom 12pt bernomor baris. Diukur dengan
jumlah kata yang bebas format: kita **9.087 kata** versus rentang venue **7.445--15.589** (median 11.398).
Kita justru di sisi pendek.

**Kesenjangan nyata: figur.** Kita hanya 3 environment figur (4 gambar) sementara median venue 9
(Phoeni6 21). Metrik utama paper, Joules-to-target, sama sekali tak punya figur; begitu pula cost-model
yang diklaim R^2=0.98. Ditambah dua figur, dibangun dari artefak yang di-commit lewat skrip baru
`make_manuscript_figures.py` (gaya mengikuti figur lama: vektor PDF, DejaVu Serif, lebar 6.6 in,
Okabe-Ito, tanpa judul-plot):
- `fig_energy_accuracy.pdf` — lintasan akurasi terhadap energi net kumulatif, 3 kedalaman x 3 seed,
  sumbu atas CO2e, bintang pada titik capai-target. Ini objek sentral paper yang selama ini tak terlihat.
  Dibangun dengan men-join `pw_*.csv` (daya 1 Hz) ke checkpoint eval di `progress_*.jsonl` run yang sama.
- `fig_cost_model.pdf` — J/step vs params x D_eff log-log, 14 titik microbench, garis fit + pita CI
  bootstrap.

**KOREKSI KEJUJURAN #1 — CI cost-model.** Naskah dan `energy_costmodel_report.md` menulis CI bootstrap
**[0.75, 0.93]**. Batas atas itu **tidak dapat direproduksi**: tiga konvensi berbeda (RNG `random` seed
20260627 dengan persentil-indeks, RNG sama dengan `np.percentile`, dan `np.random.default_rng(0)`)
semuanya memberi **0.921--0.923**, membulat ke **0.92**. Batas bawah 0.7525, eksponen 0.8423, R^2 0.982
terkonfirmasi. Dikoreksi di `main.tex` dan `zenodo/README.md`; catatan koreksi ditambahkan ke report.

**KOREKSI KEJUJURAN #2 — Joules-to-target ternyata angka SEED-0, bukan rata-rata.** Ketahuan karena figur
baru (yang memakai rata-rata seed) menunjukkan 273 Wh sementara tabel menulis 259 Wh. Dihitung ulang
per seed:
| konfigurasi | seed0 | seed1 | seed2 | capai | rata-rata |
|---|---|---|---|---|---|
| TRM D9  | 259 | 289 | 256 | **3/3** | **268 ± 19 Wh** (181 g CO2e) |
| TRM D18 | 317 | 341 | --  | 2/3 | 329 ± 17 Wh (223 g CO2e) |
| TRM D36 | --  | --  | --  | 0/3 | tak pernah |
| baseline non-rekursif | 384 | 323 | -- | 2/3 | 353 ± 43 Wh (239 g CO2e) |

Dua hal tersembunyi terungkap: D18 hanya mencapai target di **2 dari 3 seed**, dan baseline yang di tabel
lama tertulis "not reached" sebenarnya **mencapainya di 2 seed** (yang tak tercapai adalah rata-rata
akurasinya, 49.7%). Semua angka diperbarui ke statistik tiga-seed yang konsisten, lengkap dengan
pencacah seed di tabel. Perubahan ini **memperkuat** temuan: hanya konfigurasi terdangkal yang mencapai
target di setiap seed, dan dengan energi terendah (19% di bawah D18, 24% di bawah baseline).
`make_manuscript_figures.py` kini mencetak tabel ini agar telusur.

**Gaya bahasa: sudah sesuai.** Nol em-dash; kata promosi bersih (`novel`/`robust`/`crucial`/`powerful`/
`state-of-the-art` = 0; `comprehensive` hanya muncul di judul referensi); 11 kemunculan `significant`
semuanya statistik, bukan promosi; persona `we` konsisten dengan venue. Komentar LaTeX berbahasa
Indonesia di-Inggriskan agar berkas `.tex` rapi bila diminta editor.

**Lampiran tidak diperlukan:** 10 dari 11 paper venue tidak punya appendix.

Posisi akhir vs norma venue: kata 9.087 (rentang 7.445--15.589), figur 5 (3--21), tabel 4 (0--21),
persamaan 5 (0--20), algoritma 1 (0--2), referensi 46 (25--112), appendix 0 (0--1). Semua dalam rentang.
Compile 27 halaman, 0 error/undefined/overfull, 0 placeholder, 6 deklarasi lengkap, 46 sitasi 0 menggantung.

### Fase Z-bis — Verifikasi ulang appendix + tabel per-seed (2026-08-25)

Penulis menahan klaim "10 dari 11 paper venue tidak punya appendix". Benar ditahan: deteksi saya di
fase Z memakai grep awal-baris yang kasar. Diperiksa ulang menyeluruh (cari "appendix" di mana pun,
plus "supplementary material/data/information"):

| | jumlah | isi |
|---|---|---|
| punya appendix | **1**/11 | `Desislavov2023`: Appendix C--F (detail metodologi NLP, deskripsi dataset, speed-up mixed-precision). Paper survei 17 hal, 112 referensi |
| punya supplementary | **1**/11 | `Huynh2026`, tetapi isinya satu baris: *"Supplementary material will be provided upon request."* |
| tidak punya keduanya | **9**/11 | termasuk Phoeni6 |

Kesimpulan fase Z bertahan, dengan angka yang kini tepat. Sekaligus terlihat pendekatan kita **lebih
kuat** dari keduanya: Desislavov menaruh detail di appendix, Huynh menjanjikan "upon request" yang tak
dapat diaudit, sementara kita mengarsipkan seluruhnya di Zenodo yang sudah hidup.

**Celah yang justru terungkap dan ditutup: nilai per-seed tidak pernah ditampilkan.** Naskah melaporkan
mean±sd di mana-mana, tetapi tak pernah angka individualnya. Padahal fase Z baru saja membuktikan fakta
tingkat-seed bisa tersembunyi di balik rata-rata (D18 hanya capai target di 2/3 seed). Ditambahkan
Tabel per-seed (12 konfigurasi x 3 seed) untuk seluruh rezim faithful. Catatan: `perseed_exact.csv` yang
ada di `ablation/` ternyata data **pilot** (h128, exact 3--15%) sehingga tidak dipakai; angka diambil
ulang dari `recipe_summary.csv` masing-masing task. Semua cocok persis dengan mean±sd yang sudah
dilaporkan.

Tabel itu memperkuat dua argumen langsung dari angka mentah: urutan Sudoku D9>D18>D36 berlaku di tiap
seed tanpa tumpang tindih antar kelompok (itulah sebabnya statistik Welch besar), dan pada ARC sebaran
dalam satu konfigurasi (27.3--36.4 di D18) lebih lebar daripada jarak antar konfigurasi (itulah sebabnya
sumbu itu dilaporkan null, bukan tren). `make_manuscript_figures.py` kini juga mencetak tabel per-seed
agar telusur.

Posisi akhir: 9.354 kata, 5 figur, **5 tabel**, 5 persamaan, 1 algoritma, 46 referensi, 0 appendix.
Semua dalam rentang venue. Compile 28 halaman, 0 error/undefined/overfull.

### Fase AA — Menguji usul "pilot sebagai pengukur awal": hasil NEGATIF, jadi kontribusi (2026-08-25)

Penulis mengusulkan menjadikan data pilot sebagai metode pengukuran energi awal yang murah. Diuji ke
artefak, bukan diperdebatkan. **Hasilnya membantah usulan itu, dan justru bernilai sebagai temuan.**

**Sudoku, grid kedalaman yang sama, dua rezim:**
| rezim | D9 | D18 | D36 | peringkat | sebaran | sd |
|---|---|---|---|---|---|---|
| pilot h128, 5 seed | 6.17±2.84 | 9.41±2.64 | 8.83±4.27 | **D18 > D36 > D9** | 3.2 poin | 2.6--4.3 |
| faithful h512, 3 seed | 62.43±0.30 | 50.07±1.68 | 36.26±0.79 | **D9 > D18 > D36** | 26.2 poin | 0.3--1.7 |

Pilot menempatkan pemenang sebenarnya di **peringkat terakhir**, dan sebarannya (3.2 poin) lebih kecil
daripada deraunya sendiri, jadi sebenarnya tak mendukung peringkat apa pun. Sapuan pilot kedua
(`scale_out` h512, 1 seed) mengurutkan dua kedalaman yang dicakupnya ke arah sebaliknya lagi.

**Dua hal yang hampir saya pakai keliru, tertangkap saat verifikasi:**
1. **Maze pilot D36 = run GAGAL, bukan hasil.** token 0.00% dengan wall-time **31 detik** versus 3372 s
   (D9) dan 5816 s (D18). Sudah tercatat jujur di jurnal baris 73 sebagai "d36 OOM 16GB", tetapi baris
   token=0.00 tetap ada di `maze_out/maze_summary.csv` sehingga bisa menyesatkan pembaca CSV yang tak
   membuka jurnal. Maze karena itu TIDAK dipakai sebagai bukti.
2. **KOREKSI KEJUJURAN — sebab lantai cross-val 95.89% salah saya sebut.** Fase X menuliskannya sebagai
   sifat rezim pilot. Sebenarnya angka itu berasal dari salah satu run OOM 31 detik tadi: jendela
   integrasi terlalu pendek untuk kedua sampler menyatu. Dihitung ulang dengan memisahkan run <120 s:
   **lantai pilot yang benar 98.68%** (n=63 run selesai), dan empat run pendek (3x Maze D36 @31 s,
   1x scale @117 s) dikecualikan. Ini justru membuat protokol kita tampak lebih baik, bukan lebih buruk.

**Diterapkan ke naskah:**
- Subseksi baru 5.6 "Cheap runs do not screen for the expensive answer", dengan pernyataan batasan
  eksplisit: pilot dan faithful berbeda di lebar, EMA, ukuran dataset dan anggaran langkah sekaligus,
  jadi ini observasi bahwa penyaringan murah GAGAL di sini, bukan klaim faktor mana penyebabnya.
- Energy measurement protocol: sebab 95.9% dikoreksi, lantai 98.7% dinyatakan.
- Kontribusi (3) diperluas jadi dua peringatan (biaya evaluasi + penyaringan murah).
- Abstrak menyebut peringatan kedua; dirampingkan kembali ke **249 kata**.
- Practical guidance: aturan "jangan menyaring pilihan kedalaman di skala tereduksi", plus catatan bahwa
  instrumen murah tetap berguna untuk merencanakan BIAYA run, bukan memprediksi akurasinya.
- Threats to Validity: batasan observasional + sebab angka 95.9%.

**Kenapa bukan framework.** Usul awal ditolak dengan tiga alasan: data kita sendiri membantahnya
(reviewer bisa membongkarnya memakai paket Zenodo kita), kita tak pernah merancangnya sebagai metode
dan tak pernah memvalidasi "pilot memprediksi faithful", dan kontribusi berbentuk framework akan
bertanding langsung melawan Phoeni6 dengan senjata lebih lemah. Bentuk hasil-negatif tidak punya ketiga
masalah itu dan sebentuk dengan temuan biaya-evaluasi yang sudah ada.

`make_manuscript_figures.py` kini juga mencetak `pilot_vs_faithful()` dan `agreement_floor()` agar kedua
klaim baru telusur. Compile 29 halaman, 9.929 kata, 0 error/undefined/overfull.

### Fase AB — Audit repo↔naskah, reproduksibilitas, scope, dan signifikansi kontribusi (2026-08-25)

**(A) Audit angka repo↔naskah: SEMUA cocok.** Diverifikasi ulang ke artefak, bukan ke ingatan:
- Energi net 402/358/340 Wh (depth) dan 384 Wh (baseline): cocok, per-seed [405,402,400], [360,356,358],
  [341,337,344], [384,384,384].
- Tabel kalibrasi (Tabel 2): **9/9 baris** cocok persis dengan `microbench_results.csv`.
- **Seluruh uji statistik reproduksi**: Welch D9-D18 t=12.53 (klaim 12.5), D18-D36 t=12.85 (12.9),
  width h512-h256 t=10.34 p=0.0010 (10.3/0.001), h512-h768 t=6.31 p=0.0065 (6.3/0.007), D9-baseline
  t=10.22 p=0.0083 (10.2/0.008), D36-baseline t=-10.18 p=0.0040 (10.2/0.004), D18-baseline p=0.8185
  (0.82); ANOVA Maze F(2,6)=6.47 (6.46), ARC F(2,6)=3.48 (3.49); gap Maze 1.56 persis; ARC p=0.657
  (0.66) dan 0.063 (0.063).
- Sebaran energi: depth 18.2% (klaim "~20%"), width 60.7% (klaim "~60%"). Cocok.
- Test set Sudoku-Extreme penuh **422.786** instans = 4,23e5 (klaim ~4,2e5). Cocok.

**(B) Reproduksibilitas: DUA celah ditemukan dan ditutup.**
1. **Hyperparameter tidak lengkap di naskah.** Tidak disebut sama sekali: learning rate, weight decay,
   `L_layers`, ACT halting (`halt_max_steps=16`), `mlp_t`, position encodings, jumlah head, expansion,
   dan anggaran batch/step per konfigurasi Maze/ARC. Pembaca tak mungkin mengulang. Ditambahkan
   **Tabel 1: Complete configuration** yang memuat semuanya, diambil langsung dari `run_recipe.py` dan
   `config/arch/trm.yaml` yang di-vendor. Ini juga menaikkan jumlah tabel 5 -> 6, tepat median venue.
2. **KOREKSI KEJUJURAN: klaim idle "is measured" tidak punya jejak.** 4.7 W ternyata konstanta hardcode
   di semua runner; tak ada prosedur pengukuran terekam. Diperiksa ke log daya: minimum absolut yang
   pernah tercatat **4.11 W**. Kalimat diganti jadi akurat: nilai yang dipakai, konsistensinya dengan
   lantai terukur, dan argumen yang lebih kuat daripada klaim "measured" -- konstanta yang sama
   dikurangkan dari SETIAP run sehingga tak ada urutan yang bergantung pada angka persisnya.

Algorithm 1 diperiksa terhadap `run_recipe.py`: sesuai. Keempat skrip regenerasi (`make_manuscript_
figures.py`, `reconcile_totals.py`, `sig_test.py`, `joules_to_target.py`) ada di repo kerja DAN paket.

**(C) Scope.** Sudah ditangani fase X/X-bis: 8 sitasi SUSCOM, jembatan eksplisit ke Phoeni6, payoff
keberlanjutan di kalimat pembuka abstrak. Pemetaan 95 artikel menunjukkan aliran "energi dari model AI"
minoritas (9/95) tapi aktif sampai 2026.

**(D) Signifikansi kontribusi vs 11 paper venue.** Kontribusi mereka dibaca satu per satu:
| paper | bentuk kontribusi |
|---|---|
| Phoeni6 | artefak/tooling; temuan sendiri kecil (6,25% dan 30%) |
| **Desislavov2023** | **analisis empiris murni, tanpa tool** -- "(1) showcase (2) determine (3) report" + dataset publik |
| Spillo2026 | protokol + kuantifikasi empiris + temuan "yang sederhana menang" |
| Guo2025 | model prediktif (GNN, MAPE SOTA) |
| Yao2025 | modeling framework + algoritma matrix-geometric |
| Huynh2026 | framework terapan (GA + adapter + dynamic batching) |
| Castano2026 | framework data-driven (Pareto atas korpus MLPerf) |

Kesimpulan: **Desislavov2023 adalah preseden struktural terdekat** dan membuktikan paper temuan-murni
diterbitkan di venue ini. Bentuk kontribusi kita sah. Soal besaran efek, kita justru di atas norma:
Phoeni6 melaporkan selisih 6,25% dan 30%, Spillo "sederhana kompetitif"; kita melaporkan **26 poin
akurasi** pada compute tetap, penghematan energi 19--24% ke target, kontrol matched-energy, dan satu
hasil negatif tentang penyaringan murah. Kelemahan relatif kita: satu GPU, model kecil, 3 seed, dan tak
menyediakan tool -- ketiganya sudah dinyatakan terbuka di Threats.

Posisi akhir: 10.259 kata, 5 figur, 6 tabel, 5 persamaan, 1 algoritma, 46 sitasi (8 SUSCOM), 0 appendix,
abstrak 249 kata, 31 halaman, 0 error/undefined/overfull, 0 placeholder.

### Fase AC — Kontribusi & gap dipertegas; ETNAS dibaca penuh; Zenodo v3 sebagai DRAFT (2026-08-25)

**Sapuan SUSCOM diperluas** ke 106 artikel dengan jaring lebih lebar, karena classifier fase X melewatkan
Phoeni6 sendiri. Satu kandidat on-scope baru yang sah: **ETNAS** (`Dong2023`, 40:100926). Sisanya
off-scope (deteksi DDoS, manajemen energi EV, alokasi software, WSN, estimasi energi level-metode).
Dengan ini kita menyitir **9 paper SUSCOM**, mencakup praktis seluruh aliran "energi dari model AI" di
jurnal itu; dua yang tidak disitasi sudah dibaca dan ditolak karena isinya tak sesuai judul.

**ETNAS mula-mula disitasi pada tingkat judul** (klaim dibatasi pada apa yang judul lisensikan, dan
batasan itu ditulis di note bib). Penulis kemudian mengunggah PDF-nya, jadi kini dibaca penuh dan
karakterisasinya dipertajam. Isi sebenarnya: NAS multi-objektif hardware-aware berbasis **DARTS**;
menambahkan soft-loss daya ke `L_train` sehingga pencarian mengembalikan arsitektur berdaya rendah dengan
lantai akurasi; daya diperoleh dari **tabel level-operator** yang diukur di perangkat edge **FPGA/DSA**
(ZCU102, Atlas200DK). Hitungan kata: "inference" 17x, "training energy" **0x**, "GPU" 1x. Klaim
tingkat-judul saya ternyata benar, tetapi terlalu lemah; versi terverifikasi jauh lebih tajam dan
menguntungkan posisi kita.

**Paragraf `The gap` baru** di akhir Introduction, dipisah agar menonjol. Menyatakan tiga sumbu yang
sudah digarap dan menunjukkan kedalaman rekursi tak ada di ketiganya:
- sumbu **parameter** (`Desislavov2023`),
- sumbu **sistem**, knob pada model yang sudah dipilih (`He2026`, `Yao2025`, `Castano2026`),
- sumbu **arsitektur**, energi sebagai kendala atas ruang kandidat lewat pencarian (`Dong2023`) atau
  prediksi (`Guo2025`) -- keduanya menyasar biaya **inferensi** dan mengganti pengukuran sistem jadi
  dengan proxy di dalam loop pencarian.
Ditutup dengan alasan kenapa pengukuran ini layak: dua benchmark di jurnal ini menunjukkan alokasi yang
diduga intuitif ternyata bukan yang efisien (`Spillo2026`, `Huynh2026`).

**Abstrak**: klaim kebaruan yang tadinya tersirat ("a cost not yet measured") dibuat **eksplisit**
("the first measurement of that cost, to our knowledge"), lalu dipangkas kembali ke tepat **250 kata**.

**Zenodo v3 = DRAFT, sesuai permintaan penulis.** Deposisi `22094907`, `state=unsubmitted`, tarball
101,49 MB (md5 9efc087a...) + 3 berkas lepas terunggah. Diverifikasi bahwa draft yang belum di-publish
**tidak** mengubah apa pun yang dilihat pembaca: concept DOI `10.5281/zenodo.21181342` tetap HTTP 200 dan
tetap redirect ke `records/22090322` (v2). Jadi isi draft bisa diperiksa dulu, publish kapan saja siap.

Status: 47 sitasi (9 SUSCOM), 0 menggantung, 31 halaman, 0 error/undefined/overfull, abstrak 250 kata.

### Fase AD — Kelemahan relatif dibingkai ulang (berbasis bukti), dan penolakan usul metode ala ETNAS (2026-08-25)

Penulis bertanya apakah kelemahan relatif kita bisa jadi keunggulan seperti ETNAS, dan apakah sebaiknya
kita membuat metode baru ala ETNAS untuk Transformer/TRM. Premisnya diperiksa ke kesebelas paper dulu.

**Temuan: "kelemahan relatif" di audit fase AB ternyata diukur terhadap norma konferensi ML, bukan
norma venue ini.**
| | paper venue | kita |
|---|---|---|
| hardware | Phoeni6: GTX 1080 Ti (2017), Titan Z, Titan X; Spillo: Titan X; Huynh: Tesla P100 (2016); Guo: Jetson Nano/TX1; ETNAS: Atlas200DK | **RTX 5060 Ti (Blackwell)**, lebih baru dari semuanya |
| pengulangan | hanya 1 dari 5 menyebut (Huynh: 3 run). **Phoeni6 dan Spillo tidak melaporkan seed/std/pengulangan sama sekali** | 3 seed + tabel per-seed |
| statistik | tak ada yang melaporkan uji | sample std, Welch, ANOVA, Bonferroni |

**Diterapkan (Introduction + Discussion), Threats TIDAK DISENTUH:**
- Paragraf baru di Introduction: GPU konsumen tunggal dipilih sengaja, itu mesin yang benar-benar
  dimiliki kelompok yang menggarap model seukuran ini, dan itu setting normal untuk kerja pengukuran
  semacam ini (disitir ke `OliveiraFilho2025`, `Spillo2026`, `Guo2025`, `Dong2023`). Dinyatakan terbuka
  bahwa itu **mengorbankan rezim anggaran besar, yang memang tidak kita klaim**, dengan rujukan silang
  eksplisit ke §Threats (label `sec:threats` ditambahkan). Imbalannya dua: setiap run terinstrumentasi
  ujung-ke-ujung sehingga energi yang dilaporkan adalah energi yang benar-benar ditarik mesin, bukan
  angka yang dirakit dari tabel operator atau deskriptor hardware; dan angkanya langsung dapat dipakai
  oleh orang yang menghadapi pilihan ini.
- Paragraf baru di Discussion: dua fitur desain yang lebih menentukan daripada skala, yaitu tiga seed
  dengan sample std + tabel per-seed + uji terkoreksi (itulah yang memisahkan efek besar Sudoku dari
  hasil null Maze/ARC), dan kontrol non-rekursif matched-energy (tanpa itu "dangkal mengalahkan dalam"
  hanya pernyataan tentang satu keluarga, bukan tentang rekursi).
- **Keputusan gaya:** klaim komparatif tentang ketelitian statistik pihak lain sengaja TIDAK ditulis.
  Praktik kita dinyatakan positif tanpa menyiratkan yang lain kurang; sebagian penulis itu bisa jadi
  reviewer, dan klaim semacam itu tak memberi nilai tambah.

**Usul metode ala ETNAS DITOLAK, alasan terkuat dari data kita sendiri:**
1. §5.6 membantah premisnya. ETNAS menaruh **proxy** (tabel daya level-operator) di dalam loop
   pencarian; kita baru membuktikan proxy murah **membalik peringkat** pada sumbu ini.
2. Hasil kita justru **mengurangi** kebutuhan pencarian: bidang desainnya sederhana (kedalaman monoton,
   lebar ber-optimum tunggal). NAS atas ruang semonoton itu adalah anak tangga kompleksitas yang belum
   *earns its keep*.
3. Itu paper berbeda: berbulan run baru + baseline baru (random search, varian DARTS), dan bertanding di
   kandang ETNAS dengan satu GPU melawan FPGA/DSA.

**Versi yang layak, ditulis sebagai future work di Discussion:** cost model (memprediksi biaya sebelum
run) + regime map (arah mana yang berbayar) = **alokator anggaran** yang menerima plafon energi dan
mengembalikan konfigurasi. Ditulis eksplisit bahwa kita **berhenti sebelum mengusulkannya**, karena bukti
yang akan membenarkannya justru yang belum bisa disediakan paper ini, dan §5.6 menunjukkan betapa
menyesatkannya pengganti murah untuk validasi itu.

Status: 32 halaman, 10.868 kata, 47 sitasi, 0 error/undefined/overfull. Threats to Validity utuh.

### Fase AE — Uji kesetaraan kontribusi vs 8 paper SUSCOM; temuan energi iso-akurasi 53% (2026-08-25)

Penulis khawatir kontribusi kurang signifikan. Dibandingkan pada dimensi terukur, bukan kesan.

**Skala eksperimen pembanding (dari PDF):** Phoeni6 = 2 jaringan (AlexNet, MobileNet) + 2 case study;
Spillo2026 = 14 algoritma x 3 dataset; Huynh2026 = 6 Transformer x 4 strategi, 50.680 artikel, 3 run;
Castano2026 = korpus MLPerf, ~200 sistem; Guo2025 = NAS-Bench-101 + perangkat IoT; ETNAS = DARTS,
CIFAR-10/ImageNet, board edge; Desislavov2023 = review FLOPs banyak DNN + data hardware lintas tahun.
Kita = 3 task, 12 konfigurasi, 37 run faithful, 3 seed, 2 sumbu, 1 kontrol.

**Penilaian jujur.** Setara/lebih kuat: ketelitian statistik (jelas terkuat; Phoeni6 dan Spillo bahkan
tak melaporkan seed/std), telusur-artefak, dan adanya **kontrol** matched-energy yang langka di set ini.
Lebih lemah, dinyatakan apa adanya: **keluasan** (1 keluarga model vs 14 algoritma / 200 sistem), **tak
ada artefak yang bisa dipakai ulang** orang lain, dan **2 dari 3 task null**.

**Temuan yang menaikkan signifikansi tanpa run baru: framing energi iso-akurasi.** Metrik
Joules-to-target selama ini dievaluasi pada satu ambang tetap 50%, yang justru MENGECILKAN kasus kita
karena diam soal konfigurasi yang tak pernah sampai. Dievaluasi ulang pada akurasi akhir yang
benar-benar dicapai tiap pesaing (per seed, lalu dirata-rata):

| menyamai akurasi akhir | target | pesaing habiskan | D9 butuh | hemat |
|---|---|---|---|---|
| D36 (dalam) | 36,26% | 340 Wh (230 g) | **161 ± 1 Wh** (109 g), 3/3 seed | **53%** |
| D18 | 50,07% | 358 Wh (242 g) | 268 ± 19 Wh (181 g), 3/3 | 25% |
| baseline non-rekursif | 49,67% | 384 Wh (260 g) | 268 ± 19 Wh (181 g), 3/3 | 30% |

Per-seed untuk baris pertama: 162/161/160 Wh, sangat rapat. Jadi angka headline energi kita sebenarnya
**53%**, bukan 19--24% seperti yang selama ini ditulis. Itu menempatkan kita di rentang headline venue
(Huynh 57,5%; Castano 75%), dan secara metodologis justru framing yang LEBIH benar untuk paper energi:
"berapa energi untuk mutu yang sama".

Diterapkan: subseksi baru **5.6 "Energy at matched accuracy"** + Tabel 6; abstrak dipimpin angka itu
("whatever the deepest attains with its entire 340 Wh, the shallowest attains on 161+/-1 Wh in every
seed"), lalu dipangkas kembali ke **246 kata**. `make_manuscript_figures.py` mencetak
`iso_accuracy_energy()` agar telusur.

Status: 32 halaman, 7 tabel, 5 figur, 47 sitasi, 0 error/undefined/overfull.

### Fase AF — Maze n=3 -> n=5 disiapkan; TERHALANG driver GPU tak sinkron (2026-08-25)

**Alasan menambah seed (bukan demi keluasan, tapi menutup kerapuhan).** Analisis daya atas Tabel 3 kita
sendiri: Maze D36=84,68±0,65 vs D9=83,12±0,67, **Cohen d=2,37**, dan hanya butuh **n=4** untuk lolos
Bonferroni. Saat ini n=3 dengan p=0,044, tepat di bawah ambang 0,0167. Artinya klaim sentral
"adding recursion depth never yields a statistically significant accuracy gain" **bertahan hanya karena
kita berhenti di tiga seed**, dan siapa pun bisa menghitung d=2,37 dari tabel yang kita terbitkan.
Bandingkan ARC: kontras D9-vs-D18 butuh **n=135** (d=0,40), jadi itu null sejati dan tak perlu run
tambahan. ARC sengaja TIDAK ditambah.

Kedua hasil sama-sama berguna: bila signifikan, kita dapat pengecualian task-dependent yang justru
memperkuat framing "regime map, bukan hukum universal" (Sudoku sangat pro-dangkal, Maze sedikit
pro-dalam, ARC null); bila tidak, null-nya kokoh di n=5. Yang harus diganti bila signifikan hanyalah
kalimat "never yields a significant accuracy gain"; klaim energi tetap utuh karena di Maze D36 membayar
+3,5% energi untuk +1,56 poin, itu pertukaran, bukan kemenangan gratis.

**Perkiraan biaya (dari wall-time & energi run yang sudah ada):** 6 run, **12,1 jam, 1,90 kWh,
1,28 kg CO2e** (+13% dari total studi 15,08 kWh). Task keempat ditolak: TRM hanya punya tiga dataset
builder, jadi itu pekerjaan rekayasa berminggu-minggu, bukan jam GPU.

**BLOKIR: driver GPU tak sinkron.** Modul kernel termuat **595.71.05**, library NVML userspace
**595.84**. Akibatnya `nvidia-smi` gagal total dan CodeCarbon memperingatkan "pynvml initialization
failed" lalu melaporkan emisi tanpa komponen GPU. **Torch CUDA tetap jalan** (`is_available()=True`,
device terdeteksi), jadi training akan berjalan mulus sambil menghasilkan `pw_*.csv` kosong dan
`emissions_*.csv` tanpa GPU. Itu persis definisi run cacat menurut CLAUDE.md. **Run tidak dijalankan.**
`rmmod` tak mungkin karena Xorg/gnome-shell/mutter memegang `/dev/nvidia0`; perbaikan praktisnya reboot.

**Disiapkan supaya sekali perintah setelah driver beres:**
- `trm-env/run_mazedepth_s34.sh` — chain seed 3,4 dengan konfigurasi IDENTIK seed 0-2 (D9 b48 24k,
  D18 b48 12k, D36 b24 12k, HIDDEN=256, EMA, NEVAL=25, GROUPS default). Memuat **PREFLIGHT** yang
  menolak jalan bila `nvidia-smi power.draw` atau `pynvml.nvmlInit()` gagal, sehingga insiden ini tak
  bisa terulang diam-diam. Diuji: preflight memang menolak sekarang.
- `scripts/watch_mazedepth_s34.sh` — watcher Telegram, notif tiap run baru selesai (token, net Wh,
  xval, jam) plus notif saat 6 run rampung, dan notif bila preflight menolak. Idempotent lewat file
  state. Diuji: tidak mengirim apa-apa saat belum ada run.

### Fase AG — Maze n=5 SELESAI: klaim sentral PATAH, dan itu temuan sah (2026-08-26)

Reboot memperbaiki driver (modul kernel kini 595.84, cocok library; idle terbaca **4,66 W**, praktis
persis konstanta 4,7 W yang dipakai naskah). Chain jalan 19:01 -> 07:04 WIB, 6 run, semua artefak wajib
lengkap (progress/pw/emissions per run).

**HASIL: prediksi analisis daya benar. Di n=5, D36 Maze SIGNIFIKAN lebih akurat.**
| D_eff | token (n=5) | net Wh |
|---|---|---|
| 9 | 83,17 ± 0,53 | 314,4 ± 3,8 |
| 18 | 83,58 ± 0,51 | 303,3 ± 4,5 |
| **36** | **84,91 ± 0,58** | 323,7 ± 5,6 |

Welch (ambang Bonferroni 0,0167): **D36 vs D9 t=+4,96 p=0,00113 SIGNIFIKAN**; **D36 vs D18 t=+3,88
p=0,00486 SIGNIFIKAN**; D18 vs D9 p=0,247 tidak. ANOVA F(2,12)=**14,23** (di n=3: F(2,6)=6,47).
Gap D36-D9 naik dari 1,56 jadi 1,74 poin.

**Dua klaim naskah PATAH:**
1. "adding recursion depth never yields a statistically significant accuracy gain at fixed compute"
2. "deeper recursion is never the compute- or energy-optimal allocation on any task tested"

Yang kedua patah lebih dalam: D36 membayar hanya **+3,0% energi** untuk +1,74 poin, dan **D9 tidak bisa
mencapai akurasi D36 sama sekali** (maksimum D9 atas 5 seed = 83,85% < 84,91%). Jadi di Maze, kedalaman
membeli sesuatu yang kedangkalan tak bisa beli pada anggaran berapa pun.

**Penilaian: ini MEMPERKUAT paper, bukan merusaknya.** Framing yang sudah dipilih sejak awal adalah
"cross-task regime map, bukan hukum universal"; hasil ini justru mengisinya dengan variasi nyata:
Sudoku sangat pro-dangkal (26 poin, hemat energi 53%), Maze pro-dalam ringan (1,74 poin, +3% energi),
ARC null. Klaim "never" yang universal memang rapuh terhadap satu contoh-tandingan, dan kita menemukan
contoh-tandingan itu sendiri sebelum reviewer. Temuan Sudoku sebagai headline tidak tersentuh.

**TEMUAN KEDUA: lantai cross-val turun pasca-reboot.**
- seed 0-2 (driver 595.71.05): 99,36--99,95% (n=9)
- seed 3-4 (driver 595.84): **98,82--98,88%** (n=6)
Pergeseran sistematis dan konsisten, jelas efek pembaruan driver terhadap kesepakatan NVML vs nvidia-smi.
Klaim naskah "99,1--99,95% pada setiap run yang dilaporkan" karena itu **tidak lagi benar**; lantai
faithful yang benar sekarang **98,82%**. Harus dikoreksi, dan pergeserannya sendiri layak disebut karena
menunjukkan kesepakatan dua instrumen bergantung pada versi driver, bukan hanya pada instrumennya.

Revisi naskah menunggu keputusan penulis karena menyentuh klaim headline.

### Fase AH — Naskah direvisi ke temuan Maze n=5: dari "tak pernah menang" ke pembalikan tanda (2026-08-26)

Revisi menyeluruh mengikuti hasil fase AG. Yang berubah bukan satu kalimat, melainkan klaim utamanya.

**Agregat diperbarui** (6 run faithful baru): 37 -> **43 run faithful**, 12,5 -> **14,4 kWh**;
total 79 -> **85 run**, 15,1 -> **16,9 kWh**; CO2e 8,5 -> **9,7 kg** (dilaporkan) dan 10,2 -> **11,4 kg**
(seluruh studi). Lantai cross-val 99,1 -> **98,8%**.

**Klaim utama ditulis ulang.** Dari "menambah kedalaman tak pernah memberi kenaikan akurasi yang
signifikan" menjadi **efek kedalaman berubah TANDA menurut task**, dengan kondisi pemisah yang eksplisit:
di mana metrik masih membedakan konfigurasi (Sudoku exact, sebaran 26 poin), kedalaman mahal dan
kedangkalan menang telak; di mana metrik sudah jenuh (Maze token, semua dalam 2 poin dekat 85%),
kedalaman membeli sedikit akurasi dengan murah. ARC tetap null tetapi kini dinyatakan **underpowered**
(butuh ~135 seed), bukan bukti ketiadaan efek.

Bagian yang disentuh: abstrak (dirampingkan kembali ke **245 kata**), kontribusi (2), Introduction,
§Cross-task (paragraf Maze baru dgn statistik n=5 dan fakta bahwa tak ada seed D9 yang menyentuh rerata
D36), Tabel 3 (+kolom n), Tabel per-seed (catatan seed 3-4 Maze), §Synthesis (ditulis ulang total di
sekitar pembalikan tanda dan kondisi pemisahnya), §Discussion, §Practical guidance (aturan jadi
kondisional), §Threats (daya statistik dibahas per task, plus pengakuan bahwa saturasi metrik bukan
kebetulan bagi kesimpulan), dan Conclusion.

**Kejujuran metodologis yang ditonjolkan, bukan disembunyikan:** naskah menyatakan terus terang bahwa
null Maze di n=3 "akan menjadi artefak berhenti terlalu awal", dan bahwa grid diperluas justru karena
ukuran efeknya membuat null itu tak layak dipercaya. Itu mengubah kelemahan potensial jadi demonstrasi
disiplin.

**Figur crosstask diregenerasi** memakai SELURUH seed per task (`fig_crosstask()` baru di
`make_manuscript_figures.py`), menampilkan titik tiap seed, mean±sd, dan n per panel. Pembalikan tanda
kini terlihat sekali pandang; caption menyoroti perbedaan skala vertikal (Sudoku 26 poin, Maze <3).

**Pergeseran cross-val akibat driver dilaporkan** sebagai temuan pengukuran tersendiri di §protokol:
run sebelum pembaruan driver 99,36--99,95%, sesudahnya 98,82--98,88%, sistematis bukan derau, dan itu
alasan melaporkan A per run alih-alih satu angka headline.

Status: 33 halaman, 11.818 kata, 5 figur, **7 tabel**, 47 sitasi, 0 menggantung, 0 placeholder,
0 error/undefined/overfull, abstrak 245 kata.

### Fase AI — Draft Zenodo v3 disegarkan ke temuan Maze n=5 (2026-08-26)

Draft `22094907` diperbarui, **tetap `unsubmitted`** sesuai keputusan penulis.

Yang diselaraskan sebelum unggah:
- `README.md`: skala studi 79/15,1 kWh -> **85 run / 16,9 kWh**, faithful 37/12,5 -> **43/14,4 kWh**;
  cross-val 99,1 -> **98,8--99,95%** dengan dua strukturnya dijelaskan (jendela pengukuran dan versi
  driver); blok Maze ditulis ulang ke **n=5** lengkap dengan alasan perluasan grid ("melaporkan null
  n=3 akan menjadi artefak berhenti terlalu awal"); verdict lintas-task diganti dari "tak pernah menang"
  menjadi **efek berubah tanda** dengan kondisi pemisahnya; ARC dinyatakan **underpowered** (butuh ~135
  seed), bukan bukti ketiadaan efek.
- `.zenodo.json`: deskripsi diperbarui senada (2.558 karakter), memuat statistik Maze n=5 dan pernyataan
  bahwa peta rezim diorganisasi oleh posisi task pada metriknya sendiri.
- Tarball dibangun ulang: 107 MB, md5 `6081dd2f...`, memuat 6 run Maze baru (progress/pw/emissions),
  figur crosstask hasil regenerasi, dan `make_manuscript_figures.py` versi terbaru.

Verifikasi pasca-unggah: checksum tarball di Zenodo cocok byte per byte dengan lokal; metadata memuat
frasa "changes sign"; `state=unsubmitted`. **Concept DOI `10.5281/zenodo.21181342` tetap HTTP 200 dan
tetap redirect ke `records/22090322` (v2)** — draft yang belum dipublikasikan tidak mengubah apa pun
yang dilihat pembaca, persis seperti saat v3 pertama kali dibuat.

### Fase AJ — Pemeriksaan kesesuaian template Elsevier (2026-08-26)

**Guide for Authors jurnal TIDAK dapat diambil**: ScienceDirect dan elsevier.com sama-sama mengembalikan
HTTP 403 dengan halaman blokir bot; Google cache 429. Jadi pemeriksaan dilakukan terhadap dua sumber
otoritatif yang tersedia: **template resmi `elsarticle-template-num.tex`** dari distribusi TeX Live
(Copyright Elsevier Ltd) dan **11 PDF terbitan** jurnal ini. Keterbatasan ini dinyatakan apa adanya;
butir yang hanya ada di prosa Guide (mis. batas kata abstrak resmi) tidak dapat diverifikasi langsung.

**Sudah sesuai:** `\documentclass[preprint,12pt]{elsarticle}` persis sama dengan template;
`\bibliographystyle{elsarticle-num}` sama; `\affiliation` memakai bentuk key-value (`organization=`,
`addressline=`, `city=`, `country=`) sesuai contoh; `amssymb` dimuat; tak ada appendix (dan 10 dari 11
paper venue memang tak punya).

**Dua penyimpangan ditemukan dan diperbaiki:**
1. **Posisi `\linenumbers`.** Kita menaruhnya di preamble sehingga judul dan abstrak ikut bernomor.
   Template resmi menempatkannya **setelah `\end{frontmatter}`**. Dipindah; diverifikasi halaman 2
   (judul+abstrak) kini 0 nomor baris sementara halaman isi tetap bernomor (38 nomor di halaman 4).
2. **Highlights tidak disematkan.** Template menyediakan environment `highlights` di dalam frontmatter,
   dan kelas memang mendukungnya (`\elsarticlehighlightsbox`). Sebelumnya highlights hanya ada sebagai
   berkas terpisah `highlights.txt`. Kini disematkan juga, sehingga PDF menghasilkan **halaman
   Highlights tersendiri di depan** persis seperti keluaran template resmi, dan editor menerimanya dalam
   satu berkas. Berkas terpisah tetap dipertahankan karena Editorial Manager memintanya sebagai unggahan
   sendiri.

`graphicalabstract` tersedia di template tetapi opsional; tidak dipakai, sejalan dengan paper venue yang
diperiksa.

Hasil: 36 halaman (bertambah satu karena halaman Highlights), 0 error/undefined/overfull.

### Fase AK — Guide for Authors resmi dibaca; 5 ketidaksesuaian mengikat diperbaiki (2026-08-26/27)

Penulis mengunggah PDF Guide for Authors (22 halaman) setelah fase AJ gagal mengambilnya (HTTP 403).
Ini menutup satu-satunya celah verifikasi yang tersisa, dan menemukan hal-hal yang **tidak mungkin
tertangkap dari template saja**.

**Lima ketidaksesuaian mengikat, semuanya diperbaiki:**
1. **Judul seksi GenAI salah.** Guide menuliskannya harfiah ("Title of new section: ..."): harus
   *"Declaration of generative AI and AI-assisted technologies in the **manuscript preparation**
   process"*. Kita menulis *"in the writing process"*.
2. **Penutup pernyataan GenAI salah.** Guide: *"...take full responsibility for the content of the
   **published article**"*. Kita: *"...of the publication"*.
3. **CRediT.** Daftar peran resmi menulis *"Writing -- review **and** editing"*; kita memakai `&`
   (2 tempat).
4. **Funding.** Guide memberi bentuk standar: `Funding: This work was supported by ... [grant numbers
   xxxx, yyyy]`. Ditulis ulang ke bentuk itu dengan ketiga nomor kontrak dalam kurung siku.
5. **Alamat afiliasi tidak lengkap.** Guide meminta *"full name and postal address of each affiliation"*.
   Kode pos tidak saya tebak; penulis memberikan **40151**. Ditambahkan bersama `state={West Java}`.

**Dua hal yang mengubah pemahaman, bukan sekadar format:**
- *"A PDF is not an acceptable source file."* Submisi wajib berkas sumber `.tex` (atau `.docx`), jadi
  yang diunggah adalah `main.tex` + `references.bib` + `main.bbl` + `figures/`, bukan hanya PDF.
- **Research data Option C berlaku untuk jurnal ini:** *"you are required to deposit your research data
  in a relevant data repository; cite and link to this dataset in your article."* Artinya
  mempublikasikan Zenodo adalah **syarat**, bukan pilihan. Keputusan fase Y (publish v1+v2, sitasi
  concept DOI) ternyata memenuhi persyaratan yang saat itu belum kita ketahui. Kalau deposisi tetap
  draft, ini akan jadi alasan desk-reject yang sah.

**Highlights.** Guide meminta berkas **terpisah dan editable** dengan kata "highlights" di nama berkas,
3--5 bullet, maksimum 85 karakter termasuk spasi. Ditambahkan `highlights.docx` (python-docx, 5 bullet,
semua terverifikasi <=85); `highlights.txt` dipertahankan. Halaman Highlights di dalam PDF tetap ada
karena environment-nya disediakan template resmi Elsevier.

**Cover letter** diselaraskan: bagian data merujuk Option C secara eksplisit, dan Declarations
menyebutkan seksi wajib yang dibawa naskah.

**Pemeriksaan akhir: 18/18 butir Guide terpenuhi** (title page lengkap dengan email koresponden dan
alamat pos, abstrak 245/250 kata, keyword 6/7, seksi bernomor, Acknowledgements terpisah sebelum
referensi, CRediT, competing interests, funding bentuk standar, data availability dengan DOI, GenAI
judul dan penutup persis serta ditempatkan sebelum daftar pustaka, referensi bernomor kurung siku).
36 halaman, 0 error/undefined/overfull.

### Fase AL — Uji risiko desk-reject "out of scope" terhadap Aims & Scope RESMI (2026-08-27)

Penilaian scope sebelumnya (fase X/X-bis) disimpulkan dari 95--106 artikel terbitan. Kini Aims & Scope
resmi terbaca di Guide yang diunggah penulis, jadi bisa dipetakan langsung.

**Pemetaan ke butir scope resmi.** Naskah cocok secara substansi pada **5 dari 6** butir yang relevan:
- "Resource management to optimize performance and power" -- inti paper: alokasi anggaran tetap antara
  dua sumbu
- "Models for collective optimization of power and performance" -- cost-model + regime map
- "Monitoring tools for power and performance of parallel and distributed systems" -- protokol
  pengukuran dua-instrumen
- "Theoretical aspect of energy, power, and temperature" -- hukum pangkat J/step
- "Algorithms for reduced power, energy and heat" -- temuan alokasi efisien
Tidak cocok: butir hardware/arsitektur, dan itu memang bukan kita.

**Temuan penting: pernyataan scope resmi TIDAK PERNAH menyebut machine learning, AI, neural network,
atau deep learning.** Seluruhnya berkerangka sistem/perangkat keras. Itu menjelaskan kenapa aliran
"energi dari model AI" hanya 9 dari 95 artikel, dan mengapa risiko salah-golong itu nyata.

**Celah yang ditemukan: kosakata jurnal nol di prosa kita.** Diperiksa, di badan naskah (bukan daftar
pustaka): "power-aware" 0x, "energy-aware" 0x, "resource management" 0x, dan "Sustainable Computing" 0x
-- satu-satunya kemunculan adalah baris footer "Preprint submitted to...". Editor yang memindai
kesesuaian scope memakai kosakatanya sendiri; kita tidak memberi satu pun kaitnya.

**Ditutup dengan tiga sisipan yang jujur, bukan penjejalan kata kunci:**
1. Introduction: "Stated in the terms of sustainable computing, this is a question of **resource
   management to optimise performance and power**: given a fixed budget of compute, and therefore of
   Joules, how should that budget be allocated between the two axes a recursive model exposes?" --
   memakai frasa butir scope hampir verbatim, dan itu memang deskripsi akurat pertanyaan kita.
2. Kontribusi (1): dibingkai sebagai "joint optimisation of **power and task performance** under a fixed
   budget".
3. Practical guidance: "For **energy-aware** allocation of a training budget...".

Hasil: istilah scope kini muncul di badan naskah (sustainable computing 2x, resource management 1x,
optimise performance and power 1x, energy-aware 1x). Tiga sisipan saja, di tempat yang memang
kalimatnya alami.

Penilaian akhir risiko out-of-scope: **aman**. Cocok pada 5 butir scope eksplisit, 9 sitasi jurnal ini,
jembatan eksplisit ke Phoeni6, payoff keberlanjutan memimpin abstrak, dan kini kosakata scope hadir di
prosa. 36 halaman, 0 error/undefined/overfull.

### Fase AM — Sinyal scope: keyword, bukan highlights (2026-08-27)

Penulis menanyakan apakah kosakata scope perlu masuk ke halaman highlights juga. **Tidak**, dan Guide
sendiri yang menjawabnya: highlights harus "capture the **novel results** of your research as well as any
**new methods** used during your study". Itu tempat untuk hasil, bukan pemosisian. Kelima bullet kita
sudah tepat sasaran (empat hasil baru, satu temuan metodologis) dan semuanya mepet batas 85 karakter,
jadi menyisipkan kosakata scope berarti membuang isi demi kata kunci. Highlights yang berbunyi seperti
materi promosi lebih merugikan daripada netral.

Tempat yang benar adalah **keyword**, yang memang mekanisme discoverability dan masih punya slot: Guide
mengizinkan 1--7, kita memakai 6. Ditambahkan **"sustainable computing"** di posisi pertama, sehingga
menjadi 7. Diperiksa terhadap larangan Guide: tak satu pun keyword memakai "and" atau "of".

Keyword final: sustainable computing; Green AI; recursion depth; tiny recursive models; energy
efficiency; symbolic reasoning; compute-optimal.

Highlights tidak diubah. 36 halaman, 0 error/undefined/overfull.

### Fase AN — Audit kontribusi definitif atas 11 paper SUSCOM (2026-08-27)

Ekstraksi kontribusi kesebelas paper, lalu klasifikasi menurut BENTUKNYA:

**A. Artefak / alat / sistem (5):** Phoeni6 (containerized tools + DBMS + model data + metodologi);
Guo/SDEnergy (prediktor energi GNN, BiGNN); Castano (kerangka rekomendasi Pareto atas korpus MLPerf);
ETNAS (algoritma NAS berbasis DARTS dgn soft-loss daya); EXTREM-EDGE (ekstensi ISA RISC-V + unit AFU).

**B. Kerangka diterapkan ke domain (2):** Huynh (GA + adapter + dynamic batching untuk Transformer);
Yao (modeling framework antrean + algoritma matrix-geometric).

**C. Temuan empiris / benchmark (3):** Desislavov (tiga temuan tentang tren energi + dataset publik);
Spillo (protokol akuntansi karbon + kuantifikasi biaya tuning + evaluasi reduksi data);
DiGirolamo (konsep energy signature + tiga optimasi sadar-input, hemat 11,97%).

**D. Evaluasi komparatif (1):** CNN-eval (membandingkan lima model DL untuk pengenalan citra).

**Kita ada di golongan C.** Penilaian jujur di dalam golongan itu:
| dimensi | kita vs golongan C |
|---|---|
| ketelitian statistik | **terkuat**; hanya Huynh (golongan B) melaporkan pengulangan sama sekali |
| kontrol | **satu-satunya** dgn kontrol matched-energy sejati |
| besaran efek | 53% (iso-akurasi) > Phoeni6 6,25%/30%, > DiGirolamo 11,97% |
| bentuk klaim | hukum **bersyarat**; hanya Desislavov sebanding |
| keluasan | **kalah**: 3 task vs Spillo 14 algoritma x 3 dataset |
| artefak pakai-ulang | **tidak ada**; golongan A punya |

**Apa yang sebenarnya memicu vonis "not significant"?** Empat pemicu lazim, diuji satu per satu:
1. *Temuan sudah terduga* -- TIDAK. Pembalikan tanda kontra-intuitif, dan tesis TRM memprediksi
   sebaliknya.
2. *Sempit/anekdotal* -- sebagian berlaku (3 task); diredam oleh pembingkaian bersyarat + prediksi
   falsifiabel di future work.
3. *Tak mengubah tindakan siapa pun* -- TIDAK. Ada aturan alokasi konkret dan angka 53%.
4. *Bukti lemah* -- TIDAK. Bukti terkuat di seluruh set.

Aman di tiga dari empat pemicu, teredam di satu.

**Satu perbaikan dilakukan.** Pernyataan bahwa temuan melawan premis keluarga arsitektur sebelumnya
hanya ada di §Synthesis, terkubur jauh di dalam Results. Untuk penilaian signifikansi itu harus terlihat
di awal, jadi dinaikkan ke Introduction: "Both halves cut against the premise the architecture family is
built on. Tiny recursive models are motivated by the claim that iteration substitutes for parameters; we
find that at a fixed budget the substitution is uneconomic exactly where accuracy is still being won,
and pays only once it no longer is. The energy question and the architectural question turn out to be
the same question." Itu yang mengubah "sebuah pengukuran energi" menjadi "temuan yang penting".

Verdict: **setara, dan di atas median golongan temuan.** 36 halaman, 0 error/undefined/overfull.

### Fase AO — Kontra-ekspektasi di abstrak & highlights: satu ya, satu tidak (2026-08-27)

**Abstrak: tidak diubah, karena sudah ada.** Kalimat 1 menyatakan premisnya ("substitute iteration depth
for parameter count"), kalimat 3 membalikkannya ("At a fixed training budget the substitution does not
pay for itself"). Kontra-ekspektasi sudah tersampaikan dalam tiga kalimat pertama, dan hanya tersisa
5 kata dari batas 250. Menambah lagi hanya mengulang.

**Highlights: diubah, dan ada alasan kedua.** Dinilai terhadap syarat Guide ("capture the novel results
of your research as well as any new methods"), bullet 1 lama ternyata satu-satunya yang **bukan hasil
maupun metode** -- ia klaim kebaruan ("First measurement of the energy cost of recursion depth..."). Jadi
slot itu sekaligus kurang patuh dan kurang kuat. Diganti menjadi hasil yang membawa kontra-ekspektasi:
**"At a fixed training budget, trading parameters for recursion depth does not pay"** (79 karakter).
Kelima bullet kini semuanya hasil atau temuan metodologis, semua <=85 karakter.

Klaim kebaruan tidak hilang, hanya pindah ke tempat yang tepat: abstrak ("the first measurement of that
cost, to our knowledge") dan Kontribusi (1).

**Catatan proses.** Sempat terjadi ketidaksinkronan: perintah `cd` gagal karena sudah berada di folder
itu, sehingga rantai `cd && cat` tidak menulis `highlights.txt` sementara suntingan `main.tex` berhasil.
Akibatnya berkas terpisah masih versi lama sementara yang tersemat sudah baru. Terdeteksi lewat
pemeriksaan konsistensi eksplisit antara `\begin{highlights}` di `main.tex` dan `highlights.txt`;
keduanya kini diverifikasi identik, dan `highlights.docx` diregenerasi.

36 halaman, 0 error/undefined/overfull.

### Fase AP — Audit konsistensi lintas-berkas: lima angka usang ditemukan di paket rilis (2026-08-27)

Setelah insiden `highlights.txt` tertinggal di fase AO, dibangun pemeriksa konsistensi yang membandingkan
setiap angka bersama di **tujuh berkas** submisi dan rilis: `main.pdf`, `highlights.txt`,
`highlights.docx`, `cover_letter.md`, `zenodo/README.md`, `zenodo/PROTOCOL.md`, `.zenodo.json`.

**Lima angka usang ditemukan, semuanya di paket rilis (naskah sendiri bersih):**
1. `zenodo/README.md` masih memuat **"Joules-to-exact-50%: D9 = 259 Wh, D18 = 317 Wh"** -- angka seed-0
   yang sudah dikoreksi ke statistik tiga-seed di fase Z. Diganti ke 268±19 (3/3) dan 329±17 (2/3),
   sekalian ditambah baris iso-akurasi (161±1 Wh, hemat 53%) yang belum pernah masuk paket.
2. `zenodo/README.md` baris baseline juga memuat "reaching 50 % at 259 vs 384 Wh". Diganti ke
   268±19 vs 353±43 (2/3 seed).
3. `zenodo/README.md` bagian reproduksi masih menyebut agregat "12.5/15.1 kWh". Diganti 14.4/16.9.
4. `zenodo/PROTOCOL.md` dua tempat masih "99.1--99.95% on all 37 faithful-recipe runs". Diganti
   98.8--99.95% dan 43 run.
5. `.zenodo.json` deskripsi masih "79 runs ... 15.1 kWh ... 37 faithful-recipe runs ... 12.5 kWh".
   Semuanya diperbarui.

**Dua bendera yang ternyata BUKAN masalah**, diperiksa dan dinyatakan: "0.842" tak ada di README karena
di sana ditulis 0.84 (pembulatan dua desimal, CI [0.75,0.92] sama); "1.74" tak ada di highlights karena
ditulis 1.7 agar masuk batas 85 karakter. Keduanya pembulatan konsisten, bukan kontradiksi.

**Verifikasi akhir: 13/13 angka konsisten lintas tujuh berkas**, dan highlights identik di ketiga
tempatnya (`.txt` == `.docx` == blok tersemat di `main.tex`), 5 bullet, terpanjang 84 karakter.

Pelajaran yang dicatat: naskah bisa benar sepenuhnya sementara paket rilis tertinggal, karena keduanya
disunting di waktu berbeda. Pemeriksa lintas-berkas ini sebaiknya dijalankan sebelum setiap unggahan
Zenodo, bukan hanya sebelum submisi.

### Fase AQ — Draft v3 disegarkan dgn koreksi lima angka usang (2026-08-27)

Draft `22094907` diperbarui setelah fase AP; **tetap `unsubmitted`**.

Diverifikasi SEBELUM unggah, di dalam tarball: README memuat 7 kecocokan angka baru
(268 +/- 19, 161 +/- 1, 43 faithful, 16.9 kWh) dan **nol** angka usang (259 Wh, 317 Wh, 37 faithful,
15.1 kWh). Praktik baru: periksa isi tarball, bukan hanya berkas kerja, karena keduanya bisa berbeda.

Pasca-unggah: tarball 106,05 MB md5 `81051c96...` cocok byte per byte dengan lokal; metadata memuat
"85 runs", "16.9 kWh", "43 faithful" dan tak lagi memuat satu pun angka usang; `state=unsubmitted`.
Concept DOI `10.5281/zenodo.21181342` tetap HTTP 200 dan tetap ke `records/22090322` (v2), jadi apa yang
dilihat pembaca tidak berubah.

Ini unggahan draft ketiga (v3 dibuat fase AI, disegarkan fase AI-bis untuk Maze n=5, kini untuk koreksi
konsistensi). Publikasi tetap menunggu keputusan penulis; v2 yang sudah terbit sudah memenuhi syarat
Research Data Option C.

### Fase AR — Judul diganti: dari objek yang dipetakan ke temuan (2026-08-27)

**Masalah dua lapis.** (a) Judul lama 137 karakter / 19 kata, sementara rentang venue dari 11 paper
adalah **67--116 karakter / 8--15 kata** -- kita terpanjang, 21 karakter di atas yang terpanjang di
venue, dan Guide meminta judul "concise and informative". (b) Lebih penting: judul lama menyebut **apa
yang dipetakan** ("An Energy--Accuracy Frontier of Recursion Depth versus Parameters"), bukan **apa yang
ditemukan** -- dan temuannya sudah berubah jadi pembalikan tanda sejak fase AG.

Diperhatikan pula bahwa idiom judul venue cenderung menyebut ketegangan atau temuan, bukan objek:
"Trends in AI inference energy consumption: **Beyond the performance-vs-parameter laws**",
"**Balancing** carbon footprint and algorithm performance", "**Balancing** accuracy and sustainability".

**Judul baru (dipilih penulis dari empat kandidat):**
> When recursion depth pays for its energy: a cross-task regime map for tiny recursive models

91 karakter / 15 kata, dalam rentang venue. Menyatakan temuan sebagai **kondisi** (persis bentuk
kontribusi kita), memuat kata "energy" sebagai sinyal scope, dan "regime map" menamai bentuk
kontribusinya. Kandidat B ("Recursion depth is not free") ditolak karena "not free" terbaca universal
padahal temuan kita bersyarat -- di Maze kedalaman justru menang, jadi itu akan jadi overclaim yang
dibantah papernya sendiri di halaman 15.

**Propagasi.** Menerapkan pelajaran fase AP, judul lama langsung disisir ke seluruh berkas dan ditemukan
di tiga tempat selain naskah: `cover_letter.md` (baris Title), `zenodo/README.md` (judul H1 dan rujukan
judul manuscript), dan `.zenodo.json` (field title). Semuanya diperbarui; sapuan ulang mengonfirmasi
string "Energy of Recursion" tak tersisa di berkas mana pun. Verifikasi: 4/4 berkas memuat judul baru.

36 halaman, 0 error/undefined/overfull.

---

## Fase AS: memperkaya bukti visual, dan temuan konvergensi yang nyaris salah baca

**Tujuan.** Naskah punya 5 figur / 8 tabel; survei 11 paper SUSCOM memberi norma **figur 2--21
(median ~9)**, **tabel 0--21 (median ~5)**. Jadi defisitnya ada di figur, bukan tabel: menambah tabel
justru akan jadi bantalan. Target: menambah figur yang membawa bukti baru, bukan mendaur ulang tabel.

**Kandidat yang DITOLAK.** Figur jejak daya (power trace) untuk mendukung klaim "evaluasi mendominasi
biaya". Dicek dulu ke data sebelum digambar: daya praktis rata (median ~157 W; mean di jendela eval
157,6 W vs 157,3 W di luar). Tidak ada yang terlihat, jadi figur itu akan menyesatkan. Tidak dibuat.

**KOREKSI KEJUJURAN (penting).** Saat memeriksa konvergensi, pembacaan awal hanya memakai **seed 0**
dan menghasilkan kesimpulan yang salah arah: D36 seolah satu-satunya yang masih naik (+1,95) sementara
D9 (+1,50) dan D18 (+1,11) sudah plateau. Kesimpulan itu akan masuk naskah sebagai ancaman validitas
"D36 mungkin undertrained". Setelah dihitung ulang atas **ketiga seed** (mean 3 checkpoint terakhir vs
3 sebelumnya):

| config | 3 ckpt terakhir | 3 sebelumnya | delta |
|---|---|---|---|
| D_eff=9  | 61,13 | 58,01 | **+3,12** |
| D_eff=18 | 48,87 | 47,63 | +1,24 |
| D_eff=36 | 35,46 | 34,05 | +1,41 |

Arahnya terbalik: **yang paling curam justru D9, sang pemenang**. Jadi anggaran lebih panjang
diperkirakan **melebarkan** margin dangkal, bukan menutupnya. Pelajarannya sama dengan fase sebelumnya:
angka dari satu seed bukan bukti.

**Yang ditambahkan.**
1. `fig_learning.pdf` (Figur 4) -- kurva belajar Sudoku h512, akurasi vs langkah, 3 seed + mean, tiga
   kedalaman. Memperlihatkan rancangan iso-compute bekerja (D9 50k langkah vs 25k) dan memisahkan dua
   hal yang selama ini tercampur: **per langkah kedalaman menolong** (D18 unggul sampai langkah 32.000;
   di langkah 25k D9 baru 43,82% vs 48,89%), **per satuan compute tidak** (D9 masih punya separuh
   anggaran, dan menghabiskannya membawa run ke 62,43%).
2. `fig_iso_accuracy.pdf` (Figur 7) -- batang berpasangan energi iso-akurasi, visual dari Tabel 6
   (340->161 Wh = 53%; 358->268 = 25%; 384->268 = 30%).

**Kehati-hatian tabel vs figur.** Tabel 3 melaporkan **best checkpoint**, kurva menampilkan **tiap**
checkpoint. Untuk D18 keduanya beda (best 50,07% mendahului terakhir 48,89%). Ini dinyatakan eksplisit
di caption Figur 4 supaya tidak terbaca sebagai inkonsistensi oleh reviewer.

**Ancaman validitas baru (jujur, bukan disembunyikan).** Ditambahkan ke Sec. Threats: tidak satu pun
dari tiga kedalaman benar-benar konvergen pada anggaran bersama; angka +3,1/+1,2/+1,4 dilaporkan apa
adanya, dengan catatan bahwa perbandingan yang diklaim adalah **fixed compute**, bukan peringkat
asimtotik.

**Hasil.** Naskah 7 figur / 8 tabel, **37 halaman**, kompilasi bersih (0 undefined, 0 citation warning,
1 overfull 2,6pt saat `\output` aktif -- tidak terlihat). Semua angka baru telusur ke
`recipe_out/progress_h512_*.jsonl` dan `recipe_out/recipe_summary.csv`.

**Tambahan fase AS: statistik jadi dapat direproduksi, dan tata letak float diperbaiki.**

Saat memeriksa apakah ada tabel yang benar-benar kurang (bukan bantalan), ditemukan pelanggaran halus
aturan integritas #1: angka Welch t / ANOVA F / Cohen d yang dilaporkan naskah **hanya ada di prosa dan
di jurnal ini**, tidak diproduksi skrip yang di-commit. Ditulis `stats_table.py` (stdlib saja; regularized
incomplete beta via continued fraction Lentz untuk p-value t dan F) yang menghitung ulang SEMUA kontras
dari `recipe_out/`, `maze_depth_out/`, `arc_depth_out/recipe_summary.csv`.

Hasilnya **mengonfirmasi seluruh angka prosa** (jadi tidak ada koreksi isi, hanya penguatan telusur):

| kontras | t | p | d | Bonf. |
|---|---|---|---|---|
| Sudoku D9 vs D18 | 12,53 | 0,0050 | 10,23 | lolos |
| Sudoku D18 vs D36 | 12,85 | 0,0013 | 10,50 | lolos |
| Sudoku D9 vs D36 | 53,72 | 5,6e-5 | 43,86 | lolos |
| width h512 vs h256 | 10,34 | 0,0010 | 8,44 | lolos |
| width h512 vs h768 | 6,31 | 0,0065 | 5,16 | lolos |
| D9 vs non-recursive | 10,22 | 0,0083 | 8,34 | lolos |
| D36 vs non-recursive | -10,17 | 0,0040 | -8,31 | lolos |
| Maze D36 vs D9 (n=5) | 4,96 | 0,0011 | 3,14 | lolos |
| Maze D36 vs D18 (n=5) | 3,88 | 0,0049 | 2,45 | lolos |
| ARC D9 vs D36 | 3,59 | 0,0629 | 2,93 | **tidak** |

ANOVA: Sudoku depth F(2,6)=434,9 p=3,2e-7; width F(2,6)=65,7 p=8,3e-5; Maze F(2,12)=14,23 p=0,0007;
ARC F(2,6)=3,49 p=0,099. Bonferroni diterapkan **per sumbu**, bukan lintas seluruh paper; itu bacaan
lebih konservatif untuk sumbu depth (3 kontras) dan bacaan jujur untuk ARC (1 kontras, tetap gagal).
Emit: `ablation/stats_table_{contrasts,anova}.csv` + `stats_table.tex` -> Tabel 8 naskah, tanpa satu pun
angka diketik tangan.

**Tata letak float.** Ditemukan 8 dari 9 tabel terdorong ke halaman 34-38 (jauh dari rujukannya di
halaman 12-18) karena semua float memakai `[t]` dan kalah bersaing dengan tujuh figur. Tabel diubah ke
`[htbp]` plus pelonggaran `topnumber/totalnumber/topfraction`. Hasil: setiap tabel kini bersebelahan
dengan rujukannya, dan dokumen **menyusut 39 -> 37 halaman** karena tidak ada lagi halaman float di akhir.

**Status akhir naskah:** 7 figur, 9 tabel, 37 halaman, 0 undefined, 1 overfull 2,6pt (lama, tak terlihat).

---

## Fase AT: "Evaluation dominates cost" ternyata TIDAK benar untuk run yang dilaporkan

**Pemicu.** Saat menimbang figur tambahan, muncul kandidat "stacked bar train vs eval" untuk menopang
salah satu dari lima highlight: *"Evaluation dominates cost"*. Sebelum menggambar, klaimnya diukur dulu
dari stempel waktu di `progress_*.jsonl` (durasi eval = celah antar langkah train yang memuat rekaman
eval, dikurangi median durasi langkah train).

**Hasil (3 seed per konfigurasi, h512 Sudoku):**

| config | pangsa wall-time untuk eval |
|---|---|
| D_eff=9  | **1,50% ± 0,06** |
| D_eff=18 | **2,99% ± 0,14** |
| D_eff=36 | **5,78% ± 0,21** |

Rasio D36/D9 = **3,84x** untuk rentang kedalaman 4x.

**Artinya klaim itu salah sebagaimana tertulis.** Evaluasi TIDAK mendominasi biaya pada run yang
dilaporkan; pangsanya 1,5-5,8%. Yang benar: evaluasi *naif* memang mustahil (satu pass penuh test set
Sudoku-Extreme ~4,2e5 instance dengan inferensi rekursif 16 langkah tak selesai dalam 900 s, fase
kalibrasi), sehingga protokol mengunci eval ke subset 512 puzzle. Setelah dikunci, biayanya kecil,
**tetapi tumbuh sebanding dengan D_eff**.

Reviewer yang menghitung ini dari log yang kita rilis sendiri akan menemukan highlight dibantah datanya.
Risiko nyata, apalagi highlight hanya lima butir.

**Perbaikan (6 titik di naskah + surat pengantar + 3 berkas highlights):**
- Highlight 5: "Evaluation dominates cost, and a reduced-scale screen ranks the eventual winner last"
  -> **"Evaluation cost scales with depth; a reduced-scale screen ranks the true winner last"** (84 kar).
- Abstrak, kontribusi (3), definisi energi, diskusi, surat pengantar: dibingkai ulang dari "mendominasi
  biaya" menjadi "biayanya tumbuh dengan kedalaman sehingga jadwalnya harus dikunci".
- Seksi kalibrasi: ditambah angka terukur 1,50/2,99/5,78% + argumen metodologisnya.

**Argumen barunya justru lebih kuat.** Alasan jadwal eval wajib dikunci lintas konfigurasi bukan karena
eval mahal secara absolut, melainkan karena mahalnya **bergantung kedalaman**: jadwal yang dibiarkan
bervariasi akan memberi model dangkal keuntungan sistematis. Itu pembelaan yang lebih tajam terhadap
keadilan perbandingan iso-compute daripada klaim lama.

**Telusur.** Fungsi `eval_share_table()` di `make_manuscript_figures.py`; dicetak tiap kali skrip figur
dijalankan. Tidak ada angka yang diketik tangan.

---

## Fase AU: regime map dapat figurnya, dan sumbunya berhenti melingkar

**Pemicu.** Pertanyaan penulis: apakah masih figur-miskin/tabel-berat? Diukur ulang dari 12 paper SUSCOM
di `../uploads` (nomor figur/tabel tertinggi yang dirujuk):

| | min | Q1 | median | Q3 | max | kita (sebelum) |
|---|---|---|---|---|---|---|
| figur | 3 | 6,5 | 9 | 12,5 | 25 | 7 |
| tabel | 0 | 1 | 5,5 | 7,8 | 23 | 9 |

Figur 7 sudah di atas Q1 (jadi tidak lagi figur-miskin), tabel 9 di atas Q3 (memang tabel-berat).

**TEMUAN SERIUS: sumbu pengorganisasi Tabel 7 identik dengan hasilnya.** Kolom *Spread* didefinisikan
sebagai selisih akurasi terbaik-terburuk antar kedalaman. Karena urutan kedalaman **monoton di ketiga
task**, "terbaik" dan "terburuk" selalu D9 dan D36, sehingga:

| task | spread | efek (D36-D9) |
|---|---|---|
| Sudoku | 26,17 | -26,17 |
| ARC | 6,81 | -6,81 |
| Maze | 1,74 | +1,74 |

`spread == |efek|` persis. Tabel itu menyajikan spread sebagai variabel yang *menjelaskan* rezim padahal
ia nilai mutlak dari hal yang dijelaskan. Figur regime map versi naif (efek vs spread) akan jadi tiga
titik di garis y=+-x: tautologi. Reviewer bisa menyerang: "kalian melabeli rezim dari besar efek, lalu
melaporkan efeknya berbeda antar rezim."

**Yang menyelamatkan argumen:** spread tidak menentukan **tanda**. Pembalikan tanda tetap temuan sah;
yang keliru hanya operasionalisasinya.

**Perbaikan.** Sumbu diganti ke **headroom** = jarak akurasi konfigurasi terbaik ke puncak skala, yaitu
properti task+metrik yang bisa dibaca *sebelum* perbandingan dijalankan: Sudoku 38, ARC 67, Maze 15.
ARC tetap diklasifikasi lewat **daya statistik** (butuh ~135 seed), bukan headroom, dan itu dinyatakan
eksplisit di caption. Paragraf sintesis dibingkai ulang; ARC disebut **searah** dengan mekanismenya
(estimasi titiknya negatif) tetapi underpowered, bukan bukti tandingan.

**Figur baru.**
- `fig_regime_map.pdf` (Figur 9), dua panel. Kiri: posisi tiap kedalaman pada skala metriknya + panah
  headroom. Kanan: efek bertanda + CI 95% Welch (t-kuantil via bisection pada `betai`, tanpa SciPy),
  penanda terisi bila signifikan, kosong bila tidak. Panel kiri **bukan** pengulangan panel kanan: ia
  menetapkan di mana pada skala tiap task duduk, yang tak diberikan oleh efek saja.
- `fig_protocol.pdf` (Figur 1), diagram alur protokol. Sengaja memuat **gerbang penolakan** yang nyata
  (kesepakatan dua instrumen >=98,8%; run <120 s dibuang sebagai abort/OOM) supaya membawa keputusan,
  bukan hiasan. 6 dari 12 paper venue punya diagram sejenis.

**Refactor.** `stats_table.py` dibungkus `main()` + guard `__main__` supaya `betai`/`welch` bisa diimpor
figur; satu sumber kebenaran statistik, tidak ada rumus terduplikasi.

**Hasil.** 9 figur / 9 tabel (figur persis di median venue), 39 halaman, 0 undefined, 1 overfull 2,6pt lama.

**Tambahan fase AU: dua cacat di paket rilis.** Saat memeriksa apakah README paket perlu menyebut skrip
baru, ditemukan dua klaim yang salah dan sudah terlanjur ada di Zenodo v1/v2:

1. **Skrip reproduksi statistik salah sebut.** README dan PROTOCOL menunjuk `sig_test.py` sebagai
   pereproduksi angka Welch t / ANOVA / Bonferroni "above". Padahal `sig_test.py` membaca `budget_out/`
   (rezim pilot) dan dengan argumen default langsung berhenti: `NO multi-seed data found`. Reviewer yang
   mengikuti instruksi itu akan gagal mereproduksi. Diperbaiki menunjuk `stats_table.py`, dengan catatan
   eksplisit bahwa `sig_test.py` adalah skrip pilot lama.
2. **Angka run basi:** "regenerates the 37/79-run" -> **43/85**. Lolos dari sapuan sebelumnya karena
   polanya tertulis `37/79` (dengan garis miring), bukan "37 faithful"/"79 runs".

**Diuji, bukan diasumsikan:** `python3 code/stats_table.py data` dijalankan dari akar paket rilis dan
mereproduksi seluruh 10 kontras + 4 ANOVA persis seperti Tabel 8. Pelajaran: klaim reproducibility di
README harus dieksekusi dari layout paket, bukan dari folder kerja.

---

## Fase AV: kontribusi utama dipindah ke posisi kontribusi utama

**Pemicu.** Sebelum menerbitkan Zenodo v3 (permanen), penulis minta cek ulang: masih figur-miskin/
tabel-berat? dan benarkah regime map itu kontribusi utamanya?

**Audit float (metode identik dengan 12 paper venue, dari PDF):**

| | Q1 | median | Q3 | sebelum | sesudah |
|---|---|---|---|---|---|
| figur | 6,5 | 9 | 12,5 | 9 | **9** |
| tabel | 1 | 5,5 | 7,8 | 9 | **7** |
| total | 9,8 | 14 | 16 | 18 | **16** |

Figur sudah di median sejak fase AU. Yang belum: tabel 9 (di atas Q3, lebih banyak dari 9/12 paper) dan
total 18 (di atas Q3, lebih banyak dari 10/12). Jadi menambah figur lagi justru salah arah.

**Audit struktur, temuan pokok.** Judul menamai regime map, abstrak menyajikannya sebagai sintesis, dan
Kesimpulan menutup dengan pesannya. Tetapi naskah tidak memperlakukannya sebagai kontribusi utama:
- di daftar Contributions ia nomor **(2)**, di belakang pengukuran frontier;
- figurnya Figur 9, **figur terakhir**;
- subseksinya **5.6 dari 10**, disusul empat subseksi lagi, tiga di antaranya materi audit.

Akibatnya Results **tidak berakhir pada kontribusi yang dijanjikan judul**, melainkan pada tabel uji
hipotesis dan nilai per-seed.

**Tindakan.** Satu perubahan struktural yang membereskan kedua masalah sekaligus:
- Subseksi 5.8 (uji hipotesis) dan 5.9 (per-seed) dipindah ke **Appendix A dan B**. Tidak ada bukti yang
  dibuang; keduanya tetap dirujuk prosa. Tabel badan naskah 9 -> 7, total float 18 -> 16.
- **Synthesis digeser jadi subseksi terakhir Results.** Urutan baru: 5.1 depth, 5.2 width, 5.3 cross-task,
  5.4 baseline, 5.5 iso-akurasi, 5.6 screening, 5.7 pengukuran energi, **5.8 Synthesis**.

Results kini berakhir pada kalimat "we are reporting the condition under which it costs anything at all",
langsung diikuti Tabel 7 + Figur 9 (regime map), tepat sebelum Related Work.

**Detail teknis.** `\appendix` menaruh tabel lampiran sebagai A.8/B.9 karena counter tabel tidak ter-reset;
ditambahkan `\setcounter{table}{0}` di tiap seksi lampiran sehingga jadi **A.1** dan **B.1**. Kalimat yang
bergantung urutan disisir; satu-satunya ("as the next section shows" di 5.2) tetap menunjuk 5.3, jadi
tidak perlu diubah.

**Hasil.** 39 halaman, 0 undefined, 1 overfull 2,6pt lama. Badan naskah 9 figur / 7 tabel + 2 tabel lampiran.

---

## Fase AW: ARC dinaikkan ke n=5 — PRA-REGISTRASI (ditulis SEBELUM run dijalankan)

**Pemicu.** Pertanyaan penulis: kenapa ARC null, dan bisakah di-run lagi? Diperiksa, ternyata ARC
**tidak null secara seragam**. Ada dua kontras yang nasibnya berbeda jauh:

| kontras | Cohen d | p (n=3) | n/sel utk 80% power |
|---|---|---|---|
| **D9 vs D36** | **2,93** | 0,063 | **3** |
| D9 vs D18 | 0,39 | 0,657 | **102** |
| D18 vs D36 | 1,56 | 0,192 | 8 |

Angka "135 seed" yang tertulis di naskah merujuk kontras **D9-vs-D18**, dan itu memang tak tertolong:
D18 duduk di tengah dengan sd 4,68. Tetapi **D9-vs-D36 lain soal**. Efeknya besar (d=2,93, lebih besar
dari Maze yang 2,37). Ia gagal signifikan bukan karena efek kecil, melainkan karena **varian dua grup
timpang** (3,23 lawan 0,61) sehingga df Welch anjlok ke 2,14. Itu persoalan jumlah seed.

**Proyeksi (ASUMSI, bukan hasil) bila mean/sd bertahan:** n=5 -> t=4,63, p=0,0083; n=7 -> p=0,0012.

**Yang dipertaruhkan.** D9-vs-D36 adalah **tanda efek kedalaman**, yaitu justru kontras yang menyusun
regime map. Bila signifikan, ARC berhenti jadi sel kosong dan peta punya tiga rezim, bukan dua.

**Celah yang juga ditemukan.** Naskah menulis "the decisive contrast would need roughly 135 seeds"
tanpa menyebut kontras mana. Reviewer yang menghitung sendiri D9-vs-D36 akan dapat n=3 dan bisa
menuduh kita memilih kontras yang membenarkan tidak bekerja lebih keras. Kalimat itu harus menyebut
kontrasnya secara eksplisit, **terlepas dari hasil run ini**.

### KOMITMEN PRA-REGISTRASI (mengikat)

1. Jalankan **tepat 6 run**: seed 3 dan 4 x {D9, D18, D36}. Tidak lebih, tidak kurang.
2. Konfigurasi **identik** seed 0-2: D9 b48 24k, D18 b48 12k, D36 b24 12k, h256, GROUPS=3080,
   NEVAL=25, EMA, dataset `arc1-aug1k-e512`. Runner: `trm-env/run_arcdepth_s34.sh`.
3. **Laporkan apa pun hasilnya**, termasuk bila tetap null, bila melemah, atau bila berbalik arah.
4. **DILARANG berhenti begitu p<0,05.** Analisis dilakukan sekali, setelah keenam run selesai.
5. Bila hasilnya tetap null di n=5, itu dilaporkan sebagai null yang kini **berdaya cukup**, bukan
   sebagai kegagalan; dan naskah tetap menyebut ARC sebagai sel yang tak mapan.

Preflight GPU sudah lolos sebelum peluncuran: driver 595.84, nvidia-smi `power.draw` terbaca 4,39 W,
`pynvml.nvmlInit()` sukses, torch 2.11.0+cu128 CUDA aktif, VRAM 16 GB kosong.

Biaya diperkirakan **~11 jam, ~1,7 kWh** (rata-rata run ARC yang ada: D9 1,87 j/292 Wh, D18 1,79 j/284 Wh,
D36 1,76 j/281 Wh).

**Insiden peluncuran (dicatat apa adanya).** Peluncuran pertama memakai `nohup ... &`. Pemeriksaan
`ps -p <pid>` atas PID hasil `pgrep` pertama menunjukkan proses tak ada, sehingga disimpulkan rantai
mati. **Kesimpulan itu salah**: PID yang diperiksa adalah wrapper `bash -c` yang memang keluar,
sedangkan skrip rantainya (PID lain) tetap hidup. Akibatnya:

1. Artefak `*_s3` milik run yang **sedang berjalan** dihapus.
2. Rantai kedua diluncurkan, sehingga **dua training berjalan bersamaan** menulis ke tag yang sama.

Terdeteksi karena preflight rantai kedua membaca **152,79 W** padahal GPU seharusnya idle ~4,4 W, lalu
dikonfirmasi `nvidia-smi --query-compute-apps` menampilkan **dua** proses masing-masing ~5,2 GB.

**Pemulihan.** Kedua rantai dihentikan; GPU diverifikasi kosong (9,47 W, 0 compute app). Dua baris sampah
sempat masuk `arc_depth_out/recipe_summary.csv` untuk tag `h256_d9_recipe_b48_s3` (wall 160,9 s dan 84,5 s,
token 0, cross-val **53,72%** dan 96,66%). Keduanya dihapus manual beserta seluruh artefak `_s3`.
Perlu dicatat: baris 160,9 s **lolos** dari aturan buang-otomatis `<120 s`, jadi aturan itu saja tidak
cukup untuk menangkap kontaminasi jenis ini. Salinan sebelum pembersihan disimpan sementara untuk
pembanding. Verifikasi pasca-pembersihan: summary kembali 9 run, mean per sel **32,78 / 31,19 / 25,98**,
persis angka yang dilaporkan naskah.

**Peluncuran ulang** memakai `setsid nohup ... < /dev/null & disown`, dan verifikasi kini dilakukan dengan
**menghitung** proses, bukan mengecek satu PID: rantai 1, watcher 1, `nvidia-smi` compute app **1**.
Mulai 17:58:52 WIB, perkiraan rampung ~11 jam kemudian.

**Pelajaran.** (a) Jangan simpulkan proses mati dari satu PID hasil `pgrep`; hitung prosesnya.
(b) Preflight daya GPU ternyata berguna untuk hal yang tak dirancangnya: 152 W saat seharusnya idle
adalah sinyal ada run lain. (c) Aturan `<120 s` tidak menangkap run terpotong yang lebih panjang;
kontaminasi tetap perlu pemeriksaan manual.

---

## Fase AW (lanjutan): HASIL run ARC n=5 — lolos 0,05, GAGAL Bonferroni

Rantai rampung 28 Agustus 2026 pukul 04:50 WIB. Enam run bersih: cross-val minimum **98,88%**,
wall minimum **6311 s** (tak ada run terpotong seperti insiden peluncuran).

**Hasil (token accuracy, n=5):**

| config | per seed | mean |
|---|---|---|
| D9 | 36,46 / 30,40 / 31,49 / 30,14 / 27,51 | **31,20 ± 3,28** |
| D18 | 29,83 / 27,34 / 36,41 / 28,15 / 31,65 | 30,68 ± 3,61 |
| D36 | 26,11 / 25,31 / 26,51 / 25,07 / 25,98 | **25,80 ± 0,59** |

**Uji (Bonferroni per sumbu, alpha = 0,05/3 = 0,0167):**

| kontras | t | p | d | putusan |
|---|---|---|---|---|
| D9 vs D36 | 3,62 | **0,0200** | 2,29 | lolos 0,05, **GAGAL** Bonferroni |
| D18 vs D36 | 2,99 | 0,0379 | 1,89 | gagal |
| D9 vs D18 | 0,24 | 0,8168 | 0,15 | gagal, datar sungguhan |

ANOVA: **F(2,12) = 5,52, p = 0,0200** (sebelumnya F(2,6)=3,49, p=0,099).

**Proyeksi meleset, dan sebabnya jelas.** Perkiraan pra-run p≈0,008 mengandaikan mean/sd bertahan.
Dua seed D9 baru justru masuk rendah (30,14 dan 27,51), menarik mean dari 32,78 ke 31,20 sehingga
jarak D9-D36 menyusut dari 6,80 ke 5,40 poin. Proyeksi itu sudah ditandai "asumsi, bukan hasil"
sejak awal, tetapi tetap perlu dicatat bahwa ia terlalu optimistis.

**Putusan: ARC = "unresolved", bukan rezim ketiga.** Melaporkan p=0,0200 sebagai signifikan berarti
membuang koreksi Bonferroni justru pada satu task yang merepotkan, padahal koreksi itu dipakai di
sumbu Sudoku dan Maze. Reviewer yang menghitung sendiri akan menemukan inkonsistensi itu.

**Nilai run ini tetap besar meski null.** Klaim lama "butuh ~135 seed" bisa dibaca sebagai alasan malas,
dan angka itu sebenarnya milik kontras D9-vs-D18 (yang memang datar, d=0,15). Sekarang naskah bisa
menyatakan hal yang lebih kuat dan lebih jujur: pada lima seed efeknya terdeteksi di 0,05 tetapi tidak
bertahan di bawah koreksi, dan sumbu kedalaman ARC bukan sumbu bergradasi pada anggaran ini.

**Propagasi (12 titik naskah + surat pengantar + README/PROTOCOL/.zenodo.json):** prosa hasil ARC,
Tabel 4 (n 3->5, "not significant" -> "unresolved"), caption Tabel 4, caption Figur 7, abstrak,
Contribution, sintesis, caption Tabel 7, baris Tabel 7 (headroom 67 -> **69**, verifikasi: best 31,20),
caption Figur 9, dan Threats (paragraf 135-seed diganti hasil terukur + catatan pra-registrasi).

**Dua bug figur yang tertangkap saat verifikasi visual:** (a) label rezim masih "underpowered";
(b) `fig_regime_map` memakai ambang `p<0,05` sehingga ARC tergambar **terisi** alias mapan, padahal
gagal Bonferroni. Ambang diubah ke `0,05/3` agar konsisten dengan Tabel 8, dan caption menyebut
ambangnya eksplisit.

**Agregat bergeser:** faithful 43 -> **49 run**, 14,4 -> **16,1 kWh**, 9,7 -> **10,9 kg CO2e**;
total 85 -> **91 run**, 16,9 -> **18,6 kWh**, 11,4 -> **12,6 kg**. Disapu ke seluruh berkas.
Catatan: dua angka kg CO2e sempat lolos sapuan pertama karena polanya terpotong baris.

**Dua cacat tertangkap saat verifikasi visual PDF (bukan dari grep).** Setelah propagasi hasil ARC,
render halaman lampiran memperlihatkan:

1. **Tabel A.1 dan B.1 masih memuat ARC n=3.** `stats_table.py` sudah dijalankan ulang dan berkas
   `.tex`-nya benar, tetapi barisnya belum disalin ke naskah. Tabel B.1 juga masih 3 kolom seed dengan
   catatan kaki untuk dua seed Maze tambahan; kini diperluas jadi **5 kolom** sehingga Maze dan ARC
   tampil utuh dan catatan kaki itu tak diperlukan lagi.
2. **Ambang Bonferroni tidak konsisten antar sumbu.** Sudoku depth dapat alpha=0,017 (3 kontras),
   Maze 0,025 (2), ARC 0,050 (1) -- semata karena berapa baris yang kebetulan didaftarkan di
   `stats_table.py`. Itu tak bisa dibela: ambang jadi bergantung pada pilihan penyajian, dan ARC
   justru mendapat ambang paling longgar. Diseragamkan: **ketiga sumbu depth memuat semua 3 kontras
   berpasangan**, jadi alpha=0,0167 di semuanya. Maze tetap lolos (p=0,0011 dan 0,0049), ARC tetap
   gagal (p=0,0200) -- konsisten dengan yang sudah dinyatakan prosa. Kontras baru yang muncul:
   Maze D18-vs-D9 p=0,2474 (datar) dan ARC D18-vs-D36 p=0,0379 (gagal).

Andai hanya mengandalkan grep, cacat (2) tak akan ketemu: angkanya "benar" menurut skrip, yang salah
adalah definisi keluarganya. Pelajaran: setelah propagasi angka, **render dan baca halamannya**.

---

## Fase AX: audit menyeluruh naskah, figur, dan tabel

Diminta penulis sebelum penerbitan Zenodo v3. Setiap klaim diuji ke data, bukan dibaca ulang.

**Tabel: 9/9 lolos.** Skrip audit menghitung ulang seluruh mean/sd dari summary CSV lalu mencocokkan
ke isi tiap tabel. Semua cocok, termasuk yang turunan: headroom (Sudoku 37,6->38; ARC 68,8->69;
Maze 15,1->15), CO2e iso-akurasi (230/242/260/109/181 g pada 0,67593 g/Wh), persentase hemat
(53/25/30%), dan "matches deep on 47% of its energy" (161/340 = 47,2%). Tabel kalibrasi dicocokkan
baris demi baris ke `microbench_results.csv`: 9 baris, semua sama, termasuk baris OOM h768 D36 yang
memang tak punya entri CSV.

**Figur: 10/10 deterministik.** Regenerasi dua kali lalu bandingkan hasil render PNG: identik. Hash
PDF berbeda hanya karena `CreationDate` matplotlib, bukan isi. Tak ada berkas gambar yatim, tak ada
yang dirujuk tapi hilang.

**TEMUAN: Figur 9 (regime map) tidak pernah dirujuk prosa.** Elsevier mewajibkan tiap float dirujuk,
dan ironisnya yang terlewat justru figur kontribusi utama. Diperbaiki: kalimat pembuka Tabel 7 kini
merujuk keduanya. Sapuan ulang: 18 float, semuanya dirujuk.

**TEMUAN: `.zenodo.json` masih memuat klaim ARC lama** ("would need about 135 seeds per cell",
"reported as underpowered"). Lolos dari propagasi fase AW karena JSON tidak ikut disapu bersama
berkas `.tex`/`.md`. Diperbarui ke rumusan n=5.

**TEMUAN: abstrak kembali menyentuh 250 kata**, batas persis Guide for Authors, karena suntingan ARC
menambah kata. Dipangkas ke **247** tanpa mengubah klaim atau angka.

**Sitasi.** 47 sitasi unik, semuanya punya entri `.bib`, tak ada yang menggantung. 27 entri yatim
(tak disitasi) dibiarkan, itu kumpulan sumber yang wajar. Lima peringatan bibtex "empty pages"
diperiksa satu per satu: kelimanya paper NeurIPS/ICLR/ICML yang memang tak bernomor halaman, jadi
benign, bukan metadata cacat.

**Gaya.** Em-dash: 8 kemunculan, semuanya `---` di dalam tabel yang berarti "tidak berlaku", bukan
em-dash prosa. Kata promosi: robust/novel/comprehensive/crucial/powerful semuanya **0**; "first"
13 kali tetapi hanya satu klaim kebaruan ("first measurement"), sisanya "first reaches"/"first
checkpoint". Pembuka formulaik: nol.

**Frontmatter.** Abstrak 247 kata, 7 kata kunci, highlights 5 butir identik di tex/txt/docx (maks 84
karakter), CRediT + Data availability + Funding + seksi GenAI lengkap.

**Posisi terhadap venue (12 paper SUSCOM).** Figur 9 (median 9), tabel badan 7 (Q3 7,8), total 16
(Q3 16). Ketiganya di dalam pita normal. Float tersebar halaman 7-22, tak ada yang terdampar.

**Kompilasi.** 40 halaman, 0 undefined, 0 citation warning, 1 overfull 2,6pt di halaman 1
(frontmatter elsarticle, ~0,9 mm, tak terlihat pada render).

---

## Fase AY: Zenodo v3 DITERBITKAN

Atas persetujuan penulis. Tindakan permanen, jadi didahului verifikasi berlapis.

**Yang tertangkap tepat sebelum terbit (dan akan permanen kalau lolos):** deskripsi metadata di draft
Zenodo **masih memuat klaim ARC lama** ("would need about 135 seeds per cell", "reported as
underpowered", "43 faithful"). Perbaikan fase AX hanya menyentuh berkas `.zenodo.json` **lokal**;
Zenodo tidak membaca berkas itu otomatis (hanya dipakai saat impor dari GitHub), sehingga metadata di
server tak ikut berubah. Diperbaiki lewat `PUT /deposit/depositions/22094907` sebelum publish.
Pelajaran: **`.zenodo.json` lokal bukan sumber kebenaran bagi rekaman yang sudah ada** -- metadata
server harus di-PUT eksplisit dan diverifikasi.

Ditemukan juga satu sisa di `.zenodo.json`: daftar isi masih menulis "ARC-AGI-1 depth grid 3 seeds";
diubah ke "ARC-AGI-1 and Maze-Hard depth grids, 5 seeds each".

**Verifikasi pra-terbit.** Keempat berkas dicocokkan md5 terhadap salinan lokal, semuanya identik:
- `trm-energy-recursion.tar.gz` 449c15a72cd449e9e97c2962249ed839 (115.549.802 byte)
- `README.md` 1360eae1a3dda167d71470e405bf9942
- `PROTOCOL.md` 06827647b1705472663a164406550ca2
- `THIRD_PARTY_LICENSES.md` fdf456e39da47a669f299f6924c97152

**Hasil terbit.** HTTP 202, state `done`.
- DOI versi v3: **10.5281/zenodo.22094907**
- Concept DOI: **10.5281/zenodo.21181342** (dirujuk naskah, tak berubah)
- Diverifikasi: concept DOI kini me-resolve ke `zenodo.org/records/22094907` (HTTP 200), yakni v3.

Dengan ini cacat yang sempat ikut ke v1/v2 tertutup bagi pembaca yang mengikuti concept DOI: README
tak lagi menyuruh menjalankan `sig_test.py` (yang gagal), dan hasil ARC sudah n=5.

---

## Fase AZ: cermin kode di GitHub `awangga/ctrm`

Penulis menyiapkan repo dan meminta paketnya ditaruh di sana.

**Pembagian kanal.** Paket penuh 775 MB, dan 649 MB di antaranya adalah `progress_*.jsonl` (kurva
per-langkah) plus 90 MB log training mentah. Menaruh itu di git berarti tiap clone menarik 775 MB
selamanya, padahal data itu **sudah** terbit permanen di Zenodo ber-DOI. Jadi: GitHub membawa kode,
protokol, dan artefak ringkas (summary CSV, deret daya 1 Hz, emisi CodeCarbon, laporan ablasi, vendor)
= **37 MB**; Zenodo tetap arsip lengkap yang disitasi. README memuat banner eksplisit soal ini.

**Keputusan penulis (via AskUserQuestion):**
1. **Lisensi GPL-3.0 -> MIT.** Repo dibuat dengan GPL-3.0, tetapi Zenodo v1/v2/v3 sudah terbit
   permanen sebagai MIT dan `THIRD_PARTY_LICENSES.md` menyatakan "the code and analysis written for
   this study are MIT-licensed (see LICENSE)". Kode sama di bawah dua lisensi berbeda tak bisa dibela,
   dan rekaman Zenodo tak bisa ditarik. Diganti ke MIT persis salinan paket Zenodo.
2. **Naskah kini menyebut ctrm.** Membalik keputusan lama "cukup ZENODO saja tanpa repo github".
   Ditambahkan di dua tempat: *Data availability* dan *Reproducibility*, keduanya menegaskan Zenodo
   sebagai **version of record untuk sitasi** dan GitHub sebagai cermin untuk menjelajah/isu.

**Cacat README yang tertangkap saat menyusun** (README disalin dari paket Zenodo, jadi mewarisi
klaim yang tak berlaku di sini): baris isi menyebut `.zenodo.json` yang tak ada di repo ini;
"faithful-recipe regime (the manuscript results, **37 runs**)" padahal 49; Maze dan ARC masih ditulis
"3 seeds" padahal 5; dan kalimat "every empirical number traces to a run artifact in `data/`" tak lagi
benar setelah kurva per-langkah ditinggal di Zenodo. Keempatnya diperbaiki.

**Diuji dari layout repo:** `python3 code/stats_table.py data` mereproduksi 10 kontras + 4 ANOVA
persis Tabel A.1, dan `reconcile_totals.py --data-root data` mengembalikan 49 run / 16,1 kWh.

Commit `7318337` di `github.com/awangga/ctrm`. Naskah 40 halaman, 0 undefined.

---

## Fase BA: tanggal surat pengantar diisi; dua sitasi di daftar reviewer dikoreksi

Tanggal submit diisi **28 Agustus 2026**, penanda `_[isi tanggal submit]_` dibuang.

**Sapuan penanda TODO menemukan satu lagi** di blok *Suggested reviewers*, yang menyatakan afiliasi dan
email kandidat belum diverifikasi. Bagian email/afiliasi memang tugas penulis (perlu cek ke halaman
institusi), tetapi **klaim kepengarangannya bisa diverifikasi ke `references.bib`**, dan pemeriksaan itu
menemukan dua kesalahan:

1. **Wesley Armour** dideskripsikan sebagai "senior author of the SC24 work on **bias in GPU power
   sampling**". Judul sebenarnya (`Yang2024`): *"Accurate and Convenient Energy Measurements for GPUs:
   A Detailed Study of NVIDIA GPU's Built-In Power Sensor"*. Bukan soal bias sampling. Dikoreksi dan
   ditambahi rujukan eksplisit ke sitasi naskahnya.
2. **Samuel Xavier-de-Souza** disebut senior author *"Measuring the energy consumption of neural
   networks"*. Judul sebenarnya (`OliveiraFilho2025`): *"Phoeni6: A systematic approach for evaluating
   the energy consumption of neural networks"*. Dikoreksi.

Ketiga kandidat lain terverifikasi tepat: Jay dan Lefèvre memang co-author `Jay2023` (CCGrid 2023,
*An experimental comparison of software-based power meters*), García-Sánchez co-author
`Aquinobritez2025` (Sensors). Catatan: pencarian nama awal sempat melaporkan Lefèvre dan Armour "tidak
ada di .bib" karena pola grep tak menangkap "Lefevre" tanpa diakritik dan Armour sebagai penulis ketiga;
keduanya sebenarnya ada. Itu kesalahan pencarian, bukan kesalahan naskah.

Komentar TODO diperbarui: menyatakan bahwa klaim kepengarangan **sudah** diverifikasi ke `.bib`, dan yang
tersisa hanya email + afiliasi terkini yang diminta formulir Elsevier.

Surat pengantar: 993 kata, tak ada angka basi.

---

## Fase BB: laporan kemajuan DIKTI (bulan ke-7)

Penulis menyediakan folder `lapkemajuan/` berisi `README.md` (kerangka isian BIMA) dan
*Template Laporan Kemajuan 2026.docx* (poin C sampai H).

**Disusun:** `lapkemajuan/README.md` (Ringkasan, Keyword, daftar bukti pendukung) dan
`substansi_laporan_kemajuan.md` -> `.pdf` (6 halaman) mengikuti struktur template C-H.

**Bagian tersulit: poin F, karena tiga janji proposal tidak dipenuhi apa adanya.** Diputuskan
melaporkannya terbuka beserta alasan ilmiahnya, bukan dipoles:

1. **Scaling law A(P,D) dan E(P,D) dengan R2 >= 0,90 tidak diklaim pada sumbu akurasi.** Alasan:
   rentang skala 1,5-2 dex di bawah bar yang lazim, dan lebih menentukan, **arah efeknya berbalik
   antar tugas** sehingga satu eksponen tunggal akan menyembunyikan temuan yang justru paling
   menarik. Dilaporkan bahwa target R2 >= 0,90 justru **terlampaui pada sumbu energi**
   (b = 0,84, CI [0,75, 0,92], R2 = 0,98).
2. **144 sesi -> 91 run (49 utama).** Bukan pengurangan cakupan melainkan perubahan desain: grid
   penuh 8x6 pada anggaran langkah sama akan menghasilkan perbandingan iso-parameter yang tidak
   adil. Anggaran run dialihkan dari memperbanyak titik ke memperkuat daya statistik tiap titik;
   tanpa itu efek Maze akan tetap terbaca nol.
3. **Akurasi >= 90% dan ARC >= 35% tidak tercapai.** Dilaporkan apa adanya (Sudoku exact 62,4%;
   Maze dan ARC exact = 0 sehingga dipakai token), dengan penegasan bahwa yang dibandingkan adalah
   **urutan** pada compute setara, bukan skor mutlak.

Ditemukan pula satu substitusi instrumen yang belum pernah dicatat: proposal menjanjikan validasi
silang terhadap **CarbonTracker**, realisasinya memakai integrasi `nvidia-smi power.draw`. Alasannya
sah (nvidia-smi membaca sensor kartu langsung sehingga lebih independen terhadap CodeCarbon yang
sama-sama lewat NVML), tetapi tetap ditulis eksplisit di poin C.3.

**Verifikasi sebelum jadi PDF.** Seluruh angka laporan dicek ulang ke summary CSV (akurasi tiga tugas
cocok semua), dan kelima sitasi diverifikasi ke API otoritatif: CCGrid 2023 (CrossRef, halaman
106-118 dikonfirmasi), Phoeni6 (CrossRef), SC24 (CrossRef), TRM dan Kaplan (DataCite). Catatan:
Kaplan2020 tidak ada di `references.bib` naskah, jadi DOI-nya diverifikasi langsung ke DataCite.

**Cacat render yang tertangkap saat memeriksa PDF:** rumus `alpha.P^beta.D^gamma` terbaca sebagai
superscript liar di pandoc; ditulis ulang sebagai matematika LaTeX. Juga dibuang rujukan "(Tabel 1)"
yang sebenarnya menunjuk tabel di naskah artikel, bukan di laporan ini.

**Revisi fase BB: laporan difokuskan ke Tahun ke-1 saja** (arahan penulis). Dibuang: subseksi D.3 yang
menguraikan luaran Tahun 2 (diganti catatan satu kalimat bahwa belum jatuh tempo), penjadwalan Tahun 2
di poin E, dan subseksi G.2/G.3 yang menguraikan Tahap 4-6 beserta peta jalan dua tahun. Poin G kini
seluruhnya tentang sisa Tahun 1 (bulan ke-8 sampai ke-12), ditutup satu kalimat tentang tahun
berikutnya karena template memang memintanya. Header memuat baris **Cakupan** eksplisit.
Dua sebutan lintas-tahun sengaja dipertahankan: perbandingan "melampaui janji proposal yang menempatkan
rilis dataset pada bulan ke-23/24" (justru memperkuat capaian Tahun 1) dan satu kalimat penutup G.
2375 -> 2305 kata, tetap 6 halaman.

**Koreksi fase BB (ditemukan penulis): daftar pustaka tanpa sitasi.** Poin H memuat lima rujukan
bernomor, tetapi tidak satu pun `[n]` muncul di badan teks. Template mensyaratkan penomoran
**menurut urutan pengutipan** dan hanya mencantumkan pustaka yang benar-benar disitasi, jadi daftar
tanpa sitasi melanggar keduanya sekaligus. Kelima sitasi disisipkan di titik yang memang memerlukan
rujukan: [1] TRM upstream di C.2, [2] selisih antar alat ukur energi perangkat lunak di C.3,
[3] perilaku pencuplikan sensor daya NVIDIA di C.3, [4] preseden venue Phoeni6 di D.1, dan
[5] tradisi Kaplan pada pembahasan scaling law di F.1. Diverifikasi: urutan kemunculan di badan teks
tepat 1,2,3,4,5 dan himpunan yang didaftar sama dengan himpunan yang disitasi.

**Audit kepatuhan template (diminta penulis) menemukan dua pelanggaran.**

1. **Penjelasan di tiap poin terhapus.** Template menyatakan tegas: "Dilarang menghapus/memodifikasi
   template ataupun menghapus penjelasan di setiap poin." Laporan hanya memuat judul C-H tanpa satu
   pun teks penjelasannya (0 dari 6). Diperbaiki: keenam penjelasan diekstrak **verbatim** dari
   `.docx` dan disisipkan di bawah judulnya masing-masing sebagai kutipan miring, sehingga jelas
   terbedakan dari isi kami.
2. **Tidak ada gambar/grafik.** Poin C menyarankan "Penyajian data dapat berupa gambar, tabel,
   grafik"; laporan baru punya tabel. Ditambahkan dua figur dari naskah: Gambar 1 energi iso-akurasi
   (di C.4, menopang angka hemat 53%) dan Gambar 2 peta rezim (di sintesis C.4).

Sekalian tertangkap: caption pandoc terbaca "Figure 1" pada dokumen berbahasa Indonesia; diperbaiki
dengan `-V lang=id` sehingga menjadi "Gambar 1".

Catatan yang menenangkan: penjelasan poin E ternyata memuat kalimat "untuk penelitian dasar
(KATALIS, Fundamental, ...) boleh mengisi bagian ini (tidak wajib) jika melibatkan mitra", sehingga
jawaban kami yang menyatakan tidak ada mitra sudah tepat.

Verifikasi akhir: 6/6 judul ada, 6/6 penjelasan verbatim ada, tiap poin terisi (C 1197 kata, D 247,
E 42, F 563, G 185, H 172), sitasi badan teks urut 1-5, 2 gambar + 7 tabel, 8 halaman.

**Koreksi lanjutan (ditemukan penulis): gambar tak tampil di markdown.** Figur disalin sebagai `.pdf`
sehingga terrender di PDF hasil pandoc tetapi **kosong** pada penampil markdown (GitHub, editor).
Dikonversi ke PNG 300 dpi (`regime_map.png` 1987x823 px, `iso_akurasi.png` 973x699 px), cukup untuk
kualitas cetak dan tampil di kedua kanal. Salinan `.pdf` di `lapkemajuan/gambar/` dihapus supaya tidak
ada dua berkas gambar yang sama dalam satu folder dan bisa saling menyimpang; sumber vektornya tetap
ada di `manuscript/figures/` dan di paket Zenodo.

**KOREKSI KEJUJURAN (ditemukan penulis): judul penelitian di laporan adalah karangan saya sendiri.**
Laporan kemajuan menulis judul "Frontier Energi-Akurasi Recursion Depth pada Tiny Recursive Models:
Pendekatan Green AI untuk Penalaran Simbolik". Itu **tidak pernah ada**; saya menyusunnya dari isi
naskah alih-alih membaca judul resmi. Judul yang benar, dikonfirmasi dari dokumen proposal yang
diajukan (`proposal/dokumen/document_common_Isian_Substansi_Proposal_...pdf`, bagian A):

> **Hukum Penskalaan Model Kecerdasan Buatan Hemat Energi: Pendekatan Berkelanjutan untuk
> Transformasi Digital Pendidikan Tinggi Indonesia**

Catatan penting: berkas `proposal/A_judul_penelitian.md` memuat judul **berbeda lagi** ("Scaling Laws
untuk Arsitektur Rekursif: Teori Baru Optimasi Energi ..."), yakni draf yang berubah sebelum submit.
**Dokumen PDF yang diajukan adalah rujukan yang sah, bukan berkas markdown draf itu.**

**Konsekuensi ke poin F, dan ini yang membuat koreksinya bukan sekadar kosmetik.** Judul resmi memuat
frasa **Hukum Penskalaan**, yaitu persis hal yang kita putuskan tidak diklaim. Penilai yang membaca
judul lalu tidak menemukan hukum akan langsung menandainya. F.1 ditulis ulang agar menjawab hal itu
di depan, dan pembagiannya dinyatakan tegas:
- **Hukum penskalaan ENERGI tegak dan tervalidasi**: b = 0,84, CI [0,75, 0,92], R2 = 0,98, jadi target
  R2 >= 0,90 terlampaui. Hukum inilah yang menopang perencanaan anggaran energi.
- **Hukum penskalaan AKURASI tidak diklaim**, dengan dua alasan yang sudah dicatat sebelumnya, ditutup
  kalimat: menyajikan satu eksponen akurasi yang rapi sementara datanya berbalik arah akan memenuhi
  janji secara harfiah dan melanggarnya secara ilmiah.

Judul lama disapu; nol sisa di seluruh repo. Laporan tetap 8 halaman.

---

## Fase BC: audit pustaka mengikuti gerbang skill `submit`

Penulis menanyakan apakah kumpulan skill di `github.com/nulis-not-just-writing/skills` membantu tahap
persiapan submisi. Ditelaah, lalu gerbang yang **belum pernah kita jalankan sistematis** dieksekusi.

**Sebagian besar gerbangnya sudah kita lewati lebih dulu** lewat fase AP sampai AX: kecocokan lingkup
(12 paper venue dibaca), batas tipe artikel dan abstrak (Guide for Authors), pernyataan wajib (CRediT,
konflik kepentingan, pendanaan, ketersediaan data, deklarasi AI generatif), judul dan kebaruan,
konsistensi internal lintas berkas, serta kepatuhan templat elsarticle.

**Tiga gerbang yang ternyata belum pernah diperiksa sistematis, dan hasilnya:**

| Pemeriksaan | Hasil | Ambang |
|---|---|---|
| Kemutakhiran pustaka | 46 dari 47 terbit 2021 ke atas (**98%**) | mayoritas 5 tahun terakhir |
| Rasio sitasi diri | **0 dari 47 (0%)** | di bawah 20% |
| Status retraksi | **nol bermasalah**: 29 DOI penerbit dicek ke CrossRef (`update-to`, `relation`, tipe, judul), 16 DOI arXiv dicek ke DataCite | biner-kritis |

Dua rujukan tanpa DOI diperiksa terpisah karena gerbang T2 menandai rujukan tak terverifikasi sebagai
biner-kritis. Keduanya sah: **SharmaKaplan2022** terbit di JMLR dan **YangJoules2025** di PMLR, dua
venue yang memang tidak menerbitkan DOI. URL keduanya hidup (HTTP 200) dan metadata PMLR dicocokkan
langsung ke halaman resminya: judul, penulis, volume 267, halaman 70498-70514, semuanya cocok.

Diperiksa pula bahwa **DiGirolamo2022**, paper tertarik yang ditemukan pada fase sebelumnya, memang
sudah tidak disitasi naskah; ia hanya tersisa sebagai entri yatim di `.bib`.

**Kesenjangan yang tersisa dan tidak dapat ditutup di sini:** gerbang T2b (kesetiaan klaim terhadap
sumber) menuntut teks lengkap tiap rujukan. Kita hanya memiliki 12 PDF paper venue di `../uploads`,
sehingga pemeriksaan menyeluruh atas 47 rujukan belum mungkin. Dua salah kutip yang pernah ditemukan
(Yao2025 dan DiGirolamo2022) tertangkap secara kebetulan, bukan lewat sapuan sistematis.

---

## Fase BD: audit kesetiaan klaim (gerbang T2b) — satu salah kutip ditemukan

Menutup kesenjangan yang ditandai fase BC. Dari 47 rujukan, diidentifikasi **24 yang menopang argumen**
(kalimatnya memuat angka, klaim pembanding, atau penanda klaim), lalu klaim yang kita atributkan
dicocokkan ke sumbernya. Enam belas berhasil diverifikasi; sisanya rujukan latar yang tidak menopang
klaim spesifik.

**Terverifikasi cocok (16):**

| Rujukan | Klaim kita | Bukti di sumber |
|---|---|---|
| He2026 | tuas power cap dan batch size | "jointly optimizes GPU power capping and workload batch size" |
| Yao2025 | prioritas pekerjaan | "adjusting power limits based on job priority" |
| Castano2026 | pemilihan perangkat keras | judul: optimisasi konfigurasi perangkat keras inferensi |
| Guo2025 | penaksir terlatih memprediksi energi tanpa menjalankan | SDEnergy, menghindari "resource-intensive inference" |
| Spillo2026 | algoritma sederhana lebih efisien | "results consistently favored simpler alternatives", beda 50x |
| Desislavov2023 | energi tumbuh jauh lebih lunak | "much softer growth in energy consumption than previously anticipated" |
| JolicoeurMartineau2025 | TRM 7M parameter | "With only 7M parameters, TRM obtains 45% on ARC-AGI-1" |
| Saunshi2025 | kedalaman menggantikan parameter | "require a large depth but not necessarily many parameters" |
| Geiping2025 | kedalaman saat inferensi | "unrolling to arbitrary depth at test-time" |
| Csordas2024 | rekurensi pada kedalaman | Universal Transformers, "recurrence in depth" |
| Tschand2025 | efisiensi energi jadi perhatian utama | "Benchmarking the energy efficiency ... is crucial" |
| YangLooped2024, Li2025, Choshen2024, Caballero2023, OliveiraFilho2025 | sesuai judul dan abstrak masing-masing | |

**SALAH KUTIP DITEMUKAN: Huynh2026.** Naskah menyatakan di bagian Related Work bahwa "optimised large
models emit less CO2 than small baselines". Diperiksa ke tabel hasil paper aslinya, klaim itu **tidak
benar dan arahnya justru terbalik**:

- BERT + Adapter Modules (besar, dioptimalkan): **0,3984 kWh**
- ELECTRA-small (kecil, tanpa optimasi): **0,1065 kWh**

Model besar yang sudah dioptimalkan justru **3,7 kali lebih boros** daripada baseline kecil. Klaim itu
muncul di **tiga tempat** (Pendahuluan, Related Work, Diskusi) dan ketiganya dikoreksi ke temuan yang
benar-benar dilaporkan paper tersebut: adapter modules memangkas energi BERT **27,3%** sambil
**menaikkan** akurasinya ke **0,9305**, yakni penghematan tanpa mengorbankan akurasi.

Argumen kita tidak melemah karena koreksi ini, justru menjadi lebih tepat: yang ditunjukkan kedua
tolok ukur venue bukan "alokasi intuitif selalu salah", melainkan bahwa **pertukaran yang diduga antara
biaya dan mutu tidak terjadi** pada keduanya. Itu bentuk yang sama dengan temuan kita di sumbu rekursi.

**Catatan metode.** Pencocokan DOI awal meleset untuk dua paper (Desislavov2023 dan Yao2025 sebenarnya
ada di `../uploads` tetapi tak terdeteksi), sehingga sempat dikira tak punya teks lengkap. Ditemukan
lewat penyapuan ulang berdasarkan DOI di halaman pertama tiap PDF.

Naskah tetap 40 halaman, 0 rujukan menggantung.

**Sinkronisasi kanal setelah fase BD.** Diperiksa apakah paket Zenodo dan cermin kode ctrm perlu
disegarkan setelah koreksi Huynh2026.

*Zenodo v3: masih mutakhir, tidak perlu versi baru.* Diperiksa lewat riwayat git: **nol perubahan pada
`zenodo/` sejak commit publikasi v3** (67cc980). Koreksi Huynh2026 hanya menyentuh `manuscript/main.tex`,
sedangkan naskah memang tidak ikut ke dalam paket. Perlu dicatat satu jebakan: md5 tarball berbeda tiap
kali dibangun ulang meski isinya identik, karena gzip menyimpan stempel waktu. Diuji dengan membangun
dua kali berturut-turut dan hasilnya memang berbeda, sehingga **md5 tarball tidak sah dipakai
membandingkan isi**; riwayat git yang dipakai sebagai penentu.

*ctrm: tertinggal lima fase.* Jurnal eksperimen di sana berhenti di fase AY (2013 baris) sedangkan repo
sudah di BD (2267 baris). Disegarkan. Kode dan data diperiksa lebih dulu dan ternyata sudah identik;
satu-satunya beda lain adalah berkas log mentah dan README, keduanya memang sengaja berbeda mengikuti
pembagian kanal fase AZ. Commit ctrm `b10f477`.

---

## Fase BE: pemeriksaan visual figur, tiga cacat ditemukan

Kesembilan figur dirender pada 150 dpi dan diperiksa satu per satu, bukan sekadar dicek keberadaannya.

**1. Tumpang tindih pada Figur 4 (frontier energi-akurasi).** Kurva D18 melintasi tepat di atas teks
label "268 Wh (3/3)" sehingga angkanya sulit dibaca, dan label "329 Wh (2/3)" nyaris menempel padanya.
Diperbaiki: posisi label digeser per kedalaman (D9 ke kiri-bawah, sisanya kanan-bawah) dan diberi latar
putih semi-transparan agar tak tertembus kurva.

**2. DUPLIKASI: panel kiri Figur 6 memplot data yang sama persis dengan panel pertama Figur 7.** Keduanya
menampilkan akurasi exact Sudoku terhadap D_eff dengan nilai 62,4 / 50,1 / 36,3 dan bentuk plot yang
sama. Penelaah wajar bertanya mengapa satu hal digambar dua kali. Panel itu dibuang dari Figur 6, yang
kini menyisakan satu panel yang memang tidak ada duanya, yaitu akurasi akhir terhadap energi neto yang
benar-benar terpakai. Caption diperbarui dan menunjuk pembaca ke Figur 7 untuk sumbu kedalaman, serta
memuat angka baru yang diverifikasi: D9 menghabiskan 402 Wh melawan 340 Wh milik D36, yakni hanya
**18% lebih mahal** sambil paling akurat.

**3. Label terpotong dan legenda menutupi data.** Label `$D_9$` pada panel energi terpotong tepi kanan;
diperbaiki dengan offset per titik dan pelebaran batas sumbu. Pada bidang rancangan, legenda menimpa
titik D_eff=36 lalu, setelah dipindah ke kanan-bawah, menimpa penanda baseline; akhirnya ditempatkan di
pita kosong tengah-kanan dan diverifikasi bersih dari seluruh titik.

**Temuan sampingan yang lebih penting daripada ketiganya: tiga figur tidak punya skrip generator.**
`fig_sudoku_frontier`, `fig_width_optimum`, dan `fig_design_plane` dibuat secara ad hoc pada fase awal
dan tak pernah diikat ke kode, sehingga melanggar aturan integritas #1 dan tak bisa diregenerasi bila
data berubah. Ketiganya kini punya generator di `make_manuscript_figures.py`, membaca langsung dari
`recipe_summary.csv`. Bidang rancangan sekaligus diperbaiki isinya: sapuan kedalaman h=256 untuk Maze
dan ARC dulu hanya disebut sebagai teks, kini digambar sebagai titik pudar sehingga bidangnya utuh.

Naskah tetap 40 halaman, 0 undefined.

---

## Fase BF: pemeriksaan visual tabel — satu kebocoran serius ditemukan

Kesembilan tabel dirender dan diperiksa. Tabelnya sendiri rapi, tidak ada yang terpotong. Tetapi
pemeriksaan itu menyingkap cacat yang jauh lebih penting daripada tata letak.

**KEBOCORAN: catatan audit internal tercetak di daftar pustaka.** Field `note` pada 73 entri `.bib`
memuat anotasi kerja kami sendiri, dan `elsarticle-num` mencetaknya. Di PDF muncul **27 kali** dalam
bentuk seperti:

> Scopus EID 2-s2.0-105035350995; type journal-article; pillar P6; verified CrossRef sim=1.00

Ini akan terkirim ke penyunting apa adanya. Diperiksa lebih dulu apakah ada informasi sah yang hanya
tersimpan di sana: tidak ada satu pun entri yang venue-nya bergantung pada `note`, dan sebelas di
antaranya malah rusak (terpotong di `{DOI`). Seluruh 73 field `note` dibuang; 47 sitasi tetap utuh,
naskah menyusut 40 -> 39 halaman.

**Cacat format berikutnya, dan sebabnya bukan gaya melainkan metadata.** Dua belas rujukan tercetak
sebagai `1201--1216doi:10.1109/...`, halaman menempel DOI tanpa pemisah. Ditelusuri: `elsarticle-num`
menghilangkan titik pemisah bila `volume` kosong. Diperiksa ke CrossRef, ternyata **sebelas di antaranya
paper konferensi yang salah ditandai `@article`** (HPCA, CCGrid, SC24, NeurIPS, ICPADS, dan lain-lain),
satu bab buku, dan satu artikel akses-awal. Jadi ini bukan sekadar cacat kosmetik melainkan **jenis
entri yang keliru**. Diperbaiki: 11 menjadi `@inproceedings` dengan `booktitle`, satu menjadi
`@incollection`, dan pada Tschand2026 paginasi sementara `1--9` dibuang karena artikel akses-awal
memang belum berpaginasi tetap. Sisa kasus: nol.

**Cacat pada figur yang ikut tertangkap.** Caption bidang rancangan menyebut penanda baseline sebagai
"star" padahal yang digambar segitiga; dikoreksi. Legenda pada figur yang sama bertabrakan berturut-turut
dengan titik D=36, penanda baseline, lalu titik D=9, dan baru bersih setelah ditempatkan di luar jalur
data. Pada diagram protokol, kotak paling kiri dan paling kanan **terpotong tepi gambar** karena batas
sumbu dipasang persis di tepi kotak; diberi margin, kotak dilebarkan, font 7,3 -> 6,3, dan keenam panah
dihitung ulang dari tepi kotak yang baru.

**Cacat tipografi.** Kolom paragraf Tabel 2 rata kanan-kiri sehingga muncul celah menganga di baris
"Width axis"; diubah menjadi rata kiri lewat `>{\raggedright\arraybackslash}`.

Hasil akhir: 39 halaman, 0 undefined, 0 bocoran internal, 0 halaman menempel DOI, 47 sitasi utuh.

---

## Fase BG: Zenodo v4 diterbitkan

Diperiksa lebih dulu apakah versi baru memang beralasan. Riwayat git menunjukkan tiga berkas berubah
di `zenodo/` sejak commit publikasi v3 (67cc980), dan ketiganya substantif, bukan kosmetik:

| Berkas | Perubahan |
|---|---|
| `README.md` | "37 runs" -> 49; Maze dan ARC "3 seeds" -> 5 |
| `PROTOCOL.md` | Maze dan ARC "3 seeds" -> 5 |
| `code/make_manuscript_figures.py` | tiga generator figur baru (`fig_sudoku_frontier`, `fig_width_optimum`, `fig_design_plane`) yang sebelumnya tak ada, plus perbaikan tumpang tindih |

Dua yang pertama adalah **koreksi fakta**: paket v3 menyatakan jumlah run dan jumlah seed yang salah.
Yang ketiga menutup celah reproducibility, karena tiga figur naskah dulu dibuat ad hoc tanpa kode.

**Prosedur.** Versi baru dibuat lewat `POST .../actions/newversion`, menghasilkan draft id 22182985 yang
mewarisi keempat berkas v3. `THIRD_PARTY_LICENSES.md` tidak berubah sehingga dibiarkan; tiga sisanya
dihapus lalu diunggah ulang. Sebelum menerbitkan, keempat berkas dicocokkan md5 terhadap salinan lokal
dan semuanya identik. Deskripsi metadata diperiksa dan sudah benar sejak v3 (memuat 49 faithful, 91 run,
16,1 kWh, dan rumusan ARC lima seed), sehingga tidak perlu disunting.

**Hasil.** HTTP 202, state `done`.
- DOI versi v4: **10.5281/zenodo.22182985**
- Concept DOI: **10.5281/zenodo.21181342** (dirujuk naskah, tidak berubah)
- Diverifikasi: concept DOI kini me-resolve ke `zenodo.org/records/22182985`, yakni v4.

Tarball 115.557.173 byte, md5 `7266f8c3ee119b0f5434aec1749b2e29`.

---

## Fase BH: audit kelayakan submit menyeluruh + copy-edit per section (2026-09-15)

**Tujuan.** Menjawab "sudah layak submit?" dengan bukti, bukan kesan: kompilasi, sitasi, angka vs artefak,
konsistensi internal, figur/tabel, paket Zenodo, posisi terhadap paper SUSCOM, lalu perbaikan bahasa.

**Yang lolos tanpa perubahan.** 47 sitasi semuanya ada di `references.bib` (27 entri bib tak dikutip,
dibiarkan). Zenodo concept DOI resolve ke v4 (`records/22182985`), cermin `awangga/ctrm` publik.
Skrip figur `zenodo/code/make_manuscript_figures.py` byte-identik dengan versi `eksperimen/`. Semua
angka Maze n=5 (83,17/83,58/84,91; energi 314/303/324 Wh = +3,0%), ARC n=5, Sudoku, headroom (38/69/15),
CO2e (675,9 g/kWh), pilot (6,2/9,4/8,8), overhead evaluasi (1,50/2,99/5,78), konvergensi (+3,1/+1,2/+1,4)
cocok dengan CSV/report. Highlights 5 butir, semua <=85 karakter. Sepuluh figur diperiksa visual: layak.

**Cacat yang ditemukan dan diperbaiki di `main.tex`:**
1. Dua error LaTeX `Missing $ inserted` (Tabel A.1, `5.6\times 10^{-5}` di luar mode math); PDF
   sebelumnya "kebetulan" jadi. Lima em-dash Unicode di subjudul Tabel A.1 diganti titik dua.
2. **Kalimat basi pasca-Maze n=5** yang saling bertentangan dengan abstrak: Discussion masih menulis
   "neutral within noise (Maze, ARC)", "no task shows a benefit from added depth", "null results on
   Maze and ARC"; Related Work "on these tasks the shallowest setting is already past it". Semua ditulis
   ulang mengikuti verdict Maze deep-wins / ARC unresolved.
3. "three seeds" dinyatakan sebagai aturan umum di Methods, Design, Results, caption Fig. 3, Discussion,
   padahal Maze/ARC lima seed. Diperbaiki di semua tempat.
4. **Kalimat driver basi**: "the last six runs ... 98,82-98,88% vs 99,36-99,95%". Faktanya 12 run
   (Maze s3-4 DAN ARC s3-4) pasca-upgrade driver: 98,82-98,93% vs 37 run sebelumnya 99,11-99,95%.
   Dikoreksi di naskah dan `zenodo/README.md` (lokal; rekaman Zenodo v4 masih memuat versi lama).
5. Shim AdamW dan patch progress-logging dirujuk "described in Section 3" oleh Tabel 2 dan
   Reproducibility, tetapi Section 3 tidak pernah menjelaskannya. Ditambahkan satu paragraf.
6. Keyword 7 -> 6 (`compute-optimal` dibuang; batas umum Elsevier 6). Abstrak dijaga 246 kata.
7. Caption Tabel 1 menyebut kolom "power" yang tidak ada; notasi 2.2 `$D$`/`$P$` -> `$D_eff$`/`$h$`;
   "an 19%" -> "a 19%"; `artifact` -> `artefact` (British, konsisten dengan favour/organise).
8. Caption Fig. 1: ditambah bahwa tiap run juga dicek manual (wall time + step count), karena aturan
   <120 s terbukti tidak menangkap run terpotong 161 s (fase AW).
9. Threats: satu kalimat bahwa satu kartu = satu titik hardware, dan runner dirancang agar grid Sudoku
   bisa diulang di kartu kedua (antisipasi permintaan reviewer sistem).
10. `.zenodo.json` lokal: catatan "published on acceptance" sudah tidak benar, dibuang.

**Copy-edit per section (9 agen, satu berkas per section, digabung ulang).** Verifikasi otomatis
(`numcheck.py`): himpunan angka, kunci `\cite`, `\ref`/`\label`, path figur identik sebelum/sesudah,
kecuali koreksi faktual butir 4 dan 9 yang memang disengaja. Kompilasi 0 error, 40 halaman.
Surat pengantar ditambah kalimat scope ("resource management to optimise performance and power") dan
preseden Spillo/Huynh.

**Posisi terhadap SUSCOM (riset agen, DOI terverifikasi via CrossRef/OpenAlex).** Dari 447 artikel
2024-2026 hanya ~12 tentang energi model AI; studi single-GPU skala kecil memang terbit (Spillo 2026: satu
Titan X; Huynh 2026: satu P100, 3 seed; Guo 2025: satu Jetson Nano). Protokol kita (dua instrumen,
koreksi idle, Bonferroni, 3-5 seed) di atas norma venue. Risiko: dibaca sebagai "paper arsitektur ML"
oleh editor arus utama "ML untuk energi"; panjang 40 hal preprint di atas median (~13 hal terset);
permintaan validasi GPU kedua. Acceptance rate/waktu review tidak terverifikasi dari sumber mana pun.

**Keputusan penulis yang tersisa (tidak diambil di sini):** (a) memindahkan Lampiran A/B ke
supplementary untuk memangkas panjang; (b) menerbitkan Zenodo v5 agar README di rekaman memuat koreksi
kalimat driver (concept DOI tetap); (c) konfirmasi peran CRediT.

---

## Fase BI: keputusan lampiran, audit akhir 13 agen, dan temuan aturan checkpoint (2026-09-15)

**Lampiran A/B tetap di dalam artikel.** Varian supplementary dibangun (`supplementary.tex`, Tabel S1/S2,
naskah 40 -> 38 halaman) dan disimpan di commit `2f0c4d6`, lalu dibatalkan. Bukti: Guide for Authors SUSCOM
(PDF lokal 26/08/2026) mengizinkan appendix A, B dengan Table A.1 dan supplementary yang "will not be
checked, formatted or typeset"; dari 12 paper SUSCOM di `uploads/`, satu memuat sembilan appendix di dalam
artikel (Desislavov 2023, `10.1016/j.suscom.2023.100857`, 17 hlm) dan satu menulis "Supplementary material
will be provided upon request" (Huynh 2026). Penghematan hanya 2 halaman; tabel uji hipotesis dan per-seed
lebih aman di-typeset dalam PDF of record.

**Audit akhir: 13 agen audit-only** (4 berkas repo: paket Zenodo, eksperimen/log, cover letter+bib, docs;
9 section naskah), semua angka dihitung ulang dari CSV. Hasil: seluruh angka Tabel 3-7, A.1, B.1, Wh-to-50%,
iso-akurasi, cost model, overhead evaluasi, konvergensi, pilot, agregat energi cocok. Cacat yang ditemukan
dan DITUTUP (angka diverifikasi sendiri sebelum diterapkan):
1. **Langkah terealisasi != nominal** (BLOCKER, terverifikasi dari `progress_*.jsonl`): runner membulatkan
   epoch ke kelipatan 25 evaluasi; Maze 23.950/11.975/11.450, ARC 21.726/10.855/9.019 (nominal 24k/12k/12k);
   produk compute sel D36 Maze -4,4%, ARC -17%. Sudoku utuh. Diungkap di caption Tabel 2 dan Threats;
   arah biasnya melawan D36 di ARC (kasus unresolved), Maze D36 menang meski begitu.
2. Klaim "A > 98,7% pada setiap run yang selesai; satu nilai di bawahnya 95,9%" salah: dua run 31 s
   (95,89 dan 98,59) dan satu pilot selesai 224 s (98,68). Diperbaiki di Sec 2, caption Fig. 1, Threats;
   label figur protokol diubah dari "agreement >= 98.8%" (aturan tolak) menjadi lantai teramati.
3. Min daya sampler: 4,11 W hanya Sudoku; 4,04 W di seluruh log. Daya rata-rata run faithful 141,9-173,5 W
   (naskah lama "80-170 W"). Diperbaiki.
4. "43,8% pada step 25k" adalah interpolasi (checkpoint D9 tiap 2000 step): diganti 43,4% @24k; "leads until
   32k" dirumuskan ulang. "no D9 seed exceeds 83.85" hanya benar untuk checkpoint akhir (lihat temuan terbuka).
5. ARC "26-33% token" sisa n=3 -> 26-31%; "a hundred seeds" (d=0,40 lama) -> "a thousand" (d=0,15);
   "identical on every run" untuk intensitas grid -> hanya 77 run ber-emissions; pilot hidden 128-384 -> 128-768;
   R^2 ~0,6 -> 0,57; Maze "inside three points" -> two; kalimat basi "final accuracy" untuk iso-akurasi Sudoku
   -> best-checkpoint; Tabel 2 baris data Maze/ARC (1000 maze x 8 dihedral; 800 task x 1000 aug; test 512),
   baris Evaluators, baseline 12 heads + ACT loop + energy-matched; Algoritma 1 E(tau) post hoc.
6. Discussion: "extra iterations buy no accuracy and sometimes remove it" dan "pattern holds ... ARC"
   dilunakkan; "fitted to three points" -> dua titik mapan + satu unresolved; "at equal energy" -> matched compute.
7. Intro: klaim prioritas "(or FLOPs)" dibuang dan dibingkai "we know of no study"; "every task we tried"
   untuk sumbu lebar -> Sudoku saja; Bahri2024 dilepas dari klaim manipulation-vs-storage.
8. Bib: 11 judul dengan akronim tak dikurung (MLPerf, AI, GPU, LLM, FLOPs, NOT) diperbaiki; tahun Jin2024/
   Caballero2023 disamakan dengan venue; CRediT "review \& editing". Cover letter: "the result reversed" ->
   "survived the corrected threshold"; tanggal; em-dash; artefact.
9. Paket Zenodo: `rebuild_env.sh` tidak lagi hard-code path rumah; README/PROTOCOL memuat env ARCH, GROUPS,
   TRM_DIR, perintah build Maze/ARC, konstanta idle 4,7 W, butir "planned, not applied" (clock lock,
   train/inference terpisah); requirements + wandb; docstring shim AdamW; `.zenodo.json` "38 points of
   headroom"; `watch_arcdepth_s34.sh` dibuang dari paket; DEPOSITION.md v3/v4 dicatat.
10. Dokumen repo: CLAUDE.md 43 -> 49 run, idle 4-5 W; README dan eksperimen/README daftar `frontier/` nyata;
    `revisi_E_metode_isoflop.md` diberi banner SUPERSEDED; skrip hook atribusi.

Kompilasi 0 error, 41 halaman; `numcheck` vs sebelum audit: semua selisih angka = koreksi di atas.

**TEMUAN TERBUKA (keputusan penulis, belum diterapkan): aturan checkpoint tidak seragam antar task.**
Sudoku (Tabel 3, Algoritma 1: acc* = max_s) memakai checkpoint TERBAIK; Maze dan ARC memakai checkpoint
AKHIR (`final_token_pct`). Dihitung ulang dari `progress_*.jsonl` (`all/accuracy`, 25 checkpoint):

| task | aturan | D9 | D18 | D36 | D36 vs D9 | D36 vs D18 |
|---|---|---|---|---|---|---|
| Maze | akhir (naskah) | 83,17±0,53 | 83,58±0,51 | **84,91±0,58** | p=0,0011 | p=0,0049 |
| Maze | terbaik | 86,66±0,18 | 86,76±0,09 | 86,54±0,29 | p=0,42 | p=0,16 |
| ARC | akhir (naskah) | 31,20±3,28 | 30,68±3,61 | 25,80±0,59 | p=0,0200 (gagal Bonf.) | p=0,038 |
| ARC | terbaik | **36,27±3,18** | 33,34±3,37 | 30,59±1,78 | **p=0,0122 (lolos Bonf.)** | p=0,16 |
| Sudoku | akhir | 62,43 | 48,89±0,60 | 36,20±0,88 | semua p<0,001 | |

Semua konfigurasi Maze memuncak 86,5-86,8% pada checkpoint ke-2 sampai ke-4 lalu TURUN; D36 turun paling
sedikit. Jadi "deep wins" Maze = "deep degrades least after saturation", dan di bawah aturan terbaik Maze
null. ARC di bawah aturan terbaik menjadi shallow-wins signifikan. Sudoku tidak berubah arah pada aturan
mana pun. Dua opsi yang konsisten: (A) aturan TERBAIK untuk semua task, sesuai Algoritma 1: Sudoku shallow
menang, ARC shallow menang (p=0,012), Maze null jenuh; "sign flip" hilang, cerita menjadi "depth tak pernah
membayar; di metrik jenuh tak berpengaruh". (B) aturan AKHIR untuk semua task: cerita sekarang bertahan,
Sudoku D18 50,1 -> 48,9, dan mekanisme Maze harus diungkap sebagai penurunan pasca-puncak. Keduanya wajib
mengungkap dinamika puncak-lalu-turun. Zenodo v5 (draft `22765548`) DITAHAN sampai keputusan ini, karena
README-nya memuat verdict Maze; unggahan tarball juga gagal dua kali (badan kosong, lalu 502) pada laju
~55 KB/s.

---

## Fase BJ: aturan checkpoint terbaik diberlakukan ke semua task (opsi A) (2026-09-16)

**Keputusan penulis:** opsi A dari fase BI, yakni satu aturan untuk semua task: akurasi sebuah konfigurasi
= checkpoint evaluasi TERBAIK (Algoritma 1, `acc* = max_s acc(s)`), seperti yang sejak awal dipakai Sudoku.

**Data.** `add_best_token.py` menurunkan kolom `best_token_pct` dari `progress_<tag>.jsonl` untuk 49 run
faithful (kolom `final_token_pct` dipertahankan); `run_recipe.py` kini meng-emit kolom itu untuk run
berikutnya. `stats_table.py`, `make_manuscript_figures.py` (fig_crosstask, fig_regime_map, perseed,
ARC iso-akurasi baru) beralih ke `best_token_pct`. Skrip disalin ke `zenodo/code/`.

**Hasil (n=5, checkpoint terbaik, dari `ablation/stats_table_contrasts.csv`):**
| task | D9 | D18 | D36 | D9 vs D36 | ANOVA |
|---|---|---|---|---|---|
| ARC | **36,27±3,18** | 33,34±3,37 | 30,59±1,78 | t=3,48 p=0,0122 d=2,20 (lolos Bonf. 0,0167) | F(2,12)=4,90 p=0,028 |
| Maze | 86,66±0,18 | 86,76±0,09 | 86,54±0,29 | p=0,42 | F(2,12)=1,59 p=0,24 |
Kontras berdampingan ARC p=0,20 / 0,16. n=3 (seed 0-2): ARC p=0,14, Maze p=0,11, keduanya null.
Energi: ARC D9 menyamai akurasi terbaik D36 (30,59%) pada 155±82 Wh (per seed 81/105/291/160/137) lawan
278 Wh, hemat 44%; Maze D36 +3% energi (323,7 vs 314,4 Wh) tanpa selisih akurasi. Headroom: Sudoku 38,
ARC 64, Maze 13. Sudoku tak berubah.

**Naskah.** Cerita menjadi: efek kedalaman tidak pernah berbalik tanda; besarnya mengikuti sisa ruang metrik.
Sudoku shallow menang 26 poin, ARC 5,7 poin (signifikan), Maze null jenuh (+3% energi tanpa hasil). Ditulis
ulang: abstrak (249 kata), highlights (5 butir <=85 karakter; `highlights.txt`/`.docx` disinkronkan), Intro
(paragraf kontribusi), Sec 2.4 (aturan checkpoint dinyatakan), Results 5.3 (termasuk pembacaan checkpoint
akhir Maze 83,2/83,6/84,9 sebagai degradasi pasca-puncak, dilaporkan apa adanya), Tabel 4, Fig 7 caption,
5.8 sintesis, Tabel 7 (ARC discriminating/shallow/56%; Maze saturated/none), Fig 9 caption, Related Work,
Discussion, Practical guidance, Threats (paragraf daya statistik ditulis ulang + paragraf baru "checkpoint
rule matters" yang mengakui draf sebelumnya memakai checkpoint akhir untuk Maze/ARC), Conclusion, Lampiran
A (regenerasi dari `stats_table.tex`, subjudul memakai titik dua, bukan em-dash) dan B (nilai per-seed).
Cover letter, `zenodo/README.md`, `.zenodo.json`, `PROTOCOL.md` diselaraskan; blok verdict lama di README
yang sempat terduplikasi dihapus. Kompilasi 0 error, 41 halaman.

**Audit visual figur/tabel (15 halaman dirender):** Fig 1-3, 5-9, Tabel 1-7, A.1, B.1 proporsional dan
konsisten. Diperbaiki: anotasi "268 Wh (3/3)" di Fig 4 bertumpuk dengan kurva (offset digeser); caption
Fig 6 masih "final" -> "best-checkpoint"; titik seed Fig 7 panel Maze terpotong tepi sumbu (margin y).

**Zenodo v5** menyusul setelah commit ini: tarball dibangun ulang dari paket yang sudah memuat kolom
`best_token_pct` dan README baru.

---

## Fase BK: audit ulang sembilan section pasca-opsi A, semua temuan terverifikasi diterapkan (2026-09-16)

Sembilan agen audit-only (satu per section, brief: konsistensi vs artefak/Zenodo, penulisan, reproduksibilitas,
kelayakan Q1) menghitung ulang setiap angka. Semua angka Tabel 3-7, A.1, B.1, iso-akurasi, cost model,
overhead evaluasi, dan agregat energi cocok. Cacat yang ditemukan dan DITUTUP (commit 10c21a4):

1. **Klaim graded "besaran efek mengikuti headroom" tidak ditopang data** (ARC headroom 64 -> efek 5,7;
   Sudoku 38 -> 26). Abstrak, 5.8, Tabel 7 caption, Discussion, Conclusion dilunakkan ke bentuk ordinal:
   efek shallow ADA di mana metrik masih punya ruang, HILANG di mana metrik jenuh; bentuk graded dinyatakan
   belum teruji lintas metrik yang tak sepadan. Ditambah kalimat batas rezim (antara 13 dan 38 poin, belum
   terlokalisasi oleh tiga task).
2. **5.5 "shallow ahead at every point of the budget" salah terhadap baseline**: kurva rata-rata baseline
   memimpin D9 sampai ~207 Wh. Dikoreksi (benar hanya terhadap D18/D36).
3. **Dinamika puncak Maze per seed**: argmax D9 [2,2,2,2,2], D18 [4,4,4,4,11], D36 [4,14,12,21,12]; kalimat
   "semua depth memuncak dalam 4 checkpoint pertama" hanya benar untuk kurva rata-rata. Ditulis ulang di
   5.3, Threats, README Zenodo. ARC iso-akurasi 155±82 Wh dijelaskan per seed (4/5 di bawah 278 Wh, satu 291).
4. **P ∝ h² tidak didukung** (slope log-log 1,49): Sec 2/3 diganti "P set by h alone", label sumbu Fig 3
   dibuang. "49 run reported" -> "48 yang masuk hasil + 1 probe Maze". Pilot: hidden 128-768, best 0-18%.
5. **Geiping2025 bukan model tiny** (3,5 miliar parameter): atribusi di Related Work dikoreksi. Bahri2024
   dipindah ke 6.3. Huynh: trade-off muncul pada konfigurasi GA/Combined, hanya adapter yang cheaper+better.
6. Tabel 2: `GROUPS=3080` untuk ARC (tanpa itu 71k step, bukan 21,7k), langkah terealisasi "seed 0";
   Algoritma 1: E(tau) oleh `make_manuscript_figures.py` (joules_to_target.py hanya pilot; PROTOCOL diberi catatan).
7. Discussion: "moderate width" hanya Sudoku; "always costing energy" hanya pada matched accuracy (iso-compute:
   yang dalam justru lebih murah di Sudoku/ARC); pilot "worst of three" salah (pemenang pilot D18 = tengah,
   memberi 12 poin lebih rendah). Threats: paragraf checkpoint rule dijujurkan (Algoritma 1 memang lebih
   dulu, tetapi draf lama menyimpang untuk Maze/ARC); ARC 26-31% (sisa checkpoint akhir) -> 31-36%.
8. Lampiran A: kontras D18-vs-baseline (p=0,82) yang dikutip 5.4 kini masuk `stats_table.py` (keluarga
   baseline 3 kontras, alpha 0,0167; keduanya tetap lolos). Reproducibility menyebut `add_best_token.py`.
9. Paket Zenodo: agreement per-run = kolom `gross_agree_pct` (tabel rekonsiliasi hanya rentang); OOM
   h768xD36 dan `D_eff=8` baseline dicatat di Known limitations; cover letter "No study we know of".

Naskah 42 halaman, 0 error, abstrak 250 kata, highlights <=84 karakter. Tarball v5 dibangun ulang dari
paket final (md5 `2549fe2377b190239637f8c4c8ee3ef8`); README/PROTOCOL draft 22765548 diganti; unggahan
tarball berjalan (dua percobaan sebelumnya gagal: badan kosong, lalu 502; laju ~55 KB/s). Cermin
`awangga/ctrm` diselaraskan (kode, README, PROTOCOL, artefak ringkas dengan kolom `best_token_pct`).

**Zenodo v5 DITERBITKAN (2026-09-16).** Draft 22765548: `README.md` (md5 c4dc9c3b...), `PROTOCOL.md`
(bb2f457c...), `THIRD_PARTY_LICENSES.md` (tak berubah), tarball 115.593.542 byte (md5
`2549fe2377b190239637f8c4c8ee3ef8`); keempat md5 dicocokkan dengan salinan lokal sebelum publish. Percobaan
unggah ketiga berhasil (HTTP 201, 562 s). DOI versi **10.5281/zenodo.22765548**; concept DOI
`10.5281/zenodo.21181342` terverifikasi me-resolve ke `records/22765548`. Cermin `awangga/ctrm` commit
`f533770`. Naskah, paket Zenodo, dan cermin kini memuat verdict yang sama.

---

## Fase BL: desk screening lima lensa dan perbaikan yang diminta (2026-09-16)

Naskah dinilai sebagai desk editor SUSCOM oleh lima agen independen (scope, kontribusi/novelty, kesehatan
teknis, presentasi/kepatuhan GfA, etika/integritas). Kelimanya: **kirim ke reviewer** (scope dan presentasi
keyakinan tinggi; tiga lainnya sedang). Tidak ada alasan tolak di meja.

Cacat yang ditemukan dan ditutup (commit b18b371 + commit fase ini):
1. Referensi silang rusak "Section efsec:results-crosstask" (sisa karakter CR dari penulisan ulang fase BJ)
   -> "Section 5.3". Label sumbu Fig 3 masih "P ∝ h²" -> dibuang, figur diregenerasi. Tabel A.1 kini memuat
   ketiga kontras lebar (h768 vs h256 p=0,0046 ditambahkan; alpha 0,0167).
2. **Sumber benchmark tidak disitasi** padahal Data availability menyatakan "whose sources are cited":
   ditambah `WangHRM2025` (HRM, arXiv 2506.21734: Sudoku-Extreme, Maze-Hard) dan `Chollet2019`
   (arXiv 1911.01547: ARC), keduanya diverifikasi DataCite; disitir di Sec 4, Tabel 2, Data availability.
3. Literatur terdekat: `JinTTC2025` (arXiv 2505.14733, energi test-time compute LLM) di Intro;
   `GaoLoopie2026` (arXiv 2607.16051) untuk mengakui bahwa "N x parameter mengalahkan N x looping pada
   compute pre-training sama" sudah dikenal; kalimat "cut against the premise" diganti: yang baru adalah
   sisi energinya. Intro menegaskan lingkup = alokasi anggaran pelatihan, energi inferensi di luar ukur.
4. Threats: paragraf baru tiga keterbatasan sumbu kedalaman: (a) sel D36 memakai batch setengah (96 vs 192;
   24 vs 48) tanpa retune lr, sehingga kontras yang melibatkan D36 menguji depth+batch; D9-vs-D18 bersih;
   (b) ACT halting membuat compute per langkah adaptif, Eq. 1 nominal; (c) bias seleksi checkpoint terbaik
   pada subset evaluasi yang sama tanpa held-out, terbesar di ARC.
5. Cover letter: komentar HTML internal dan reviewer ke-5 berlabel "confirm before submitting" dibuang
   (empat reviewer tersisa). Sec 2.3 "49 run (48 masuk hasil + 1 probe)". Kosmetik: "joules" huruf kecil,
   "Fig." di tengah kalimat.

Naskah 43 halaman, 0 error, abstrak 250 kata, 51 sitasi. Catatan untuk reviewer (bukan alasan meja):
signifikansi ARC bertumpu pada sel D36 yang 17% di bawah budget dan batch setengah; regime map tiga titik;
hanya energi pelatihan; panjang di ujung atas jurnal.

---

## Fase BM: simulasi peer review dua reviewer, dan temuan baseline Maze yang mengubah klaim (2026-09-16)

Naskah dinilai oleh dua reviewer tersimulasi dengan profil yang lazim dipakai SUSCOM: R1 pakar
pengukuran energi/sistem, R2 pakar arsitektur rekursif/statistik. Keduanya memverifikasi angka langsung
dari paket Zenodo, keduanya **major revision**, keduanya menyatakan naskah dapat diterima setelah revisi.

### Temuan terbesar: metrik token Maze berada DI BAWAH baseline sepele

R2 curiga Maze berada di lantai kelas mayoritas. Diperiksa dengan `trivial_baselines.py` (metrik
direplikasi persis dari `models/losses.py`: mask label, rata-rata per urutan):

| task | metrik dilaporkan | majority token | copy-input token | model terbaik |
|---|---|---:|---:|---:|
| Sudoku-Extreme | exact | 0,00 | 0,00 | 62,4 |
| ARC-AGI-1 | token | 25,00 | 25,00 | 36,3 |
| Maze-Hard | token | 50,03 | **87,51** | 86,8 |

Sel lintasan hanya 12,5% dari grid 30x30, sehingga **menyalin input sudah mencetak 87,51%**, di atas
setiap konfigurasi yang kita latih (86,5-86,8%). Artinya metrik token Maze **tidak mengukur penguasaan
task** pada anggaran ini, dan "efek kedalaman" di sana adalah lantai metrik. Ini lebih kuat daripada
dugaan reviewer. Konsekuensinya diterapkan apa adanya (aturan #5):

- Baris ketiga peta rezim **runtuh**: Maze dilaporkan sebagai **null measurement**, bukan rezim jenuh.
- **"Cross-task regime map" DICABUT** dari judul, abstrak, sintesis, Tabel 7 (dihapus), dan Fig 9
  (diganti figur baseline-vs-efek). Sumbu pengorganisasi peta (headroom) memang tak sepadan antar
  metrik, persis keberatan R2-M4, dan basis empirisnya tinggal dua titik.
- **Judul baru: "The energy cost of recursion depth in tiny recursive models".**
- Klaim naskah kini: kedalaman tak pernah membeli akurasi; di dua task yang akurasinya bisa dihargai ia
  memakan 26 poin (Sudoku) dan 5,7 poin (ARC); di Maze hanya sisi energinya yang bisa dilaporkan (+3%).
- Kontribusi metodologis naik jadi **tiga jebakan**: biaya evaluasi tumbuh dengan kedalaman; screen
  skala-kecil menempatkan pemenang terakhir; **metrik proksi wajib diadu baseline sepele**. Yang ketiga
  kini jadi aturan repo (CLAUDE.md butir 9 baru).

### Revisi lain yang diterapkan (tanpa run baru)

R1-M1 "dua instrumen" dikoreksi: CodeCarbon dan `nvidia-smi` membaca sensor NVML yang sama, jadi
kesepakatan 98,8-99,95% membatasi galat integrasi 1 Hz, bukan akurasi sensor. Versi driver
(595.71.05 -> 595.84) dan setelan CodeCarbon (3.2.8, `measure_power_secs=5`, `tracking_mode=machine`)
masuk Tabel 2. R1-M2: cost model divalidasi ke 49 run (`validate_costmodel.py`): over-predict Sudoku
+15..+24%, meleset >90% di Maze/ARC (seq 900 lawan 81), b naik ke 0,94 pada subset P*D<50, fit dua
eksponen P^0,82 D^0,92; klaim "validated tool for planning" dicabut dan dibatasi ke rezim ukurnya.
R1-M3: resolusi E(tau) = grid checkpoint (~16 Wh Sudoku, 12-14 Wh Maze/ARC), dinyatakan sebagai batas
atas terkuantisasi. R1-M4: paragraf "what is and is not measured" (GPU-only; CPU/RAM adalah model;
faktor grid statis; energi net melacak wall time pada plateau 140-175 W). R2-M2: sensitivitas estimator
(`estimator_sensitivity.py`) 5,67/5,40/5,52/3,29 poin dengan Welch p 0,0122/0,0200/0,0164/0,0241 dan
permutasi eksak 0,0159/0,0079/0,0079/0,0079 (lantai 0,0079) dilaporkan di Results. R2-M5: lima rujukan
2026 terverifikasi DataCite ditambahkan (Schwethelm iso-depth phi=0,46; DeepLoop; Ingolfsson quantizing
TRM/HRM tiga task yang sama; Jim compression cell-vs-exact; Ren mekanistik HRM), dengan pernyataan apa
yang ditambahkan pengukuran joule ini. R2-M7: `PREREGISTRATION.md` kini ikut paket Zenodo. Minor:
langkah ACT 9,6-14,9 per contoh, subset 512 = 512 pertama (bukan acak), laju family-wise seluruh naskah
(15 kontras, alpha 0,0033: Sudoku dan baseline lolos, ARC tidak), Maze +3% lawan D9 tetapi +7% lawan D18,
kontras lebar ketiga (h768 vs h256, p=0,0046) masuk Tabel A.1.

Naskah 45 halaman, 0 error, abstrak 241 kata, 56 sitasi, nol referensi menggantung.

### Yang TIDAK bisa dijawab tanpa run baru, dan sudah dipra-registrasi

`prereg_BM.md` (dokumen terpisah, di-commit bersama entri ini, SEBELUM run apa pun): 18 run terkunci,
yakni D36 dengan gradient accumulation di ARC (5) dan Sudoku (3) untuk membuang konfon batch, baseline
non-rekursif iso-compute di ARC (5), dan ulangan dengan logging prediksi per-instance (5) supaya subset
512 bisa dibelah jadi separuh-seleksi dan separuh-pelaporan. Kontras yang ditargetkan, uji, dan komitmen
melaporkan hasil apa pun, termasuk **pencabutan klaim 5,7 poin ARC bila signifikansinya hilang**, ada di
dokumen itu. Perkiraan 34 jam GPU. Bila penulis memilih submit lebih dulu, keterbatasan ini sudah
tertulis di Threats.

---

## Fase BM (lanjutan): patch akumulasi diverifikasi, rantai 18 run diluncurkan (2026-09-16)

**Patch ketiga ke pohon ter-vendor** (`patch_pretrain_accum_preds.py`), keduanya OFF secara default:
`TRM_ACCUM=N` (gradient accumulation) dan `TRM_EVAL_PREDS=<dir>` (akurasi token per-instance tiap
checkpoint, untuk pembelahan separuh-seleksi/separuh-pelaporan). Carry ACT membuat batch berperan
sebagai kolam slot paralel, sehingga N kolam micro yang hidup berdampingan setara satu batch N x micro;
`total_steps` dibagi N agar `train_state.step` tetap menghitung langkah optimizer, dan EMA hanya
diperbarui di batas akumulasi.

**Uji kesetaraan (ARC D9 pendek, EPOCHS=20, `accum_check_out/`):**

| | batch 48, accum 1 | micro 24 x accum 2 |
|---|---:|---:|
| langkah optimizer | 1240 | 1201 |
| micro-batch | 1240 | 2402 (tepat 2x) |
| token acc akhir | 22,68% | 23,00% |
| energi net | 20,78 Wh | 20,46 Wh |
| wall | 508 s | 515 s |

Kurva train-loss berimpit (selisih per desil <=10%, dalam derau urutan data), overhead wall 1,4%.
Catatan jujur: akumulasi kehilangan ~3% langkah optimizer (1201 lawan 1240) karena micro-batch sisa di
ujung tiap epoch tidak menutup jendela akumulasi. Dicatat, bukan dikoreksi.

**Jebakan yang nyaris merusak seluruh batch: `GROUPS` adalah variabel khusus bash.** `export GROUPS=3080`
TIDAK ikut ke environment anak (begitu pula bentuk awalan `GROUPS=3080 cmd`); hanya `env GROUPS=3080 cmd`
yang bekerja. Percobaan pertama diam-diam memakai default 1000 sehingga epoch salah. Runner ARC lama
memang sudah memakai bentuk `env`, jadi grid lama aman. Rantai baru memakai bentuk `env` dan
environment proses diperiksa lewat `/proc/<pid>/environ` setelah peluncuran.

**Rantai `run_BM_chain.sh` diluncurkan** (preflight: 0 compute app, 5,36 W; sesudahnya 1 compute app,
1 rantai). A1 seed 0 berjalan dengan `epochs=75 eval_interval=3 L_cycles=12 global_batch_size=24
TRM_ACCUM=2 GROUPS=3080`, memori GPU 7,4 GB. Notifikasi Telegram per run, commit+push per batch.
Analisis pra-spesifikasi ada di `analyze_BM.py`, ditulis sebelum hasil ada.

---

## Fase BN: copy-edit sembilan section, audit visual figur/tabel, dan penyelarasan tiga kanal (2026-09-16)

Naskah hasil penulisan ulang fase BM belum pernah diperiksa ulang secara menyeluruh, jadi dijalankan 16
agen paralel: sembilan menyunting satu section masing-masing di berkas terpisah, empat memeriksa figur dan
tabel secara visual, tiga menyelaraskan dokumentasi repo kerja, paket Zenodo, dan cermin `ctrm`. Enam agen
sempat mati karena batas sesi API; tiga di antaranya sudah menulis suntingan sebelum mati, jadi berkasnya
diverifikasi satu per satu lalu agen pengganti diminta memeriksa hasil separuh jadi itu lebih dulu.

**Penggabungan diverifikasi.** Himpunan angka, kunci `\cite`, `\ref`/`\label`, path figur, dan keseimbangan
lingkungan identik dengan sebelum suntingan, kecuali dua penghapusan sengaja yang keduanya duplikasi
caption-versus-badan: rentang daya 140-175 W (tetap ada di paragraf cakupan) dan angka langkah 24k/32k/43,4/48,9
di caption Fig. 5 (tetap ada di badan). Setiap angka diperiksa masih muncul minimal sekali. Kompilasi
46 halaman, 0 error, abstrak 241 kata, highlights <=81 karakter.

**Dua cacat figur yang mengubah pesan, ditemukan hanya lewat pemeriksaan visual.**
1. **Fig. 9 tidak menyampaikan klaimnya.** Pada panel kiri berskala akurasi absolut 0-100%, marker Maze
   (86,5-86,8%) dan garis baseline 87,51% terpisah kurang dari 1% lebar sumbu, sehingga mustahil melihat
   marker ada di bawah garis; klaim utama figur hanya hidup di caption. Panel kiri diubah menjadi **jarak
   ke baseline sepele**: Sudoku +36,3 sampai +62, ARC +5,6 sampai +11,3, Maze -1,0 sampai -0,7.
2. **Fig. 7 panel Maze menyesatkan.** Rentang sumbu-y 86,2-87,0 tidak memuat garis 87,51, sehingga profil
   terbaca seolah punya puncak di D18. Tiap panel kini menggambar baseline metriknya (0 / 25,0 / 87,51)
   dan rentang-y diperluas agar garis itu masuk.
3. **Fig. 1** melanggar ambang artwork Elsevier: font kotak 6,3 pt tercetak ~5,1 pt (minimum 7 pt) dan teks
   tembus kotak. Kotak diperlebar, font dinaikkan, dan panah penolakan dipindah dari kotak cross-validate
   ke kotak training serta diubah jadi abu, karena kesepakatan instrumen **bukan** gerbang buang, persis
   yang dinyatakan caption.

**Kontradiksi teks yang ditutup.** Caption Fig. 3 menyebut sumbu $(P,D_\text{eff})$ padahal sumbunya $h$,
dan mengklaim seluruh titik iso-compute padahal baseline energy-matched. Sec 5.7 masih menyebut cost model
"validated tool for planning" padahal Sec 3 kini melaporkan galat prediksinya (+15..24% Sudoku, >90% meleset
di seq 900). Abstrak menulis 87,5 sedangkan angka terverifikasi 87,51. Algoritma 1 menyebut skrip tanpa
awalan `code/`. Proksi token tak lagi disebut "informative" tanpa syarat.

**Tiga kanal diselaraskan.** Repo kerja (README, eksperimen/README, CLAUDE.md), paket Zenodo, dan cermin
`ctrm` (kode fase BM disalin, tanpa `progress_*.jsonl`). Arsip pra-registrasi diberi **catatan editorial**:
kutipannya tetap verbatim, termasuk bahasa "regime map" yang kini dicabut, karena pra-registrasi yang
ditulis ulang setelah hasil keluar tidak ada gunanya; pembaca diberi tahu di teks pembungkusnya.

Rantai 18 run fase BM terus berjalan selama seluruh pekerjaan ini, tidak terganggu.

---

## Fase BO: butir reviewer minor dituntaskan, tiga klaim dikoreksi dari artefak (2026-09-16)

Gelombang lima agen berkas-disjoint (satu pemilik per berkas) menutup sisa butir revisi R1/R2 yang tidak
memerlukan run baru, sementara rantai 18 run fase BM masih berjalan (run 1/18 saat entri ini ditulis).

### Naskah (commit `52e9194`)

- **Section 2.** Lantai idle dijelaskan sebagai lantai kartu dingin, bukan level kartu bekerja. Dihitung
  ulang dari 49 berkas `pw_*.csv` faithful: **14 run pernah turun di bawah 5 W, 32 tidak pernah membaca
  di bawah 8,8 W, tiga sisanya di antara keduanya** (min-of-mins 4,08 W, max-of-mins 11,32 W). Sisa draw
  konteks CUDA yang tertinggal di energi net paling besar 2%, tak cukup membalik urutan pasangan mana pun.
  Jendela integrasi dinyatakan memuat penyiapan dataset/embedding (100-120 W, 7-58 s). Klaim "tabel rilis
  memberi indeks checkpoint" dikoreksi: yang dicatat adalah nomor **step**. Kalimat plateau 140-175 W
  dibatasi ke fase training saja agar tidak bertabrakan dengan fase penyiapan.
- **Section 3/4.** Tiga fakta kalibrasi dibuat terbaca sebagai tiga; caption Tabel 2 diramping (langkah
  terealisasi, `GROUPS=3080`, deskripsi baseline pindah ke badan teks); cakupan cost model dibatasi ke
  rezim ukurnya; pembuka Section 4 diperbaiki jadi "All comparisons **between TRM configurations** are
  iso-compute", karena baseline justru energy-matched.
- **Section 5.** Hasil Sudoku kini dipimpin fakta per-seed (urutan bertahan di setiap seed, rentang tidak
  tumpang tindih) dan statistik Welch menyusul, menjawab keberatan df 2,1-2,8 yang rapuh. Definisi
  Wh-to-50% tidak diulang di tiap caption. Caption Tabel 4 merujuk Tabel A.1. Header kolom jadi
  "Target (%)". Subbagian screen murah dipadatkan. Pembacaan iso-energi sumbu lebar ditambahkan
  (h512 mencapai 38,02% pada 213-216 Wh lawan 503 Wh milik h256; 42,97% pada 256-274 Wh lawan 313 Wh
  milik h768).
- **Section 1/9.** Triad jebakan diseragamkan di abstrak, highlights, Contribution, dan Conclusion
  (sebelumnya Contribution memakai triad berbeda). ARC diberi kualifikasi "just inside the corrected
  threshold". Caption Tabel A.1/B.1 diramping; kolom seed 3/4 yang kosong pada baris Sudoku dinyatakan
  kosong **by design**, bukan data hilang.
- **Abstrak** dipangkas ke **249 kata** (sempat 262 setelah semua tambahan).
- **Highlights**: butir Maze ditulis ulang. Versi lama "Where the metric saturates (Maze), depth buys
  nothing" masih memakai bahasa rezim jenuh yang sudah dicabut; sekarang "On Maze the proxy metric never
  beats copying the input, so depth cannot be judged".

### Tiga koreksi yang lahir dari pengecekan ulang ke artefak

1. **Plateau ARC D36.** Naskah menulis "peak at the sixth or seventh of 25 checkpoints and then hold
   between 25,3 dan 26,8%". Dihitung ulang dari lima `progress_*d36*.jsonl`: puncak memang di checkpoint
   6 atau 7 (28,65-33,39%), tetapi checkpoint terakhir jatuh ke **25,07-26,51%**. Angka lama adalah sisa
   dari era n=3. Diperbaiki jadi 25,1-26,5%, dan "while the shallow cells keep moving" diganti pernyataan
   terukur "the $D_9$ seeds peak between the tenth checkpoint and the last" (puncak D9 di checkpoint
   10/15/19/23/25).
2. **Statistik baseline.** Badan teks memakai $t{=}10.2$ untuk dua kontras berbeda. Dari
   `stats_table_contrasts.csv`: D9 vs non-rekursif $t{=}+10.22$, D36 vs non-rekursif $t{=}-10.17$. Tanda
   negatif itu penting karena kontras kedua berarti TRM terdalam **kalah**.
3. **Crossover baseline.** "The non-recursive control leads for roughly the first 200 Wh" diverifikasi
   dengan menggabungkan `pw_*.csv` dan `progress_*.jsonl` ke kurva akurasi-lawan-energi: baseline unggul
   pada 140-210 Wh, lalu disusul ketika keduanya masih sekitar 45%. Kalimat disesuaikan. Pada kesempatan
   yang sama klaim "the shallow configuration is ahead of both deeper TRM settings at every point of the
   budget" **diverifikasi benar** pada grid 20-340 Wh (D9 di atas D18 dan D36 di setiap titik).

### Lantai cross-validasi dipisah dua

Threats menulis "every completed run agrees to at least 98,68%" sementara abstrak memakai 98,8%. Dicek ke
75 baris summary: dua nilai terendah (95,89% dan 98,59%) berasal dari dua run pilot yang gagal setelah
~31 s, dan setelah keduanya dikeluarkan lantai memang 98,68% untuk seluruh run selesai, 98,8% untuk run
yang menopang hasil. Kedua angka kini dinyatakan bersama, bukan bersaing.

### Frasa "dua instrumen" yang masih tersisa

Koreksi fase BM (CodeCarbon dan `nvidia-smi` membaca sensor NVML yang sama) belum sampai ke tiga tempat:
Section 5.7, Threats, dan surat pengantar. Ketiganya diganti jadi "two accounting paths" dengan rujukan ke
Section 2; label `\label{sec:protocol}` ditambahkan ke Methodology untuk rujukan itu.

### Figur (commit `41f6261`)

Enam figur diperbaiki tanpa mengubah satu angka pun: Gbr 2 (garis acuan slope 1, warna+penanda per
$D_{\text{eff}}$, font naik, label sumbu $P\cdot D_{\text{eff}}$), Gbr 3 (anotasi melayang jadi entri
legenda), Gbr 4 (sumbu CO2e diberi faktor konversi, arah anotasi diseragamkan, D18/D36 dibedakan
linestyle), Gbr 5 (checkpoint terbaik D18 ditandai), Gbr 6 (titik per-seed + error bar, encoding warna
disamakan), Gbr 8 (label persen dinaikkan di atas error bar, akurasi target ditulis di bawah tiap
pasangan batang). Caption Gbr 3 disesuaikan karena titik (256,18) digambar sebagai lingkaran faded yang
melingkari kotak width-sweep, bukan digeser: menggeser akan menggambar konfigurasi yang tak pernah
dijalankan.

### Skrip analisis

`analyze_BM.py` sebelumnya menghilangkan bagian A1/A2/B secara diam-diam bila summary CSV belum ada,
sehingga laporan kosong bisa terbaca sebagai null. Sekarang tiap bagian yang datanya belum lengkap
mencetak keterangan eksplisit. Tidak ada uji, kontras, atau ambang yang berubah, jadi pra-registrasi tetap
utuh.

### Bahan laporan akhir

`lapakhir/BAHAN_perubahan_setelah_lapkemajuan.md` dibuat (commit `dfcdc2d`): tujuh butir perubahan klaim
sesudah laporan kemajuan dikirim ke BIMA, supaya laporan akhir bisa menjelaskan selisihnya. `lapkemajuan/`
sendiri tidak disentuh.

**Status:** 48 halaman, 0 error LaTeX, 0 rujukan menggantung, satu overfull 2,61 pt (bawaan float).
Commit: `41f6261`, `52e9194`, `48ec0b3`, `dfcdc2d`. Rantai BM masih berjalan.

---

## Fase BP: judul diganti agar membawa temuan (keputusan penulis, 2026-09-16)

Penulis menilai judul fase BM ("The energy cost of recursion depth in tiny recursive models") akurat
tetapi datar: ia menyebut topik, bukan hasil. Dua alasan teknis mendukung penggantian.

1. **Pola judul venue.** Dari 12 paper SUSCOM yang disurvei, bentuk dua klausa dengan titik dua dan
   penyebutan pertukaran secara eksplisit adalah pola dominan: "Trends in AI inference energy
   consumption: Beyond the performance-vs-parameter", "Balancing carbon footprint and algorithm
   performance in recommender systems", "Trade-offs between power consumption and response time in deep
   learning systems".
2. **Temuan kita lebih kuat daripada judulnya.** Klaim inti (kedalaman tak pernah membeli akurasi) sudah
   berdiri di abstrak tetapi tidak terbaca di judul.

**Judul baru:** *The energy cost of recursion depth: deeper tiny recursive models never bought accuracy
at a fixed training budget*.

Pagar anti-overclaim yang sengaja dipertahankan: kata "energy" tetap di klausa pertama (sinyal scope
supaya editor tidak menggolongkan naskah sebagai paper arsitektur ML), dan "at a fixed training budget"
membatasi klaim ke rezim yang benar-benar diukur. Frasa "never bought accuracy" dipilih, bukan "shallower
is more accurate", karena yang terakhir tidak sah di Maze-Hard: di sana metriknya di bawah baseline
copy-input sehingga akurasi tak bisa dihargai sama sekali. "Never bought accuracy" benar di ketiga task.

**Cacat yang ketahuan saat menyapu judul:** blok `highlights` di `main.tex` dan berkas `highlights.txt`
sudah **tidak sinkron**. Dua suntingan fase BO (butir 1 dicakupkan ke tiga task, butir Maze dilepaskan
dari bahasa saturasi) hanya masuk ke `highlights.txt`, sementara PDF mengambil dari `main.tex`, sehingga
halaman Highlights yang tercetak masih memuat versi lama. Keduanya kini diseragamkan ke satu himpunan
lima butir, masing-masing <= 85 karakter. Pelajaran: kedua sumber itu harus diperlakukan sebagai satu
berkas, dan halaman Highlights wajib dirender dan dibaca, bukan hanya `highlights.txt` yang di-grep.

Kanal yang disapu: `main.tex`, `highlights.txt`, `cover_letter.md`, `README.md`, `zenodo/README.md`,
`zenodo/.zenodo.json`, cermin `ctrm/README.md`, `CLAUDE.md`, dan
`lapakhir/BAHAN_perubahan_setelah_lapkemajuan.md`. `EXPERIMENT_LOG` tidak ditulis ulang (append-only):
judul lama tetap apa adanya di entri fase BM, penggantiannya dicatat di entri ini.

**Status:** 48 halaman, 0 error, halaman Highlights dirender dan diperiksa. Rantai BM tetap berjalan
(run 2/18).
