# Pre-registration records

Every batch of runs added after the first grids was pre-registered before it was launched: the run
count, the contrast it targeted and the commitment to report whatever came out were written down and
committed first, and the analysis was run once, after the last run finished. Three such batches exist.

- **ARC-AGI-1, three seeds to five** (phase AW of the working journal, commit `6ad7c3b`). Reproduced
  **verbatim below**, in Indonesian, the language of the journal. Outcome reported in the manuscript and
  in `README.md`: D9 leads D36 by 5.7 token points, Welch p = 0.012, which clears the Bonferroni
  threshold of 0.0167.
- **Maze-Hard, three seeds to five** (phases AF and AG of the journal: AF fixes the six runs and the
  commitment, AG reports the outcome). Not reproduced here, because the entry is written in the
  superseded framing this package no longer uses, and because the accuracy outcome it targeted has since
  been withdrawn: `code/trivial_baselines.py` showed that copying the input scores 87.51% token accuracy
  on the same 512-instance subset while every trained Maze configuration reaches 86.5-86.8%, so the Maze
  grid is now reported as a **null measurement** and contributes only its energy comparison. The
  pre-registered commitment is what forces that correction to be reported rather than quietly dropped.
- **Phase BM, 18 runs** answering the reviewer requests that need GPU time (deepest cell rerun with
  gradient accumulation, non-recursive control on ARC-AGI-1, per-instance evaluation logging):
  `prereg_BM.md` in this package, with amendment 1, both written before any run was launched. The batch
  was **still running** when this version was assembled, so it has no outcome to report here.

The working journal itself (`eksperimen/frontier/EXPERIMENT_LOG.md`, every phase from A onward) is an
internal laboratory notebook of the private working repository and is not redistributed; the entries
that bind a reported result are the ones above.

**Editorial note (September 2026).** The entry below is reproduced verbatim as it was committed, so it
still uses the framing that was current when it was written, including the phrase "regime map" and the
expectation that ARC-AGI-1 would become a third regime. That framing was later retracted: a
trivial-baseline check showed that the Maze-Hard token metric never clears a copy-the-input predictor,
so the manuscript reports two measured tasks and one null measurement rather than a map of regimes.
What the entry fixes, the six runs, their configuration and the commitment to report any outcome, is
what binds; its motivating language does not. Pre-registrations are left unedited on
purpose: one that is rewritten after the fact is worthless.

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
