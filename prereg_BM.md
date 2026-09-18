# PRA-REGISTRASI fase BM (ditulis dan di-commit SEBELUM run dijalankan)

Tanggal: 16 September 2026. Pemicu: dua laporan peer review tersimulasi (fase BM), keduanya
*major revision*. Tiga permintaan mereka tidak bisa dijawab dari artefak yang sudah ada dan
memerlukan run baru. Dokumen ini mengunci desain, jumlah run, kontras yang ditargetkan, dan
komitmen pelaporan, sesuai aturan CLAUDE.md #7.

## Latar: mengapa run ini perlu

**(A) Sel D36 terkonfon batch.** Kartu 16 GB tidak memuat batch penuh pada `D_eff=36`, sehingga
sel terdalam tiap grid berjalan pada setengah batch tetangganya (Sudoku 96 lawan 192; Maze/ARC 24
lawan 48) tanpa penyetelan ulang learning rate. Setiap kontras yang melibatkan D36 karena itu
menguji kedalaman **dan** batch sekaligus. Bukti tambahan dari log yang sudah ada: kelima seed ARC
D36 memuncak di checkpoint ke-6 atau ke-7 dari 25 lalu bertahan di 25,3-26,8%, pola yang lebih mirip
optimisasi macet daripada efek kedalaman.[^koreksi-plateau]

[^koreksi-plateau]: KOREKSI (fase BO, 16 September 2026, sesudah dokumen ini dikunci): angka
    "25,3-26,8%" dihitung ulang dari kelima `progress_*d36*.jsonl` dan yang benar adalah
    **25,07-26,51%** pada checkpoint terakhir, dengan puncak 28,65-33,39% di checkpoint ke-6 atau
    ke-7. Teks di atas dibiarkan apa adanya karena pra-registrasi tidak ditulis ulang; koreksi
    dicatat di sini. Alasan menjalankan run tidak berubah. Karena yang membatasi adalah memori, bukan compute,
**gradient accumulation** memulihkan batch efektif pada compute yang sama.

**(B) Kontrol non-rekursif hanya ada di Sudoku.** Baseline transformer 8-layer dijalankan pada
Sudoku saja, energy-matched, dengan positional encoding berbeda. ARC adalah task lintas-task yang
metriknya informatif, jadi kontrol di sana yang paling menambah nilai.

**(C) Bias seleksi checkpoint tidak bisa dibuang dari artefak lama.** Skor dipilih sebagai checkpoint
terbaik pada subset 512 yang sama yang dilaporkan. Memisahkan separuh-seleksi dan separuh-pelaporan
menuntut prediksi per-instance, sedangkan runner hanya menyimpan akurasi agregat per checkpoint dan
tidak menyimpan bobot. Jadi ini **tidak dapat** dikerjakan ulang dari log; perlu run dengan logging
per-instance.

## Komitmen yang mengikat

1. **Jumlah run dikunci: 18.**
   - Batch A1: ARC-AGI-1 `D_eff=36`, h256, **micro-batch 24 x accum 2 = batch efektif 48**, langkah
     disamakan dengan sel D18 (10.855 langkah terealisasi), seed 0-4. **5 run.**
   - Batch A2: Sudoku-Extreme `D_eff=36`, h512, **micro-batch 96 x accum 2 = batch efektif 192**,
     25k langkah, seed 0-2. **3 run.**
   - Batch B: ARC-AGI-1 baseline non-rekursif (`ARCH=transformers_baseline`, h256, 8 layer),
     iso-compute dengan sel D9 ARC, seed 0-4. **5 run.**
   - Batch C: ulangan evaluasi dengan **logging prediksi per-instance** pada satu sel per task
     (Sudoku D9, ARC D9, Maze D9), seed 0-4 untuk ARC dan seed 0-2 untuk Sudoku/Maze dipilih
     sebanyak **5 run** total (ARC D9 s0-s4), cukup untuk membelah subset 512 menjadi separuh
     seleksi dan separuh pelaporan pada task yang kontrasnya diperdebatkan.
   Tidak lebih, tidak kurang. Perkiraan waktu GPU: A1 ~9 jam, A2 ~7 jam, B ~9 jam, C ~9 jam;
   total sekitar **34 jam** pada RTX 5060 Ti, ditambah overhead akumulasi.

2. **Kontras yang ditargetkan, dinyatakan di muka.**
   - A1: ARC `D9` lawan `D36` (token, checkpoint terbaik). Nilai sekarang, dengan batch terkonfon:
     selisih 5,67 poin, Welch p=0,0122, ambang Bonferroni 0,0167.
   - A2: Sudoku `D18` lawan `D36` dan `D9` lawan `D36` (exact). Nilai sekarang: 13,8 dan 26,2 poin.
   - B: ARC `D9` (TRM) lawan baseline non-rekursif pada compute yang sama.
   - C: ARC `D9` lawan `D36` di bawah skema separuh-seleksi/separuh-pelaporan.

3. **Analisis dilakukan SEKALI, setelah seluruh 18 run selesai.** Dilarang mengintip lalu berhenti
   saat p sudah lolos. Uji yang dipakai sama persis dengan yang sekarang: Welch dua-sisi, Bonferroni
   0,05/3 per sumbu, plus uji permutasi eksak sebagai pendamping.

4. **Komitmen pelaporan, apa pun hasilnya.** Secara khusus, ketiga kemungkinan berikut akan ditulis
   apa adanya di naskah:
   - Bila selisih ARC menyusut dan **kehilangan** signifikansi setelah batch dipulihkan, klaim
     "5,7 poin, p=0,012" **dicabut** dari abstrak, highlights, dan Results, dan ARC dilaporkan
     sebagai null. Konsekuensinya naskah tinggal punya satu task dengan sumbu akurasi berbayar
     (Sudoku); itu diterima.
   - Bila selisih Sudoku menyusut, angka 26 poin dikoreksi ke nilai baru, termasuk bila klaim
     "shallow menang" melemah.
   - Bila baseline non-rekursif ARC mengalahkan TRM dangkal, itu dilaporkan sebagai bukti yang
     melemahkan nilai rekursi secara umum, bukan disembunyikan.

5. **Artefak wajib per run** tetap sama (aturan CLAUDE.md #6): `progress_<tag>.jsonl`,
   `pw_<tag>.csv`, `emissions_<tag>.csv`, plus baris summary self-describing. Batch C menambah
   `preds_<tag>_ckpt<k>.npz` (prediksi per-instance pada subset 512).

6. **Prosedur peluncuran** mengikuti aturan #8: `nvidia-smi --query-compute-apps` harus menunjukkan
   0 proses sebelum start, 1 sesudahnya; `setsid nohup ... < /dev/null &`; watcher Telegram per run.

7. **Bila run tidak jadi dijalankan** (misalnya penulis memutuskan submit lebih dulu), naskah harus
   menyatakan keterbatasan ini secara eksplisit di Threats, sebagaimana sudah dilakukan pada versi
   sekarang untuk konfon batch dan bias seleksi checkpoint. Dokumen ini tetap berlaku sebagai
   rencana yang terikat bila run dijalankan kemudian.

---

## AMANDEMEN 1 (16 September 2026, ditulis SEBELUM run apa pun diluncurkan)

**Masalah yang ditemukan saat menyiapkan runner.** Rencana awal menulis "langkah disamakan dengan sel
D18 (10.855 langkah)". Itu keliru secara desain: pada batch efektif 48, menjalankan 10.855 langkah
berarti mengonsumsi **dua kali** contoh dibanding sel D36 lama, sehingga `C = P x D_eff x B x S` naik
dua kali lipat dan perbandingan berhenti iso-compute. Persis batasan yang membuat sel D36 lama memakai
batch setengah: itu bukan semata kompromi memori, melainkan bagian dari kendala iso-compute.

**Koreksi yang dipakai: samakan JUMLAH EPOCH, bukan jumlah langkah.** Jumlah contoh yang dikonsumsi
ditentukan oleh epoch, bukan oleh batch. Dengan menahan `epochs` persis sama seperti run D36 lama,
compute identik (produk `P x D_eff x B x S` sama persis), dan yang berubah hanya alokasinya: batch
efektif dua kali lipat, langkah optimizer setengahnya.

| sel | lama | baru (amandemen) | epoch | contoh dikonsumsi |
|---|---|---|---|---|
| ARC D36 | batch 24, ~9.019 langkah | micro 24 x accum 2 = 48, ~4.510 langkah | 75 (sama) | sama |
| Sudoku D36 | batch 96, 25.000 langkah | micro 96 x accum 2 = 192, 12.500 langkah | 2400 (sama) | sama |

**Konsekuensi tafsir, dinyatakan di muka.** Pada anggaran tetap, batch besar berarti langkah sedikit;
tidak ada satu alokasi pun yang "bebas konfon". Karena itu kedua alokasi akan **dilaporkan berdampingan**
sebagai kurung: alokasi lama (batch kecil, langkah banyak) dan alokasi baru (batch besar, langkah
sedikit). Bila defisit D36 bertahan pada keduanya, efek kedalaman kokoh terhadap pilihan alokasi; bila
hilang pada alokasi baru, klaim ARC dicabut dan defisit Sudoku dinyatakan sebagian sebagai artefak
alokasi. Keduanya akan ditulis apa adanya.

**Perubahan lain:** batch B (baseline non-rekursif ARC) disamakan epoch-nya dengan sel ARC D9
(EPOCHS=350), bukan "iso-compute" yang tak terdefinisi untuk model tanpa rekursi. Batch C (logging
prediksi per-instance) dijalankan pada sel ARC D9 seed 0-4, dan run A1 juga menyimpan prediksi
per-instance, sehingga pembelahan subset 512 bisa dilakukan pada kedua lengan kontras.

**Jumlah run tetap 18.** Ambang uji, kontras yang ditargetkan, dan komitmen pelaporan tidak berubah.

---

## AMANDEMEN 2 (17 September 2026, ditulis SETELAH batch B gagal, SEBELUM diulang)

**Kejadian.** Kelima run batch B (baseline non-rekursif ARC) mati setelah 12 detik dan tidak
menghasilkan data apa pun. Sebabnya murni konfigurasi, bukan hasil:
`config/arch/transformers_baseline.yaml` mengunci `num_heads: 12`, sehingga pada `hidden=256`
dimensi per-head menjadi 21 (ganjil). RoPE membelah dimensi itu lewat `rotate_half`, jadi nilai
ganjil mustahil. Baseline Sudoku yang sudah berjalan memakai `hidden=512`, head_dim 42, dan karena
itu lolos. Bukti kegagalan disimpan di `arc_baseline_FAILED_rope/` dan folder itu sengaja dinamai
agar `analyze_BM.py` tidak menemukannya.

**Perubahan.** Batch B diulang dengan `HEADS=8` sehingga head_dim menjadi 32. Nilai 8 dipilih karena
itulah jumlah head yang dipakai TRM, sehingga baseline ARC justru lebih sebanding dengan TRM
daripada bila memakai 12.

**Yang TIDAK berubah.** Jumlah run tetap 5 seed. Kontras yang ditargetkan tetap sama: ARC `D9` (TRM)
lawan baseline non-rekursif pada compute yang sama. Uji tetap Welch dua-sisi + permutasi eksak
dengan ambang Bonferroni 0,0167 per sumbu. Analisis tetap dijalankan **sekali** setelah seluruh run
selesai. Komitmen pelaporan apa adanya tetap berlaku, termasuk bila baseline mengalahkan TRM dangkal.

**Mengapa ini bukan penyimpangan protokol.** Run yang gagal tidak menghasilkan satu pun angka, jadi
tidak ada hasil yang dilihat lalu dijadikan dasar mengubah desain. Amandemen ini ditulis dan
di-commit sebelum run ulang diluncurkan, sesuai aturan CLAUDE.md #7.

**Baseline Sudoku tidak diulang.** Ia berjalan sah dengan 12 head dan hasilnya sudah dilaporkan di
naskah. Karena itu `HEADS` dibiarkan kosong secara default di `run_recipe.py`, supaya run lama tetap
dapat direproduksi persis. Perbedaan jumlah head antara baseline Sudoku (12) dan baseline ARC (8)
akan dinyatakan di naskah.

---

## AMANDEMEN 3 (17 September 2026): pengerasan higienis data pada skrip analisis, dan satu pengungkapan

**Apa yang diubah.** `analyze_BM.py` diberi penjagaan higienis data:
1. baris summary dengan `wall_s < 1000` dibuang (run gagal fase BM berdurasi 12 detik);
2. baris duplikat per tag dideduplikasi, diambil yang terakhir, karena `run_recipe.py` meng-APPEND
   ke `recipe_summary.csv` sehingga run ulang menumpuk baris;
3. jumlah run tiap batch diwajibkan (A1 5, A2 3, B 5) dan ketidaksesuaian dicetak sebagai peringatan
   di kepala laporan;
4. tag batch C yang jumlah checkpointnya bukan 25 ditolak, sehingga run yang masih berjalan atau
   yang grid stepnya tercampur tidak ikut;
5. lantai uji permutasi ikut dicetak, karena pada n=3 lawan 3 hanya ada C(6,3)=20 pembelahan sehingga
   p terkecil yang mungkin adalah 0,10 dan tanpa keterangan itu mudah dibaca sebagai bukti null;
6. kegagalan satu kontras tidak lagi membatalkan seluruh laporan;
7. berkas keluaran mengikuti `data_root`, tidak lagi menimpa laporan repo.

**Mengapa.** Audit fase BR membuktikan dengan percobaan langsung bahwa lima baris run gagal yang
menumpuk di summary batch B mengubah hasilnya dari `+1,87 poin, p=0,2707, gagal` menjadi
`+19,07 poin, p=0,0090, LOLOS`. Baris sampah dapat **membalik kesimpulan menjadi signifikan palsu**,
dan skrip tidak mengeluh sama sekali. Penjagaan ini melindungi pra-registrasi, bukan melonggarkannya.

**Yang TIDAK berubah.** Tidak ada uji, ambang, kontras, arah hipotesis, atau definisi metrik yang
diubah. Welch dua-sisi, permutasi eksak, Bonferroni 0,05/3 per sumbu, checkpoint terbaik, semuanya
tetap. Komitmen melaporkan hasil apa pun tetap berlaku.

**Pengungkapan.** Saat menguji penjagaan itu, skrip dijalankan atas data yang belum lengkap dan
**keluaran bagian A1 sebagian terlihat**. Skrip sudah dalam bentuk final saat itu dan tidak diubah
lagi sesudahnya; laporan sementara yang tertulis langsung dihapus. Hal ini dicatat di sini supaya
rekamannya lengkap: analisis pra-registrasi yang sah adalah yang dijalankan **sekali** setelah
seluruh 18 run selesai, dan itu belum terjadi.

---

## AMANDEMEN 4 (17 September 2026): definisi keluarga Bonferroni untuk baseline ARC, dikunci SEBELUM hasil

**Mengapa amandemen ini ada.** Audit fase BR menemukan bahwa keluarga Bonferroni di `stats_table.py`
ditulis tangan. Batch B akan menambahkan kontras baru (TRM lawan baseline non-rekursif di ARC), dan
**cara mengelompokkannya menentukan ambangnya**:

- bila dilipat ke keluarga "ARC depth" yang sudah ada, keluarga itu menjadi 6 kontras dan ambangnya
  0,05/6 = **0,0083**, sehingga kontras kedalaman ARC yang sekarang p=0,0122 akan **GUGUR**;
- bila menjadi keluarga sendiri, ambang keluarga kedalaman tetap 0,0167.

Memutuskan hal ini **setelah** melihat hasil adalah p-hacking lewat definisi keluarga. Karena itu
diputuskan sekarang, sebelum batch B dijalankan ulang dan sebelum analisis dijalankan.

**Keputusan: baseline ARC menjadi keluarga sendiri, "ARC baseline".**

Alasannya preseden, bukan kenyamanan. Struktur keluarga yang sudah berlaku di naskah:

| keluarga | kontras | ambang |
|---|---:|---:|
| Sudoku depth | 3 | 0,0167 |
| Sudoku width | 3 | 0,0167 |
| **Sudoku baseline** | **3** | **0,0167** |
| Maze depth | 3 | 0,0167 |
| ARC depth | 3 | 0,0167 |

Baseline non-rekursif Sudoku **sudah** menjadi keluarga terpisah dari kedalaman Sudoku sejak fase
sebelumnya. Memperlakukan baseline ARC dengan cara berbeda dari baseline Sudoku tidak punya dasar,
dan keduanya memang menjawab pertanyaan berbeda: "apakah kedalaman berbayar" lawan "apakah rekursi
berbayar sama sekali".

**Konsekuensi yang diterima di muka.** Total kontras naskah menjadi 18 (dari 15), dan ambang
gabungan seluruh-naskah untuk pembaca yang menuntut satu keluarga menjadi 0,05/18 = 0,0028 (dari
0,0033). Klaim mana pun yang lolos hari ini dan tidak lolos pada ambang baru akan dinyatakan apa
adanya.

**Yang TIDAK berubah.** Uji tetap Welch dua-sisi + permutasi eksak. Ambang per sumbu tetap 0,05/3.
Kontras yang ditargetkan tetap seperti butir 2 pra-registrasi. Analisis tetap dijalankan sekali
setelah seluruh run selesai.

---

## AMANDEMEN 5 (17 September 2026): subset evaluasi ARC diperbaiki, kontras penentu dijalankan ulang

Ditulis dan di-commit **SEBELUM** satu pun run baru diluncurkan.

### Cacat yang ditemukan

`make_small_eval.py` memotong test split dengan `[:N]`. Itu benar untuk Sudoku-Extreme dan Maze-Hard,
yang test split-nya tidak diaugmentasi, tetapi **salah untuk ARC-AGI-1**, yang test split-nya
diaugmentasi sekitar 1001 contoh per grup. Akibatnya `arc1-aug1k-e512` berisi 512 baris yang seluruhnya
jatuh di grup 0, yaitu **satu task ARC beserta augmentasinya**.

Bukti terukur, dihitung langsung dari berkas dataset:

| | `arc1-aug1k-e512` (lama) | `arc1-aug1k-g400` (baru) |
|---|---:|---:|
| contoh | 512 | 419 |
| label unik | **72** | **419** |
| input unik | 453 | 419 |
| posisi berlabel per contoh | **selalu 48** | 3 sampai 900, median 159 |
| konsistensi indeks | **rusak** | konsisten |

Seluruh **25 run ARC** proyek ini (`arc_depth_out` 15, `arc_d36_accum_out` 5, `arc_d9_preds_out` 5)
memakai subset lama. Sudoku (21 run) dan Maze (16 run) memakai subset yang sehat dan tidak terpengaruh.

### Yang dikunci sekarang

1. **Subset baru:** `data/arc1-aug1k-g400`, dibangun `make_group_eval.py`, mengambil puzzle PERTAMA
   setiap grup, yakni contoh asli tanpa augmentasi. 400 grup, 400 puzzle, 419 contoh, 419 label unik.
   Split train **tidak diubah**, sehingga `GROUPS=3080` dan seluruh konfigurasi pelatihan tetap sama;
   yang berubah hanya apa yang dievaluasi.
2. **Sepuluh run, dikunci:** ARC `D_eff=9` lima seed (batch 48, accum 1) dan ARC `D_eff=36` lima seed
   (micro-batch 24 x accum 2 = batch efektif 48). Alokasi D36 memakai akumulasi gradien, mengikuti
   Amandemen 1, supaya kontras menguji kedalaman saja dan bukan kedalaman bercampur batch.
   Tidak lebih, tidak kurang. Perkiraan 19 jam GPU.
3. **Kontras yang ditargetkan:** ARC `D9` lawan `D36`, akurasi token pada checkpoint terbaik.
   Uji: Welch dua-sisi + permutasi eksak, ambang Bonferroni per sumbu 0,05/3 = 0,0167.
4. **Baseline sepele dihitung ulang** pada subset baru sebelum efek apa pun dibaca (aturan repo #9).
   Angka 25,00% yang ada sekarang berasal dari subset lama dan tidak boleh dipakai.
5. **Batch B dibatalkan.** Baseline non-rekursif ARC tidak dijalankan ulang. Konsekuensinya kontrol
   non-rekursif hanya ada di Sudoku, dan itu dinyatakan sebagai keterbatasan di naskah.

### Komitmen pelaporan, apa pun hasilnya

- Bila kontras ARC **bertahan** pada subset yang sah, klaim dilaporkan dengan angka baru, dan angka
  lama dinyatakan superseded.
- Bila kontras ARC **hilang**, klaim "5,7 poin, p=0,012" **dicabut** dari abstrak, highlights, dan
  Results, dan ARC dilaporkan sebagai null. Naskah kemudian tinggal punya satu task dengan sumbu
  akurasi yang bisa dihargai, yaitu Sudoku-Extreme, dan itu diterima.
- **Apa pun hasilnya**, naskah wajib menyatakan bahwa subset evaluasi ARC yang dipakai pada seluruh
  run sebelumnya adalah augmentasi satu task, bahwa itu ditemukan sendiri dalam audit internal, dan
  bahwa angka ARC lama tidak menopang klaim tingkat-task.
- Analisis dijalankan **sekali**, setelah kesepuluh run selesai.

### Yang TIDAK berubah

Sudoku dan Maze tidak disentuh. Aturan checkpoint terbaik tetap. Definisi metrik tetap. Struktur
keluarga Bonferroni tetap seperti Amandemen 4. Artefak wajib per run tetap tiga plus baris summary.
