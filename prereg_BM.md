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
optimisasi macet daripada efek kedalaman. Karena yang membatasi adalah memori, bukan compute,
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
