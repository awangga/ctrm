#!/usr/bin/env python3
"""Baseline sepele per task pada subset evaluasi 512 (fase BM, respons reviewer R2-M3).

Metrik direplikasi persis dari models/losses.py upstream: mask = (labels != ignore),
akurasi dirata-rata PER URUTAN lalu antar urutan. Dua baseline:
  - majority-class : prediksi satu kelas paling sering di posisi berlabel
  - copy-input     : prediksi = token input (untuk Maze = labirin tanpa lintasan)
Dataset tidak diredistribusi (lisensi); bangun dulu dengan code/rebuild_env.sh.
Usage: python3 trivial_baselines.py [TRM_DIR]
"""
import json, os, sys
import numpy as np
TRM = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TRM_DIR", "/home/adb/awangga/trm-env/TRM")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation", "trivial_baselines.md")
rows = []
TASKS = [("sudoku-aug1k", "Sudoku-Extreme"), ("maze-aug", "Maze-Hard"),
         # fase BS: subset ARC yang SAH, satu puzzle asli per grup (400 task berbeda)
         ("arc1-aug1k-g400", "ARC-AGI-1"),
         # subset lama DIPERTAHANKAN sebagai rekaman: 512 baris yang seluruhnya jatuh di grup 0,
         # yakni SATU task ARC beserta augmentasinya. Tidak boleh dipakai untuk klaim apa pun.
         ("arc1-aug1k-e512", "ARC-AGI-1 subset lama (1 task + augmentasi, TIDAK SAH)")]
for name, label in TASKS:
    d = os.path.join(TRM, "data", name, "test")
    if not os.path.exists(d):
        print(f"SKIP {name}: dataset belum dibangun"); continue
    meta = json.load(open(os.path.join(d, "dataset.json")))
    # Pakai SEMUA baris contoh, BUKAN meta["total_groups"]. Subset ARC g400 punya 400 grup tetapi
    # 419 contoh (sebagian puzzle punya dua contoh uji), dan model dinilai pada ke-419-nya
    # (seen=419 di npz prediksi). Memotong ke total_groups akan membuang 19 contoh dan membuat
    # baseline tidak sebanding. Untuk subset lain jumlah grup = jumlah baris, jadi angkanya sama.
    L = np.load(os.path.join(d, "all__labels.npy"))
    I = np.load(os.path.join(d, "all__inputs.npy"))
    assert L.shape == I.shape, f"{name}: bentuk labels {L.shape} != inputs {I.shape}"
    n = L.shape[0]
    ign = meta["ignore_label_id"]
    mask = (L != ign); cnt = mask.sum(-1).clip(1)
    per_seq = lambda pred: float(((mask & (pred == L)).sum(-1) / cnt).mean() * 100)
    per_seq_exact = lambda pred: float((((mask & (pred == L)).sum(-1) == mask.sum(-1))).mean() * 100)
    vals, c = np.unique(L[mask], return_counts=True); maj = int(vals[np.argmax(c)])
    rows.append((label, n, float(mask.sum(-1).mean()), per_seq(np.full_like(L, maj)), per_seq(I),
                 per_seq_exact(np.full_like(L, maj)), per_seq_exact(I)))
    print(f"{label}: labelled/ex {rows[-1][2]:.0f}, token majority {rows[-1][3]:.2f}% copy {rows[-1][4]:.2f}%, "
          f"exact majority {rows[-1][5]:.2f}% copy {rows[-1][6]:.2f}%")
txt = ["# Baseline sepele pada subset evaluasi 512 (auto-generated)\n",
       "Metrik identik dengan `models/losses.py` upstream (mask label, rata-rata per urutan).\n",
       "| task | n | posisi berlabel/contoh | majority token | copy-input token | majority exact | copy-input exact |",
       "|---|---:|---:|---:|---:|---:|---:|"]
for lab, n, k, mj, cp, mje, cpe in rows:
    txt.append(f"| {lab} | {n} | {k:.0f} | {mj:.2f}% | {cp:.2f}% | {mje:.2f}% | {cpe:.2f}% |")
txt.append("\nCatatan: pada Maze-Hard seluruh 900 posisi berlabel dan sel lintasan hanya 12,5% dari grid,")
txt.append("sehingga menyalin input sudah mencapai 87,51%. Akurasi token model (86,5-86,8%) berada DI BAWAH")
txt.append("angka itu, jadi metrik token Maze tidak mengukur penguasaan task pada anggaran ini.")
arc = [r for r in rows if r[0] == "ARC-AGI-1"]
if arc:
    txt.append(f"Pada ARC-AGI-1 (subset sah, {arc[0][1]} contoh dari 400 task) baseline token majority "
               f"{arc[0][3]:.2f}% dan copy-input {arc[0][4]:.2f}%. Angka ini dihitung ulang pada fase BS; "
               f"angka 25,00% lama berasal dari subset yang ternyata satu task beserta augmentasinya.")
txt.append("Untuk akurasi EXACT (metrik yang dilaporkan pada Sudoku) baseline sepele adalah 0% di ketiga task,")
txt.append("sehingga 62,4% Sudoku tidak dapat dibandingkan dengan baseline token 30,92% di sumbu yang sama.")
os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w").write("\n".join(txt) + "\n")
print("tulis", OUT)
