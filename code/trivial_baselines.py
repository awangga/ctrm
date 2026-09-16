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
for name, label in [("sudoku-aug1k", "Sudoku-Extreme"), ("maze-aug", "Maze-Hard"), ("arc1-aug1k-e512", "ARC-AGI-1")]:
    d = os.path.join(TRM, "data", name, "test")
    if not os.path.exists(d):
        print(f"SKIP {name}: dataset belum dibangun"); continue
    meta = json.load(open(os.path.join(d, "dataset.json")))
    n = meta["total_groups"]
    L = np.load(os.path.join(d, "all__labels.npy"))[:n]
    I = np.load(os.path.join(d, "all__inputs.npy"))[:n]
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
txt.append("Pada ARC-AGI-1 kedua baseline token 25,00% sedangkan model mencapai 30,6-36,3%, jadi metriknya informatif.")
txt.append("Untuk akurasi EXACT (metrik yang dilaporkan pada Sudoku) baseline sepele adalah 0% di ketiga task,")
txt.append("sehingga 62,4% Sudoku tidak dapat dibandingkan dengan baseline token 30,92% di sumbu yang sama.")
os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w").write("\n".join(txt) + "\n")
print("tulis", OUT)
