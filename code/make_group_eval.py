#!/usr/bin/env python3
"""Bangun subset evaluasi dengan SATU puzzle per grup, bukan N baris pertama.

Kenapa skrip ini ada (fase BS, 17 September 2026). `make_small_eval.py` memotong test split
dengan `[:N]`. Itu BENAR untuk dataset yang test split-nya tidak diaugmentasi (Sudoku-Extreme,
Maze-Hard: 512 baris pertama = 512 puzzle berbeda), tetapi SALAH untuk ARC-AGI-1, yang test
split-nya diaugmentasi ~1001 contoh per grup. Akibatnya `arc1-aug1k-e512` berisi 512 baris yang
seluruhnya jatuh di grup 0, yakni SATU task ARC beserta augmentasinya: 512 contoh tetapi hanya
72 grid keluaran unik, dan tiap contoh punya persis 48 posisi berlabel.

Skrip ini mengambil puzzle PERTAMA dari setiap grup, yaitu contoh asli tiap task tanpa augmentasi,
sehingga subsetnya berisi task yang benar-benar berbeda.

Identifier puzzle TIDAK diremap dan split train disalin apa adanya, sehingga tabel puzzle
embedding tetap berukuran sama dan indeksnya tetap sah.

Pakai: python3 make_group_eval.py <src_dataset_dir> <dst_dataset_dir> [maks_grup]
"""
import json, os, shutil, sys
import numpy as np

src = sys.argv[1]
dst = sys.argv[2]
maxg = int(sys.argv[3]) if len(sys.argv) > 3 else 0     # 0 = semua grup

S, D = f"{src}/test", f"{dst}/test"
os.makedirs(dst, exist_ok=True)
if not os.path.exists(f"{dst}/train"):
    shutil.copytree(f"{src}/train", f"{dst}/train")
shutil.copy(f"{src}/identifiers.json", f"{dst}/identifiers.json")
os.makedirs(D, exist_ok=True)

inp = np.load(f"{S}/all__inputs.npy")
lab = np.load(f"{S}/all__labels.npy")
pid = np.load(f"{S}/all__puzzle_identifiers.npy")
pidx = np.load(f"{S}/all__puzzle_indices.npy")
gidx = np.load(f"{S}/all__group_indices.npy")

ngroups = len(gidx) - 1
pick = list(range(ngroups)) if maxg <= 0 else list(range(min(maxg, ngroups)))

# puzzle pertama tiap grup yang dipilih; tiap puzzle bisa punya 1 atau 2 contoh
rows, new_pidx, new_pid, new_gidx = [], [0], [], [0]
for g in pick:
    p = int(gidx[g])                       # indeks puzzle pertama grup ini
    a, b = int(pidx[p]), int(pidx[p + 1])  # rentang baris contoh puzzle itu
    rows.extend(range(a, b))
    new_pidx.append(len(rows))
    new_pid.append(int(pid[p]))
    new_gidx.append(len(new_pid))          # satu puzzle per grup

rows = np.array(rows, dtype=np.int64)
np.save(f"{D}/all__inputs.npy", inp[rows])
np.save(f"{D}/all__labels.npy", lab[rows])
np.save(f"{D}/all__puzzle_identifiers.npy", np.array(new_pid, dtype=pid.dtype))
np.save(f"{D}/all__puzzle_indices.npy", np.array(new_pidx, dtype=pidx.dtype))
np.save(f"{D}/all__group_indices.npy", np.array(new_gidx, dtype=gidx.dtype))

m = json.load(open(f"{S}/dataset.json"))
m["total_groups"] = len(pick)
m["total_puzzles"] = len(new_pid)
m["mean_puzzle_examples"] = len(rows) / len(new_pid)
json.dump(m, open(f"{D}/dataset.json", "w"))

uniq = len(np.unique(lab[rows], axis=0))
print(f"{dst}: {len(pick)} grup, {len(new_pid)} puzzle, {len(rows)} contoh, "
      f"{uniq} label unik")
if uniq < len(pick) * 0.9:
    print("PERINGATAN: label unik jauh di bawah jumlah grup, periksa lagi struktur dataset")
    sys.exit(2)
