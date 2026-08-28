#!/usr/bin/env python3
"""Create a pilot dataset with a TRUNCATED test split (solves the eval-cost bottleneck).
The Sudoku-Extreme test split is ~4.2e5 puzzles; full evaluation per checkpoint is infeasible on one
GPU. We truncate the test split to N puzzles (1 example/puzzle, no augmentation) and reuse the train
split. Usage: python make_small_eval.py <src_dataset_dir> <dst_dataset_dir> <N>"""
import numpy as np, json, os, shutil, sys

src = sys.argv[1] if len(sys.argv) > 1 else "data/sudoku-cal"
dst = sys.argv[2] if len(sys.argv) > 2 else "data/sudoku-pilot"
N   = int(sys.argv[3]) if len(sys.argv) > 3 else 512

os.makedirs(dst, exist_ok=True)
if not os.path.exists(f"{dst}/train"):
    shutil.copytree(f"{src}/train", f"{dst}/train")
shutil.copy(f"{src}/identifiers.json", f"{dst}/identifiers.json")

os.makedirs(f"{dst}/test", exist_ok=True)
np.save(f"{dst}/test/all__inputs.npy",             np.load(f"{src}/test/all__inputs.npy")[:N])
np.save(f"{dst}/test/all__labels.npy",             np.load(f"{src}/test/all__labels.npy")[:N])
np.save(f"{dst}/test/all__puzzle_identifiers.npy", np.load(f"{src}/test/all__puzzle_identifiers.npy")[:N])
np.save(f"{dst}/test/all__puzzle_indices.npy",     np.load(f"{src}/test/all__puzzle_indices.npy")[:N + 1])
np.save(f"{dst}/test/all__group_indices.npy",      np.load(f"{src}/test/all__group_indices.npy")[:N + 1])
m = json.load(open(f"{src}/test/dataset.json"))
m["total_groups"] = N; m["total_puzzles"] = N
json.dump(m, open(f"{dst}/test/dataset.json", "w"))
print(f"pilot dataset ready at {dst} (test truncated to N={N})")
