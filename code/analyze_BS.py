#!/usr/bin/env python3
"""Analisis pra-spesifikasi AMANDEMEN 5 (fase BS): kontras kedalaman ARC pada subset SAH.

Ditulis dan di-commit SEBELUM akurasi model pada subset g400 dibaca. Satu-satunya angka yang
sudah terlihat saat skrip ini ditulis adalah baseline sepele g400 (majority 41,82%, copy-input
60,75%), yang memang wajib dihitung lebih dulu menurut Amandemen 5 butir 4 dan aturan repo #9.

Kontras dikunci di Amandemen 5: ARC D9 lawan D36 (D36 dengan akumulasi, batch efektif 48),
akurasi token pada checkpoint terbaik, Welch dua-sisi + permutasi eksak, Bonferroni 0,05/3.

Sebelum kontras dibaca, skrip membandingkan akurasi tiap lengan dengan baseline copy-input.
Bila lengan mana pun berada di bawahnya, metrik token ARC tidak mengukur penguasaan task pada
anggaran ini, persis seperti Maze-Hard, dan kontrasnya dilaporkan sebagai tidak dapat dibaca.
"""
import csv, os, sys, itertools, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats_table import welch

ALPHA = 0.05 / 3
MIN_WALL_S = 1000.0
COPY_INPUT = 60.75      # trivial_baselines.md, subset arc1-aug1k-g400, dihitung fase BS
MAJORITY = 41.82


def load(rel, expect):
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, rel)))
            if float(r["wall_s"]) >= MIN_WALL_S]
    seen = {}
    for r in rows:
        seen[r["tag"]] = r
    out = [float(r["best_token_pct"]) for r in seen.values()]
    assert len(out) == expect, f"{rel}: {len(out)} run sah, diharapkan {expect}"
    return out, [float(r["smi_net_Wh"]) for r in seen.values()]


def perm_p(a, b):
    obs = abs(st.mean(a) - st.mean(b)); pool = a + b; n = len(a); hit = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]; y = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1; hit += abs(st.mean(x) - st.mean(y)) >= obs - 1e-12
    return hit / tot, 2 / tot


d9, e9 = load("arc_d9_g400_out/recipe_summary.csv", 5)
d36, e36 = load("arc_d36_g400_out/recipe_summary.csv", 5)

L = ["# Hasil AMANDEMEN 5: kontras kedalaman ARC pada subset sah (dijalankan sekali)\n",
     "Subset `arc1-aug1k-g400`: 400 task ARC berbeda, 419 contoh. D36 memakai akumulasi gradien",
     "(micro 24 x 2 = batch efektif 48, sama dengan D9).\n",
     "## 1. Pemeriksaan baseline sepele (wajib sebelum kontras dibaca)\n",
     f"Baseline token pada subset ini: majority {MAJORITY:.2f}%, copy-input {COPY_INPUT:.2f}%.\n",
     "| lengan | akurasi token terbaik (per seed) | rerata | di atas copy-input? |",
     "|---|---|---:|---|"]
below = False
for lab, v in (("D9", d9), ("D36", d36)):
    ab = st.mean(v) > COPY_INPUT
    below |= not ab
    L.append(f"| {lab} | {', '.join(f'{x:.2f}' for x in v)} | {st.mean(v):.2f}±{st.stdev(v):.2f} | "
             f"{'ya' if ab else '**TIDAK**'} |")

L.append("\n## 2. Kontras D9 lawan D36\n")
t, df, p, dd = welch(d9, d36)
pp, floor = perm_p(d9, d36)
L += ["| kontras | D9 | D36 | selisih | Welch | permutasi | Bonferroni 0,0167 |",
      "|---|---:|---:|---:|---|---|---|",
      f"| D9 lawan D36 | {st.mean(d9):.2f}±{st.stdev(d9):.2f} | {st.mean(d36):.2f}±{st.stdev(d36):.2f} | "
      f"{st.mean(d9)-st.mean(d36):+.2f} | t={t:.2f}, df={df:.1f}, p={p:.4f}, d={dd:.2f} | "
      f"p={pp:.4f} (lantai {floor:.4f}) | {'LOLOS' if p < ALPHA else 'gagal'} |"]
L.append(f"\nEnergi net: D9 {st.mean(e9):.1f}±{st.stdev(e9):.1f} Wh, D36 {st.mean(e36):.1f}±{st.stdev(e36):.1f} Wh.\n")

L.append("## 3. Vonis\n")
if below:
    L.append("**Akurasi token model berada DI BAWAH baseline copy-input pada subset sah.** Metrik token ARC")
    L.append("karena itu tidak mengukur penguasaan task pada anggaran ini, persis seperti Maze-Hard, dan")
    L.append("kontras di atas TIDAK dapat dibaca sebagai efek kedalaman. Sesuai komitmen Amandemen 5,")
    L.append("klaim ARC 5,7 poin p=0,012 dicabut dan ARC dilaporkan sebagai null measurement.")
elif p < ALPHA:
    L.append("Kedua lengan berada di atas baseline copy-input dan kontras lolos ambang Bonferroni sumbunya.")
else:
    L.append("Kedua lengan berada di atas baseline copy-input, tetapi kontras TIDAK lolos ambang Bonferroni.")
    L.append("Sesuai komitmen Amandemen 5, klaim ARC 5,7 poin p=0,012 dicabut dan ARC dilaporkan null.")

txt = "\n".join(L) + "\n"
open(os.path.join(HERE, "ablation", "bs_results.md"), "w").write(txt)
print(txt)
