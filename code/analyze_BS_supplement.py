#!/usr/bin/env python3
"""Analisis pelengkap fase BS, dijawab atas keberatan audit desk editor dan dua penelaah (18 Sep 2026).

Semua angka yang masuk naskah dari audit itu dihasilkan DI SINI supaya tertelusur (aturan repo #1),
bukan dari kode ad-hoc. Bukan analisis pra-registrasi baru: bagian J adalah analisis batch C yang SUDAH
dipra-registrasi di prereg_BM.md tetapi belum dilaporkan; sisanya pelaporan tambahan atas data yang sama.

  J  pembelahan separuh-seleksi / separuh-pelaporan (pra-registrasi batch C), subset sah g400
  K  pembacaan checkpoint AKHIR ARC
  L  energi ke target yang SAMA untuk kedua lengan (menjawab keberatan asimetri joules-to-target)
  M  uji batch A2 Sudoku (pra-registrasi butir 2)
  N  pemeriksaan tumpang tindih uji-latih ARC

Usage: python3 analyze_BS_supplement.py      (butuh TRM_DIR untuk bagian N)
"""
import csv, datetime, glob, hashlib, itertools, json, os, re, statistics as st, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from stats_table import welch
IDLE = 4.7
TRM = os.environ.get("TRM_DIR", "/home/adb/awangga/trm-env/TRM")
L = ["# Analisis pelengkap fase BS (auto-generated oleh analyze_BS_supplement.py)\n"]

def perm(a, b):
    pool = a + b; n = len(a); obs = abs(st.mean(a) - st.mean(b)); hit = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]; y = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1; hit += abs(st.mean(x) - st.mean(y)) >= obs - 1e-12
    return hit / tot, 2 / tot

def row(lab, a, b):
    t, df, p, d = welch(a, b); pp, fl = perm(a, b)
    return (f"| {lab} | {st.mean(a):.2f}+-{st.stdev(a):.2f} | {st.mean(b):.2f}+-{st.stdev(b):.2f} | "
            f"{st.mean(a)-st.mean(b):+.2f} | t={t:.2f} df={df:.1f} p={p:.2e} d={d:.2f} | "
            f"perm {pp:.4f} (lantai {fl:.4f}) | min1 {min(a):.2f} > maks2 {max(b):.2f}: {min(a) > max(b)} |")
H = "| kontras | grup 1 | grup 2 | selisih | Welch | permutasi | pemisahan total |\n|---|---|---|---|---|---|---|"

# ---- J
def heldout(d):
    res = {}
    for f in sorted(glob.glob(os.path.join(HERE, d, "preds", "preds_*_step*.npz"))):
        m = re.match(r"preds_(.+)_step(\d+)\.npz$", os.path.basename(f))
        z = np.load(f); idx = np.where(z["seen"])[0]; sel, rep = idx[0::2], idx[1::2]
        res.setdefault(m.group(1), []).append((int(m.group(2)), float(np.nanmean(z["token_acc"][sel])) * 100,
                                                float(np.nanmean(z["token_acc"][rep])) * 100))
    out = []
    for tag, v in sorted(res.items()):
        assert len(v) == 25, (tag, len(v)); k = max(range(25), key=lambda i: v[i][1]); out.append(v[k][2])
    return out, len(sel), len(rep)
h9, ns, nr = heldout("arc_d9_g400_out"); h36, _, _ = heldout("arc_d36_g400_out")
L += [f"\n## J. Pembelahan separuh-seleksi / separuh-pelaporan (pra-registrasi batch C)\n",
      f"Seleksi checkpoint pada separuh genap ({ns} contoh), pelaporan pada separuh ganjil ({nr} contoh).\n", H,
      row("ARC D9 lawan D36, held-out", h9, h36)]

# ---- K
def col(d, c): return [float(r[c]) for r in csv.DictReader(open(os.path.join(HERE, d, "recipe_summary.csv")))]
L += ["\n## K. ARC, checkpoint AKHIR\n", H,
      row("ARC D9 lawan D36, final", col("arc_d9_g400_out", "final_token_pct"), col("arc_d36_g400_out", "final_token_pct"))]

# ---- L
def cum(f):
    ts, pv = [], []
    for r in csv.reader(open(f)):
        if len(r) < 2: continue
        try: ts.append(datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()); pv.append(float(r[1]))
        except ValueError: pass
    ts = np.array(ts); p = np.array(pv) - IDLE
    return ts, np.concatenate([[0], np.cumsum(np.diff(ts) * (p[1:] + p[:-1]) / 2)]) / 3600
def curve(d, tag, key):
    ts, e = cum(os.path.join(HERE, d, f"pw_{tag}.csv")); T, A = [], []
    for l in open(os.path.join(HERE, d, f"progress_{tag}.jsonl")):
        j = json.loads(l)
        if j.get("phase") == "eval": T.append(j["t"]); A.append(j[key] * 100)
    return np.interp(np.array(T), ts, e), np.maximum.accumulate(np.array(A)), e[-1]
def first(E, A, x):
    i = np.where(A >= x)[0]; return float(E[i[0]]) if len(i) else None
L.append("\n## L. Energi ke target yang SAMA untuk kedua lengan\n")
for name, dd, ds, key, seeds in (
        ("Sudoku", lambda s: ("recipe_out", f"h512_d36_recipe_b96_s{s}"), lambda s: ("recipe_out", f"h512_d9_recipe_b192_s{s}"), "all/exact_accuracy", range(3)),
        ("ARC g400", lambda s: ("arc_d36_g400_out", f"h256_d36_recipe_b24_s{s}"), lambda s: ("arc_d9_g400_out", f"h256_d9_recipe_b48_s{s}"), "all/accuracy", range(5))):
    D = [curve(*dd(s), key) for s in seeds]; S = [curve(*ds(s), key) for s in seeds]
    tgt = st.mean([a.max() for _, a, _ in D])
    sh = [first(E, A, tgt) for E, A, _ in S]; dp = [first(E, A, tgt) for E, A, _ in D]
    tot = [t for _, _, t in D]; ok = [x for x in sh if x]; okd = [x for x in dp if x]
    L += [f"**{name}**, target {tgt:.2f}% (rerata akurasi terbaik lengan dalam)",
          f"- dangkal ke target: {[round(x) if x else None for x in sh]} -> rerata {st.mean(ok):.0f}+-{st.stdev(ok):.0f} Wh, median {st.median(ok):.0f} ({len(ok)}/{len(sh)} seed)",
          f"- dalam ke target: {[round(x) if x else None for x in dp]} -> " + (f"rerata {st.mean(okd):.0f}" + (f"+-{st.stdev(okd):.0f}" if len(okd) > 1 else "") + f" Wh ({len(okd)}/{len(dp)} seed)" if okd else "tidak pernah"),
          f"- dalam, energi total run: rerata {st.mean(tot):.0f} Wh",
          f"- rasio dangkal/total {st.mean(ok)/st.mean(tot):.0%}" + (f", dangkal/dalam-ke-target {st.mean(ok)/st.mean(okd):.0%}" if okd else "") + "\n"]

# ---- M
R = list(csv.DictReader(open(os.path.join(HERE, "recipe_out", "recipe_summary.csv"))))
g = lambda D: [float(r["best_exact_pct"]) for r in R if r["hidden"] == "512" and r["D_eff"] == D and "recipe" in r["tag"]]
new = col("sudoku_d36_accum_out", "best_exact_pct")
L += ["\n## M. Batch A2 Sudoku (pra-registrasi butir 2)\n", H,
      row("D9 lawan D36 batch disamakan", g("9"), new), row("D36 lama lawan D36 baru", g("36"), new)]

# ---- N
te, tr = os.path.join(TRM, "data", "arc1-aug1k-g400", "test"), os.path.join(TRM, "data", "arc1-aug1k-g400", "train")
if os.path.isdir(te) and os.path.isdir(tr):
    hh = lambda a: hashlib.md5(a.tobytes()).hexdigest()
    ti, tl = np.load(os.path.join(te, "all__inputs.npy")), np.load(os.path.join(te, "all__labels.npy"))
    pairs = {hh(np.concatenate([i, l])) for i, l in zip(ti, tl)}
    TI = np.load(os.path.join(tr, "all__inputs.npy"), mmap_mode="r"); TL = np.load(os.path.join(tr, "all__labels.npy"), mmap_mode="r")
    hit = 0
    for s in range(0, TI.shape[0], 200000):
        for i, l in zip(np.asarray(TI[s:s + 200000]), np.asarray(TL[s:s + 200000])):
            hit += hh(np.concatenate([i, l])) in pairs
    L += ["\n## N. Tumpang tindih uji-latih ARC\n",
          f"Pasangan (input, label) uji yang identik dengan suatu pasangan latih: **{hit} dari {len(ti)}**. "
          "Contoh uji task evaluasi ditahan oleh protokol upstream (`dataset/build_arc_dataset.py`); "
          "demonstrasinya dilatih. Kecocokan ini kebetulan dan mengenai kedua kedalaman setara."]
else:
    L.append("\n## N. dilewati: dataset TRM tidak ditemukan\n")

txt = "\n".join(L) + "\n"
open(os.path.join(HERE, "ablation", "bs_supplement.md"), "w").write(txt); print(txt)
