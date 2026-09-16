#!/usr/bin/env python3
"""Analisis fase BM, dijalankan SEKALI setelah seluruh 18 run selesai (prereg_BM.md butir 3).

Skrip ini ditulis dan di-commit SEBELUM hasil ada, sehingga analisisnya pra-spesifikasi.
Empat bagian, persis kontras yang dipra-registrasi:

  A1  ARC D36 alokasi baru (batch efektif 48) vs D9/D18 lama  -> apakah defisit D36 bertahan
  A2  Sudoku D36 alokasi baru (batch efektif 192) vs D9/D18   -> idem
  B   ARC D9 (TRM) vs baseline non-rekursif pada epoch sama
  C   ARC D9 vs D36 di bawah pembelahan separuh-seleksi / separuh-pelaporan

Uji: Welch dua-sisi + permutasi eksak; ambang Bonferroni per sumbu 0,05/3 = 0,0167.
Usage: python3 analyze_BM.py [data_root]
"""
import csv, glob, itertools, json, math, os, re, sys, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else HERE
sys.path.insert(0, HERE)
from stats_table import welch

ALPHA = 0.05 / 3


def rows(rel):
    p = os.path.join(ROOT, rel)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def cells(rs, col, key="D_eff"):
    out = {}
    for r in rs:
        try:
            out.setdefault(r[key], []).append(float(r[col]))
        except (KeyError, ValueError):
            pass
    return out


def perm_p(a, b):
    obs = st.mean(a) - st.mean(b)
    pool = a + b
    n = len(a)
    hit = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1
        hit += abs(st.mean(x) - st.mean(y)) >= abs(obs) - 1e-12
    return hit / tot, 2 / tot


def line(label, a, b):
    if len(a) < 2 or len(b) < 2:
        return f"| {label} | n terlalu kecil | | | | |"
    t, df, p, d = welch(a, b)
    pp, floor = perm_p(a, b)
    verd = "LOLOS" if p < ALPHA else "gagal"
    return (f"| {label} | {st.mean(a):.2f}±{st.stdev(a):.2f} | {st.mean(b):.2f}±{st.stdev(b):.2f} | "
            f"{st.mean(a)-st.mean(b):+.2f} | p={p:.4f} (perm {pp:.4f}) | {verd} |")


out = ["# Hasil fase BM (analisis pra-spesifikasi, dijalankan sekali)\n",
       f"Ambang Bonferroni per sumbu: {ALPHA:.4f}.\n"]

# ---------------------------------------------------------------- A1 ARC
arc_old = rows("arc_depth_out/recipe_summary.csv")
arc_new = rows("arc_d36_accum_out/recipe_summary.csv")
if arc_old and arc_new:
    o = cells(arc_old, "best_token_pct")
    n36 = [float(r["best_token_pct"]) for r in arc_new]
    out += ["\n## A1. ARC-AGI-1: sel D36 dengan batch efektif 48 (compute sama, epoch sama)\n",
            "| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |", "|---|---|---|---|---|---|",
            line("D9 vs D36 (alokasi LAMA, batch 24)", o["9"], o["36"]),
            line("D9 vs D36 (alokasi BARU, batch efektif 48)", o["9"], n36),
            line("D18 vs D36 (alokasi BARU)", o["18"], n36),
            line("D36 lama vs D36 baru (efek alokasi saja)", o["36"], n36)]
    e_old = st.mean([float(r["smi_net_Wh"]) for r in arc_old if r["D_eff"] == "36"])
    e_new = st.mean([float(r["smi_net_Wh"]) for r in arc_new])
    out.append(f"\nEnergi net D36: lama {e_old:.0f} Wh, baru {e_new:.0f} Wh "
               f"({100*(e_new-e_old)/e_old:+.0f}%). Langkah optimizer baru kira-kira separuh lama, "
               f"contoh yang dikonsumsi sama.")

# ---------------------------------------------------------------- A2 Sudoku
sud_old = rows("recipe_out/recipe_summary.csv")
sud_new = rows("sudoku_d36_accum_out/recipe_summary.csv")
if sud_old and sud_new:
    o = cells([r for r in sud_old if r["hidden"] == "512" and "recipe" in r["tag"]], "best_exact_pct")
    n36 = [float(r["best_exact_pct"]) for r in sud_new]
    out += ["\n## A2. Sudoku-Extreme: sel D36 dengan batch efektif 192\n",
            "| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |", "|---|---|---|---|---|---|",
            line("D9 vs D36 (alokasi LAMA, batch 96)", o["9"], o["36"]),
            line("D9 vs D36 (alokasi BARU, batch efektif 192)", o["9"], n36),
            line("D18 vs D36 (alokasi BARU)", o["18"], n36),
            line("D36 lama vs D36 baru (efek alokasi saja)", o["36"], n36)]

# ---------------------------------------------------------------- B baseline ARC
base = rows("arc_baseline_out/recipe_summary.csv")
if base and arc_old:
    o = cells(arc_old, "best_token_pct")
    b = [float(r["best_token_pct"]) for r in base]
    out += ["\n## B. ARC-AGI-1: TRM dangkal vs baseline non-rekursif (epoch sama)\n",
            "| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |", "|---|---|---|---|---|---|",
            line("D9 vs baseline non-rekursif", o["9"], b),
            line("D36 (lama) vs baseline non-rekursif", o["36"], b)]
    eb = st.mean([float(r["smi_net_Wh"]) for r in base])
    e9 = st.mean([float(r["smi_net_Wh"]) for r in arc_old if r["D_eff"] == "9"])
    out.append(f"\nEnergi net: D9 {e9:.0f} Wh, baseline {eb:.0f} Wh ({100*(eb-e9)/e9:+.0f}%).")

# ---------------------------------------------------------------- C held-out split
def best_from_preds(d, half):
    """Akurasi token per run: pilih checkpoint terbaik pada separuh SELEKSI, lapor separuh lain."""
    try:
        import numpy as np
    except ImportError:
        return {}
    res = {}
    for f in sorted(glob.glob(os.path.join(d, "preds_*_step*.npz"))):
        m = re.match(r"preds_(.+)_step(\d+)\.npz$", os.path.basename(f))
        if not m:
            continue
        tag, step = m.group(1), int(m.group(2))
        z = np.load(f)
        acc, seen = z["token_acc"], z["seen"]
        idx = np.where(seen)[0]
        sel, rep = idx[0::2], idx[1::2]          # separuh genap = seleksi, ganjil = pelaporan
        res.setdefault(tag, []).append((step, float(np.nanmean(acc[sel])) * 100,
                                        float(np.nanmean(acc[rep])) * 100))
    chosen = {}
    for tag, v in res.items():
        v.sort()
        k = max(range(len(v)), key=lambda i: v[i][1])   # pilih pada separuh seleksi
        chosen[tag] = v[k][2] if half == "report" else v[k][1]
    return chosen

c9 = best_from_preds(os.path.join(ROOT, "arc_d9_preds_out", "preds"), "report")
c36 = best_from_preds(os.path.join(ROOT, "arc_d36_accum_out", "preds"), "report")
if c9 and c36:
    a, b = list(c9.values()), list(c36.values())
    out += ["\n## C. ARC-AGI-1: checkpoint dipilih pada separuh subset, dilaporkan pada separuh lain\n",
            "| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |", "|---|---|---|---|---|---|",
            line("D9 vs D36 (held-out half)", a, b),
            f"\nPer run D9: {', '.join(f'{k}={v:.2f}' for k, v in sorted(c9.items()))}",
            f"\nPer run D36: {', '.join(f'{k}={v:.2f}' for k, v in sorted(c36.items()))}"]
else:
    out.append("\n## C. Belum ada berkas prediksi per-instance; batch C belum selesai.\n")

txt = "\n".join(out) + "\n"
os.makedirs(os.path.join(HERE, "ablation"), exist_ok=True)
open(os.path.join(HERE, "ablation", "bm_results.md"), "w").write(txt)
print(txt)
