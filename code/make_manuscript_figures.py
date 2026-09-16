#!/usr/bin/env python3
"""Bangun figur manuscript dari artefak run yang di-commit.

Gaya mengikuti figur yang sudah dipakai naskah: vektor PDF, DejaVu Serif, lebar 6.6 in,
palet colorblind-safe Okabe-Ito, tanpa judul-plot, spine bersih.

Hanya memakai rezim faithful-recipe (recipe_out/ dan microbench); data pilot TIDAK dipakai
untuk figur hasil.

Pakai:  python3 make_manuscript_figures.py [--out ../../manuscript/figures]
"""
import argparse
import csv
import glob
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IDLE_W = 4.7           # daya idle terukur, dikurangkan spt Eq. (2) di naskah
GRID = 0.67593         # kg CO2e/kWh, dari CodeCarbon (Indonesia), konstan di semua run

# Gbr. 6 = dua panel berdampingan; ukuran fontnya dikunci lewat dua konstanta ini supaya
# panel kiri tidak terbaca lebih kecil daripada kanan. Skala cetak diukur dari lebar PDF
# keluaran (bukan figsize, karena bbox="tight" memangkas): textwidth elsarticle 390 pt,
# panel kiri 0.50\linewidth / 225,7 pt = 0,864x, panel kanan 0.43\linewidth / 194,1 pt = 0,864x.
PANEL6_TICK = 9.5      # efektif 8,21 pt
PANEL6_LABEL = 10.0    # efektif 8,64 pt

# Okabe-Ito
OI = {"blue": "#0072B2", "orange": "#E69F00", "vermillion": "#D55E00",
      "green": "#009E73", "sky": "#56B4E9", "grey": "#555555",
      "reddish": "#CC79A7"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def read_power(path):
    """Kembalikan (waktu unix, daya W) dari log nvidia-smi 1 Hz."""
    ts, w = [], []
    for line in open(path):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            t = datetime.strptime(parts[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            p = float(parts[1])
        except ValueError:
            continue
        ts.append(t)
        w.append(p)
    return np.array(ts), np.array(w)


def cumulative_wh(ts, w):
    """Energi net kumulatif (Wh) via trapesium atas (P - P_idle), Eq. (2).

    TIDAK di-clip pada nol, supaya identik dengan run_recipe.py yang menghasilkan
    angka energi yang dilaporkan. Sampel di bawah lantai idle mengurangi total.
    Selisih terhadap versi ter-clip di seluruh run yang dilaporkan < 0,001%.
    """
    net = w - IDLE_W
    if len(ts) < 2:
        return np.zeros_like(net)
    dt = np.diff(ts)
    seg = 0.5 * (net[1:] + net[:-1]) * dt          # Joule
    return np.concatenate([[0.0], np.cumsum(seg)]) / 3600.0


def eval_points(path, metric="all/exact_accuracy"):
    """(step, acc%) dari tiap checkpoint eval, plus waktu unix."""
    out = []
    for line in open(path):
        if '"phase": "eval"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append((float(d["t"]), int(d["step"]), 100.0 * float(d.get(metric, 0.0))))
    return out


def trajectory(tag, out_dir, metric="all/exact_accuracy"):
    """Lintasan (Wh kumulatif, exact%) untuk satu run."""
    pw = os.path.join(out_dir, f"pw_{tag}.csv")
    pg = os.path.join(out_dir, f"progress_{tag}.jsonl")
    if not (os.path.exists(pw) and os.path.exists(pg)):
        return None
    ts, w = read_power(pw)
    if len(ts) < 2:
        return None
    cum = cumulative_wh(ts, w)
    xs, ys = [], []
    for t, _step, acc in eval_points(pg, metric):
        i = int(np.searchsorted(ts, t))
        if i >= len(cum):
            i = len(cum) - 1
        xs.append(cum[i])
        ys.append(acc)
    return np.array(xs), np.array(ys)


def fig_energy_accuracy(out_path, data_root):
    """Lintasan energi-akurasi Sudoku h512, tiga kedalaman x tiga seed."""
    rec = os.path.join(data_root, "recipe_out")
    # Linestyle membedakan D18 dan D36 juga dalam cetak hitam-putih: #E69F00 dan
    # #D55E00 nyaris sama begitu dikonversi ke grayscale.
    specs = [(9, "b192", OI["blue"], "o", "-"), (18, "b192", OI["orange"], "s", (0, (4.5, 1.8))),
             (36, "b96", OI["vermillion"], "^", (0, (1.4, 1.4)))]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    TARGET = 50.0
    for depth, batch, colour, marker, ls in specs:
        curves = []
        for seed in (0, 1, 2):
            tr = trajectory(f"h512_d{depth}_recipe_{batch}_s{seed}", rec)
            if tr is None:
                continue
            curves.append(tr)
            ax.plot(tr[0], tr[1], color=colour, alpha=0.28, lw=0.8, zorder=1)
        if not curves:
            continue
        n = min(len(c[0]) for c in curves)
        xm = np.mean([c[0][:n] for c in curves], axis=0)
        ym = np.mean([c[1][:n] for c in curves], axis=0)
        ax.plot(xm, ym, color=colour, marker=marker, markersize=2.6, markevery=3, ls=ls,
                label=rf"$D_{{\mathrm{{eff}}}}={depth}$", zorder=3)
        # Titik capai-target dihitung PER SEED lalu dirata-rata, konsisten dengan tabel
        # di naskah; kurva rata-rata tidak dipakai untuk ini karena seed yang tak pernah
        # mencapai target akan menyeret rata-ratanya.
        reach = []
        for cx, cy in curves:
            h = np.where(cy >= TARGET)[0]
            if len(h):
                reach.append(cx[h[0]])
        if reach:
            mx = float(np.mean(reach))
            ax.plot(mx, TARGET, marker="*", markersize=11, color=colour,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4)
            # Arah offset diseragamkan (kanan-bawah bintang) supaya mata tidak perlu
            # menebak label mana milik bintang mana; jarak vertikalnya saja yang beda
            # agar teks tidak ditembus kurva tetangga.
            ax.annotate(f"{mx:.0f} Wh ({len(reach)}/{len(curves)})", (mx, TARGET),
                        textcoords="offset points", xytext=(9, {9: -21}.get(depth, -13)),
                        fontsize=8, color=colour,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92))
    ax.axhline(TARGET, color=OI["grey"], ls=":", lw=0.9, zorder=0)
    # Teks target dipindah ke ruang kosong di atas garis putus-putus (sebelum kurva mana
    # pun memotongnya) supaya tidak berdesakan dengan bintang capai-target.
    ax.annotate("target 50% exact", (0.30 * ax.get_xlim()[1], TARGET), textcoords="offset points",
                xytext=(0, 5), ha="left", fontsize=8, color=OI["grey"])
    ax.set_xlabel("Cumulative net GPU energy (Wh)")
    ax.set_ylabel("Exact accuracy (\\%)" if False else "Exact accuracy (%)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    sec = ax.secondary_xaxis("top", functions=(lambda wh: wh / 1000 * GRID * 1000,
                                               lambda g: g / (GRID * 1000) * 1000))
    sec.set_xlabel(f"Cumulative CO$_2$e (g), at {GRID * 1000:.1f} g/kWh", fontsize=8.5)
    sec.tick_params(labelsize=8)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_cost_model(out_path, data_root):
    """Cost-model energi per langkah dari microbench hardware, log-log + CI bootstrap."""
    csv_path = os.path.join(data_root, "microbench_results.csv")
    P, J, Dv = [], [], []
    for r in csv.DictReader(open(csv_path)):
        try:
            P.append(float(r["params_M"]) * float(r["D_eff"]))
            J.append(float(r["J_per_step"]))
            Dv.append(int(float(r["D_eff"])))
        except (KeyError, ValueError):
            continue
    P, J, Dv = np.array(P), np.array(J), np.array(Dv)
    lx, ly = np.log(P), np.log(J)
    b, la = np.polyfit(lx, ly, 1)
    pred = la + b * lx
    r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    # Bootstrap direplikasi PERSIS spt fit_extrapolation.py (RNG dan konvensi persentil
    # yang sama) supaya CI di figur identik dengan yang dilaporkan di naskah.
    import random as _random
    _random.seed(20260627)
    bs = []
    n = len(P)
    for _ in range(5000):
        idx = [_random.randrange(n) for _ in range(n)]
        bxx = lx[idx]
        if len(set(bxx.tolist())) < 2:
            continue
        bs.append(np.polyfit(bxx, ly[idx], 1)[0])
    bs.sort()
    lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))]

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    xs = np.linspace(lx.min(), lx.max(), 100)
    ax.fill_between(np.exp(xs), np.exp(la + lo * xs), np.exp(la + hi * xs),
                    color="0.55", alpha=0.16, lw=0)
    ax.plot(np.exp(xs), np.exp(la + b * xs), color="0.2", lw=1.2, zorder=2,
            label=rf"fit, $b={b:.2f}$")
    # Acuan kemiringan 1 diikat ke titik terkecil: tanpa garis ini sub-linearitas
    # (b < 1) tidak terlihat pada sumbu log-log, karena mata tak punya pembanding.
    i0 = int(np.argmin(P))
    ax.plot(np.exp(xs), J[i0] * np.exp(xs) / P[i0], ls=(0, (4, 2)), color="0.55", lw=1.0,
            zorder=2, label="slope 1 (linear)")
    # Warna+penanda per D_eff: 14 titik itu grid 5 lebar x 3 kedalaman, dan tanpa
    # pembedaan ini beberapa titik terbaca saling tindih. Palet sama dgn Gbr. 4-5.
    dcol = {9: OI["blue"], 18: OI["orange"], 36: OI["vermillion"]}
    dmark = {9: "o", 18: "s", 36: "^"}
    for d in (9, 18, 36):
        sel = Dv == d
        if not sel.any():
            continue
        ax.scatter(P[sel], J[sel], s=20, color=dcol[d], marker=dmark[d], zorder=4,
                   edgecolor="white", linewidth=0.4,
                   label=rf"$D_{{\mathrm{{eff}}}}{{=}}{d}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$P \cdot D_{\mathrm{eff}}$ (M)")
    ax.set_ylabel("Energy per step (J)")
    ax.tick_params(labelsize=9)
    ax.text(0.04, 0.96, f"$b={b:.2f}$ [{lo:.2f}, {hi:.2f}]\n$R^2={r2:.2f}$, $n={len(P)}$",
            transform=ax.transAxes, va="top", fontsize=8.5)
    # Legenda 8.5 pt: figur ini dicetak pada 0,82x (0.5\linewidth elsarticle preprint),
    # jadi apa pun di bawah 8.5 pt jatuh di bawah ambang 7 pt Elsevier.
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", handlelength=1.4,
              handletextpad=0.45, labelspacing=0.22, borderpad=0.1)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}  (b={b:.3f} CI[{lo:.2f},{hi:.2f}] R2={r2:.3f})")


def joules_to_target_table(data_root, target=50.0):
    """Tabel Joules-to-target per konfigurasi, dihitung per seed lalu dirata-rata."""
    rec = os.path.join(data_root, "recipe_out")
    cfgs = [("TRM D_eff=9", "h512_d9_recipe_b192"), ("TRM D_eff=18", "h512_d18_recipe_b192"),
            ("TRM D_eff=36", "h512_d36_recipe_b96"),
            ("non-recursive baseline", "h512_transformers_baseline_b192")]
    rows = []
    for name, base in cfgs:
        per = []
        for seed in (0, 1, 2):
            tr = trajectory(f"{base}_s{seed}", rec)
            if tr is None:
                continue
            x, y = tr
            h = np.where(y >= target)[0]
            per.append(float(x[h[0]]) if len(h) else None)
        ok = [v for v in per if v is not None]
        rows.append((name, per, len(ok), len(per),
                     float(np.mean(ok)) if ok else None,
                     float(np.std(ok, ddof=1)) if len(ok) > 1 else None))
    print(f"\n  Joules-to-target ({target:.0f}% exact), per seed lalu dirata-rata:")
    for name, per, nok, ntot, m, sd in rows:
        cells = ", ".join(f"{v:.0f}" if v is not None else "--" for v in per)
        tail = (f"{m:.0f}" + (f" +/- {sd:.0f}" if sd else "") +
                f" Wh, {m/1000*GRID*1000:.0f} g CO2e") if m else "never reached"
        print(f"    {name:24s} [{cells}]  {nok}/{ntot}  -> {tail}")
    return rows


def perseed_table(data_root):
    """Cetak nilai akurasi per seed untuk tiap konfigurasi rezim faithful.

    Dipakai untuk Tabel per-seed di naskah; dicetak di sini supaya angkanya
    diregenerasi dari summary CSV, bukan disalin tangan.
    """
    import collections
    import re as _re
    import statistics as _st
    src = [("Sudoku-Extreme (exact)", "recipe_out/recipe_summary.csv", "best_exact_pct"),
           ("Maze-Hard (token)", "maze_depth_out/recipe_summary.csv", "best_token_pct"),
           ("ARC-AGI-1 (token)", "arc_depth_out/recipe_summary.csv", "best_token_pct")]
    print("\n  Akurasi per seed (rezim faithful):")
    for task, rel, col in src:
        path = os.path.join(data_root, rel)
        if not os.path.exists(path):
            continue
        groups = collections.defaultdict(dict)
        for r in csv.DictReader(open(path)):
            m = _re.search(r"_s(\d+)$", r["tag"])
            if not m:
                continue
            groups[r["tag"][:m.start()]][int(m.group(1))] = float(r[col])
        print(f"    {task}")
        for base in sorted(groups):
            v = groups[base]
            got = [v[s] for s in sorted(v)]
            cells = " ".join(f"{v[s]:6.2f}" if s in v else "    --" for s in (0, 1, 2))
            sd = _st.stdev(got) if len(got) > 1 else 0.0
            print(f"      {base:40s} {cells}  -> {_st.mean(got):5.1f} +/- {sd:.1f}")


def pilot_vs_faithful(data_root):
    """Bandingkan peringkat kedalaman rezim pilot vs faithful pada Sudoku.

    Menjawab: bisakah run murah dipakai menyaring pilihan kedalaman lebih dulu?
    Angka dicetak dari CSV yang di-commit supaya klaim di naskah telusur.
    """
    import statistics as _st
    rep = os.path.join(data_root, "budget_out", "replication_summary.csv")
    xv = os.path.join(data_root, "budget_out", "energy_xval.csv")
    rec = os.path.join(data_root, "recipe_out", "recipe_summary.csv")
    if not (os.path.exists(rep) and os.path.exists(xv) and os.path.exists(rec)):
        return
    pilot = {}
    for r in csv.DictReader(open(rep)):
        pilot[int(r["D_eff"])] = [float(r[f"seed{i}_exact"]) for i in (0, 1, 2)]
    for r in csv.DictReader(open(xv)):          # seed 3-4 disimpan sebagai fraksi
        pilot.setdefault(int(r["D_eff"]), []).append(float(r["exact_acc"]) * 100)
    faith = {}
    for r in csv.DictReader(open(rec)):
        m = None
        for d, tag in ((9, "h512_d9_recipe_b192"), (18, "h512_d18_recipe_b192"),
                       (36, "h512_d36_recipe_b96")):
            if r["tag"].startswith(tag):
                m = d
        if m:
            faith.setdefault(m, []).append(float(r["best_exact_pct"]))
    print("\n  Pilot vs faithful, sumbu kedalaman Sudoku (exact %):")
    for label, data in (("pilot h128 (5 seed)", pilot), ("faithful h512 (3 seed)", faith)):
        means = {d: _st.mean(v) for d, v in sorted(data.items())}
        sds = {d: _st.stdev(v) for d, v in data.items() if len(v) > 1}
        rank = " > ".join(f"D{d}" for d in sorted(means, key=lambda k: -means[k]))
        spread = max(means.values()) - min(means.values())
        print(f"    {label:24s} " +
              "  ".join(f"D{d}={means[d]:5.2f}+/-{sds.get(d, 0):.2f}" for d in sorted(means)))
        print(f"    {'':24s} peringkat {rank};  sebaran {spread:.1f} poin, "
              f"sd {min(sds.values()):.1f}-{max(sds.values()):.1f}")


def agreement_floor(data_root):
    """Lantai kesepakatan CodeCarbon vs nvidia-smi di rezim pilot, memisahkan run batal."""
    import glob as _glob
    ok, short = [], []
    for f in _glob.glob(os.path.join(data_root, "*", "*summary*.csv")) + \
            [os.path.join(data_root, "budget_out", "energy_xval.csv")]:
        if not os.path.exists(f):
            continue
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        ak = [k for k in rows[0] if "agree" in k.lower()]
        wk = [k for k in rows[0] if k in ("wall_s", "smi_dur_s")]
        if not ak:
            continue
        for r in rows:
            v = r.get(ak[0])
            if not v:
                continue
            dur = float(r.get(wk[0], 1e9)) if wk else 1e9
            (short if dur < 120 else ok).append((float(v), r.get("tag", ""), dur))
    if ok:
        print(f"\n  Kesepakatan dua instrumen, run pilot yang selesai: "
              f"min {min(a for a, _, _ in ok):.2f}%  (n={len(ok)})")
    if short:
        print(f"    run batal <120 s (dikecualikan): " +
              ", ".join(f"{t} {a:.2f}% @{d:.0f}s" for a, t, d in sorted(short)))


def iso_accuracy_energy(data_root):
    """Energi yang dibutuhkan D9 untuk MENYAMAI akurasi checkpoint-terbaik tiap pesaing.

    Perbandingan iso-akurasi: metrik Joules-to-target yang sama, tetapi target
    diambil dari akurasi akhir yang benar-benar dicapai pesaing, bukan satu ambang
    tetap. Ini yang menjawab "berapa energi untuk mutu yang sama".
    """
    import statistics as _st
    rec = os.path.join(data_root, "recipe_out")
    rivals = [("D_eff=36", 36.26, 340.4), ("D_eff=18", 50.07, 358.0),
              ("non-recursive baseline", 49.67, 384.0)]
    print("\n  Energi iso-akurasi: biaya D9 untuk menyamai akurasi terbaik pesaing")
    for name, target, rival_wh in rivals:
        per = []
        for seed in (0, 1, 2):
            tr = trajectory(f"h512_d9_recipe_b192_s{seed}", rec)
            if tr is None:
                continue
            x, y = tr
            h = np.where(y >= target)[0]
            per.append(float(x[h[0]]) if len(h) else None)
        ok = [v for v in per if v is not None]
        if not ok:
            continue
        m = _st.mean(ok)
        sd = _st.stdev(ok) if len(ok) > 1 else 0.0
        cells = ", ".join(f"{v:.0f}" if v is not None else "--" for v in per)
        print(f"    vs {name:24s} target {target:5.2f}%  D9=[{cells}] -> {m:5.1f}+/-{sd:.0f} Wh "
              f"({len(ok)}/{len(per)});  pesaing {rival_wh:.0f} Wh, hemat {100*(rival_wh-m)/rival_wh:.0f}%"
              f";  CO2e {rival_wh/1000*GRID*1000:.0f} -> {m/1000*GRID*1000:.0f} g")


def iso_accuracy_energy_arc(data_root):
    """ARC-AGI-1 (token, checkpoint terbaik): energi D9 untuk menyamai akurasi terbaik D36."""
    import statistics as _st
    arc = os.path.join(data_root, "arc_depth_out")
    sp = os.path.join(arc, "recipe_summary.csv")
    if not os.path.exists(sp):
        return
    rows = list(csv.DictReader(open(sp)))
    d36 = [float(r["best_token_pct"]) for r in rows if r["D_eff"] == "36"]
    wh36 = [float(r["smi_net_Wh"]) for r in rows if r["D_eff"] == "36"]
    target, rival_wh = _st.mean(d36), _st.mean(wh36)
    per = []
    for seed in range(5):
        tr = trajectory(f"h256_d9_recipe_b48_s{seed}", arc, metric="all/accuracy")
        if tr is None:
            continue
        x, y = tr
        h = np.where(y >= target)[0]
        per.append(float(x[h[0]]) if len(h) else None)
    ok = [v for v in per if v is not None]
    cells = ", ".join(f"{v:.0f}" if v is not None else "--" for v in per)
    if ok:
        m = _st.mean(ok); sd = _st.stdev(ok) if len(ok) > 1 else 0.0
        print(f"\n  ARC iso-akurasi (token, checkpoint terbaik): D9 menyamai D36 {target:.2f}%: "
              f"D9=[{cells}] -> {m:.0f}+/-{sd:.0f} Wh ({len(ok)}/{len(per)}); D36 spent {rival_wh:.0f} Wh, "
              f"hemat {100*(rival_wh-m)/rival_wh:.0f}%")


def fig_crosstask(out_path, data_root):
    """Efek kedalaman di tiga task, memakai SELURUH seed yang tersedia per task."""
    import collections
    import statistics as _st
    # Fase BM: tiap panel menampilkan garis baseline sepele metriknya (trivial_baselines.py),
    # karena tanpa garis itu panel Maze menyesatkan: seluruh rentangnya ada DI BAWAH 87,51%.
    src = [("Sudoku-Extreme", "recipe_out/recipe_summary.csv", "best_exact_pct",
            "Exact accuracy (%)", ("h512_d9_recipe", "h512_d18_recipe", "h512_d36_recipe"), 0.0),
           ("ARC-AGI-1", "arc_depth_out/recipe_summary.csv", "best_token_pct",
            "Token accuracy (%)", ("h256_d9_recipe", "h256_d18_recipe", "h256_d36_recipe"), 25.0),
           ("Maze-Hard", "maze_depth_out/recipe_summary.csv", "best_token_pct",
            "Token accuracy (%)", ("h256_d9_recipe", "h256_d18_recipe", "h256_d36_recipe"), 87.51)]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
    for ax, (task, rel, col, ylab, prefixes, base) in zip(axes, src):
        path = os.path.join(data_root, rel)
        if not os.path.exists(path):
            continue
        vals = collections.defaultdict(list)
        for r in csv.DictReader(open(path)):
            for d, pre in zip((9, 18, 36), prefixes):
                if r["tag"].startswith(pre):
                    vals[d].append(float(r[col]))
        xs = sorted(vals)
        m = [_st.mean(vals[d]) for d in xs]
        e = [_st.stdev(vals[d]) if len(vals[d]) > 1 else 0.0 for d in xs]
        n = min(len(vals[d]) for d in xs)
        for i, d in enumerate(xs):                       # titik tiap seed
            ax.scatter([i] * len(vals[d]), vals[d], s=9, color=OI["grey"], alpha=0.45, zorder=2)
        ax.errorbar(range(len(xs)), m, yerr=e, color=OI["blue"], marker="o", markersize=4,
                    capsize=3, lw=1.3, zorder=3)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{d}" for d in xs])
        ax.set_xlabel(r"$D_{\mathrm{eff}}$")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(f"{task}  ($n{{=}}{n}$)", fontsize=8.5)
        ax.margins(x=0.22, y=0.18)
        lo, hi = ax.get_ylim()
        allv = [v for d in xs for v in vals[d]]
        if base > min(allv) - 50:                       # baseline dekat/di dalam rentang data
            ax.set_ylim(min(lo, base - 0.45 * (hi - lo)), max(hi, base + 0.18 * (hi - lo)))
            ax.axhline(base, color=OI["vermillion"], ls="--", lw=1.0, zorder=1)
            ax.text(0.98, base, f"trivial baseline {base:g}%", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=6.8, color=OI["vermillion"])
    fig.tight_layout(w_pad=1.4)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_learning(out_path, data_root):
    """Kurva belajar Sudoku h512: akurasi vs langkah, semua seed, tiga kedalaman.

    Menjawab pertanyaan reviewer yang belum dijawab figur mana pun: apakah run
    benar-benar dilatih cukup lama? Sekaligus memperlihatkan rancangan iso-compute
    bekerja (D9 menjalankan dua kali lipat langkah D18/D36).
    """
    rec = os.path.join(data_root, "recipe_out")
    specs = [(9, "b192", OI["blue"], "o", "-"), (18, "b192", OI["orange"], "s", (0, (4.5, 1.8))),
             (36, "b96", OI["vermillion"], "^", (0, (1.4, 1.4)))]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    for depth, batch, colour, marker, ls in specs:
        curves = []
        for seed in (0, 1, 2):
            pg = os.path.join(rec, f"progress_h512_d{depth}_recipe_{batch}_s{seed}.jsonl")
            if not os.path.exists(pg):
                continue
            xs, ys = [], []
            for _t, step, acc in eval_points(pg):
                xs.append(step)
                ys.append(acc)
            if xs:
                curves.append((np.array(xs), np.array(ys)))
                ax.plot(xs, ys, color=colour, alpha=0.25, lw=0.8, zorder=1)
        if not curves:
            continue
        n = min(len(c[0]) for c in curves)
        xm = curves[0][0][:n]
        ym = np.mean([c[1][:n] for c in curves], axis=0)
        ax.plot(xm, ym, color=colour, marker=marker, markersize=2.6, markevery=3, ls=ls,
                label=rf"$D_{{\mathrm{{eff}}}}={depth}$ ({xm[-1]/1000:.0f}k steps)", zorder=3)
        if depth == 18:
            # Checkpoint terbaik per seed ditandai: angka yang dibahas caption berasal
            # dari puncak tiap seed, bukan dari puncak kurva rata-rata, dan tanpa
            # penanda ini pembaca tak bisa menemukannya di figur.
            bx = [float(cx[int(np.argmax(cy))]) for cx, cy in curves]
            by = [float(np.max(cy)) for cx, cy in curves]
            ax.plot(bx, by, "*", ms=8.5, mfc=colour, mec="white", mew=0.6, ls="", zorder=5)
            ax.annotate(rf"best checkpoints, mean {np.mean(by):.1f}\%".replace("\\%", "%"),
                        (float(np.mean(bx)), float(np.max(by))), textcoords="offset points",
                        xytext=(6, 7), fontsize=8, color=colour,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9))
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_iso_accuracy(out_path, data_root):
    """Energi untuk menyamai akurasi akhir tiap pesaing: batang berpasangan."""
    import statistics as _st
    rec = os.path.join(data_root, "recipe_out")
    rivals = [(r"$D_{\mathrm{eff}}{=}36$", 36.26, 340.4), (r"$D_{\mathrm{eff}}{=}18$", 50.07, 358.0),
              ("non-recursive", 49.67, 384.0)]
    labels, rival_wh, d9_wh, d9_sd, targets = [], [], [], [], []
    for name, target, rwh in rivals:
        per = []
        for seed in (0, 1, 2):
            tr = trajectory(f"h512_d9_recipe_b192_s{seed}", rec)
            if tr is None:
                continue
            x, y = tr
            h = np.where(y >= target)[0]
            if len(h):
                per.append(float(x[h[0]]))
        if not per:
            continue
        labels.append(name)
        targets.append(target)
        rival_wh.append(rwh)
        d9_wh.append(_st.mean(per))
        d9_sd.append(_st.stdev(per) if len(per) > 1 else 0.0)
    idx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.bar(idx - 0.19, rival_wh, 0.36, color=OI["grey"], label="rival, full budget")
    ax.bar(idx + 0.19, d9_wh, 0.36, yerr=d9_sd, capsize=3, color=OI["blue"],
           label=r"$D_{\mathrm{eff}}{=}9$ to match")
    # Label persen ditaruh DI ATAS ujung error bar: dengan offset tetap 14 Wh ia
    # ditembus error bar yang s.d.-nya mencapai 19 Wh.
    for i, (r, d, s) in enumerate(zip(rival_wh, d9_wh, d9_sd)):
        ax.text(i + 0.19, d + s + 13, f"\u2212{100*(r-d)/r:.0f}%", ha="center", fontsize=8.5,
                color=OI["blue"], fontweight="bold")
    ax.set_xticks(idx)
    # Akurasi target ikut dicetak: angka inilah yang mendefinisikan arti "matched".
    ax.set_xticklabels([f"{n}\nmatch {t:.1f}%" for n, t in zip(labels, targets)], fontsize=8.6)
    ax.set_ylabel("Net energy (Wh)")
    ax.tick_params(axis="y", labelsize=8.6)
    ax.set_ylim(0, max(rival_wh) * 1.30)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def eval_share_table(data_root):
    """Pangsa wall-time yang dipakai evaluasi berkala, per kedalaman.

    Menopang klaim protokol: biaya evaluasi TIDAK dominan setelah di-subset ke 512
    puzzle, tetapi tumbuh sebanding dengan D_eff -- itulah alasan jadwal evaluasi
    harus dikunci lintas konfigurasi, sebab jadwal yang bervariasi akan memberi
    keuntungan sistematis kepada model dangkal.
    """
    import bisect
    import statistics as _st
    rec = os.path.join(data_root, "recipe_out")
    print("\n=== EVAL-SHARE (pangsa wall-time untuk evaluasi) ===")
    rows = []
    for depth, batch in ((9, "b192"), (18, "b192"), (36, "b96")):
        vals = []
        for seed in (0, 1, 2):
            path = os.path.join(rec, f"progress_h512_d{depth}_recipe_{batch}_s{seed}.jsonl")
            if not os.path.exists(path):
                continue
            train_t, eval_t = [], []
            for line in open(path):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "t" not in d:
                    continue
                (eval_t if d.get("phase") == "eval" else train_t).append(d["t"])
            if len(train_t) < 10:
                continue
            train_t.sort()
            eval_t.sort()
            total = train_t[-1] - train_t[0]
            step = _st.median([b - a for a, b in zip(train_t, train_t[1:])])
            spent = 0.0
            for e in eval_t:
                i = bisect.bisect_left(train_t, e)
                if 0 < i < len(train_t):
                    spent += max(0.0, (train_t[i] - train_t[i - 1]) - step)
            vals.append(100.0 * spent / total)
        if vals:
            m = _st.mean(vals)
            sd = _st.stdev(vals) if len(vals) > 1 else 0.0
            rows.append((depth, m, sd, len(vals)))
            print(f"  D_eff={depth:>2}  eval {m:5.2f}% +- {sd:.2f}  (n={len(vals)} seed)")
    if len(rows) >= 2:
        print(f"  rasio D36/D9 = {rows[-1][1] / rows[0][1]:.2f}x untuk kedalaman {rows[-1][0] // rows[0][0]}x")
    return rows


def _tcrit(df, alpha=0.05):
    """Kuantil t dua-sisi via bisection pada betai (tanpa SciPy)."""
    from stats_table import betai
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betai(df / 2.0, 0.5, df / (df + mid * mid)) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fig_baseline_distance(out_path, data_root):
    """Kiri: posisi tiap task relatif terhadap baseline sepele dan langit-langit skalanya.
    Kanan: efek kedalaman bertanda dengan CI 95% Welch.

    Fase BM: panel kiri tidak lagi memakai 'headroom' (sumbu peta rezim lama), karena
    reviewer benar bahwa jarak-ke-100% tak sebanding antar metrik. Yang ditampilkan kini
    adalah acuan yang bisa diperiksa siapa pun: baseline copy-input / majority-class pada
    subset evaluasi yang sama. Maze-Hard duduk DI BAWAH baseline copy-input, sehingga
    metrik tokennya tidak mengukur penguasaan task pada anggaran ini.
    """
    from stats_table import welch
    rec = os.path.join(data_root, "recipe_out")
    # Baseline sepele dari ablation/trivial_baselines.md. Sudoku memakai akurasi EXACT, yang
    # baseline sepelenya 0%; Maze dan ARC memakai akurasi token (copy-input 87,51% dan 25,00%).
    specs = [
        ("Sudoku-Extreme", "exact", os.path.join(rec, "recipe_summary.csv"), "best_exact_pct",
         lambda r: r["hidden"] == "512" and "recipe" in r["tag"], 0.0, "informative"),
        ("ARC-AGI-1", "token", os.path.join(data_root, "arc_depth_out", "recipe_summary.csv"),
         "best_token_pct", lambda r: True, 25.0, "informative"),
        ("Maze-Hard", "token", os.path.join(data_root, "maze_depth_out", "recipe_summary.csv"),
         "best_token_pct", lambda r: True, 87.5, "at floor"),
    ]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True,
                                   gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.08})
    shades = {"informative": OI["blue"], "at floor": OI["vermillion"]}
    marks = {"9": "o", "18": "s", "36": "^"}
    ylabels = []
    for row, (task, metric, path, col, filt, base, status) in enumerate(specs):
        y = len(specs) - 1 - row
        groups = {}
        for r in csv.DictReader(open(path)):
            if filt(r):
                groups.setdefault(r["D_eff"], []).append(float(r[col]))
        means = {k: float(np.mean(v)) for k, v in groups.items()}
        # Fase BM: sumbu kiri adalah JARAK ke baseline sepele, bukan akurasi absolut. Pada skala
        # absolut 0-100% jarak Maze (-1 poin) tak terlihat, sehingga pesan utama figur hilang.
        axL.plot([-6, 94], [y, y], color="0.88", lw=5, solid_capstyle="butt", zorder=1)
        for k in ("9", "18", "36"):
            axL.errorbar(means[k] - base, y, xerr=np.std(groups[k], ddof=1), fmt=marks[k], ms=4.4,
                         color=shades[status], ecolor=shades[status], elinewidth=1, capsize=2,
                         zorder=3, mfc="white" if status == "at floor" else shades[status])
        lo_d = min(means[k] - base for k in ("9", "18", "36"))
        hi_d = max(means[k] - base for k in ("9", "18", "36"))
        axL.annotate(f"{lo_d:+.1f} to {hi_d:+.1f}", (hi_d, y), textcoords="offset points",
                     xytext=(8, -2), fontsize=7.4, color=shades[status], va="center")
        d9, d36 = groups["9"], groups["36"]
        t, df, pval, _ = welch(d36, d9)
        diff = means["36"] - means["9"]
        ci = _tcrit(df) * abs(diff / t)
        sig = pval < 0.05 / 3
        axR.errorbar(diff, y, xerr=ci, fmt="D", ms=5, color=shades[status], ecolor=shades[status],
                     elinewidth=1.3, capsize=3, mfc=shades[status] if sig else "white", zorder=3)
        axR.text(diff - 1.2, y - 0.34, (f"{diff:+.1f}" + ("" if sig else " n.s.")), ha="right",
                 fontsize=8.6, color=shades[status])
        ylabels.append((y, f"{task}\n({metric}), {status}"))
    axL.axvline(0, color=OI["vermillion"], lw=1.1, zorder=2)
    axL.text(0, len(specs) - 0.5, "trivial baseline", rotation=90, ha="right", va="top",
             fontsize=6.8, color=OI["vermillion"])
    # Batas kanan 94, bukan 82: label rentang Sudoku ("+36.3 to +62.4", 53 pt pada 7.4 pt)
    # mulai 8 pt di kanan penanda D=9 dan dulu menembus spine, lalu tertutup latar putih
    # panel kanan sehingga teksnya terbaca terpotong. Batas ini murni ruang gambar.
    axL.set_xlim(-6, 94)
    axL.set_ylim(-0.75, len(specs) - 0.35)
    axL.set_yticks([y for y, _ in ylabels])
    axL.set_yticklabels([lab for _, lab in ylabels], fontsize=8.2)
    axL.set_xlabel("Distance from the trivial baseline (points)")
    axR.axvline(0, color="0.45", lw=0.9, zorder=2)
    axR.set_xlabel(r"Depth effect, $D_{36}-D_9$ (points)")
    # Idem: "-26.2" rata-kanan di bawah diamond Sudoku melewati spine kiri pada batas -33.
    axR.set_xlim(-38, 12)
    for ax in (axL, axR):
        ax.grid(axis="x", alpha=0.25, lw=0.5)
    hs = [plt.Line2D([], [], marker=m, ls="", color="0.35", ms=5.2,
                     label=rf"$D_{{\mathrm{{eff}}}}{{=}}{k}$") for k, m in marks.items()]
    axL.legend(handles=hs, frameon=False, fontsize=8.0, loc="lower left",
               ncol=3, handletextpad=0.2, columnspacing=0.9, borderpad=0.1)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")

def fig_protocol(out_path):
    """Diagram alur protokol pengukuran. Skema, bukan figur data.

    Sengaja memuat gerbang penolakan yang benar-benar dipakai (kesepakatan dua
    instrumen >=98.8%, run <120 s dibuang) supaya diagram ini menyampaikan
    keputusan, bukan sekadar hiasan kotak-panah.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    # Beri margin: tanpa ini kotak paling kiri dan paling kanan terpotong tepi gambar.
    ax.set_xlim(-4, 108)
    ax.set_ylim(-2, 47)
    ax.axis("off")

    def box(x, y, w, h, lines, fc="white", ec="0.35", fs=7.6, mono=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9, zorder=2))
        ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
                fontsize=fs, zorder=3,
                fontfamily="monospace" if mono else None, linespacing=1.45)

    def arrow(x1, y1, x2, y2, color="0.35", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     color=color, lw=0.9, ls=ls, shrinkA=1, shrinkB=1, zorder=1))

    box(0, 19, 21, 12, ["Task and", "iso-compute budget",
                        r"$P \times D_\mathrm{eff} \times$ steps", "held fixed"])
    box(24.5, 19, 22, 12, ["Train, one GPU", "fixed thermal env.", "eval schedule pinned",
                           "to a 512-puzzle subset"])
    box(48, 33, 20, 10, ["nvidia-smi", "power.draw, 1 Hz"], fc="#EAF2FA")
    box(48, 5, 20, 10, ["CodeCarbon", "NVML energy"], fc="#EAF2FA")
    box(68.5, 19, 20, 12, ["idle subtraction", "4.7 W constant", "cross-validate:",
                           "agree 98.8-99.95%"])
    box(91.5, 19, 12.5, 12, ["Joules to", "target", "accuracy"], fc="#FDF0E3")

    arrow(21.3, 25, 24.2, 25)
    arrow(46.8, 27.6, 47.7, 35.0)
    arrow(46.8, 22.4, 47.7, 13.0)
    arrow(68.3, 36.0, 68.4, 27.6)
    arrow(68.3, 12.0, 68.4, 22.4)
    arrow(88.8, 25, 91.2, 25)

    arrow(33, 18.3, 33, 3.6, color="0.45", ls=(0, (2.5, 1.6)))
    ax.text(50, 1.2, "rejected: wall time < 120 s (aborted / OOM) or incomplete budget",
            ha="center", va="bottom", fontsize=8.4, color="0.45", style="italic")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_sudoku_frontier(out_path, data_root):
    """Akurasi akhir Sudoku terhadap energi neto. SATU panel.

    Panel "akurasi vs kedalaman" yang dulu ada di sini dibuang karena menduplikasi
    panel pertama fig_crosstask_depth: data dan bentuk plotnya sama persis.
    """
    rec = os.path.join(data_root, "recipe_out")
    # Encoding kedalaman disamakan dengan Gbr. 4-5 (biru/oranye/vermillion, o/s/^);
    # kotak hitam-putih yang lama tak bisa dihubungkan pembaca ke figur tetangga.
    style = {9: (OI["blue"], "o"), 18: (OI["orange"], "s"), 36: (OI["vermillion"], "^")}
    pts = []
    for depth, batch in ((9, "b192"), (18, "b192"), (36, "b96")):
        acc, wh = [], []
        for r in csv.DictReader(open(os.path.join(rec, "recipe_summary.csv"))):
            if r["hidden"] == "512" and r["D_eff"] == str(depth) and "recipe" in r["tag"]:
                acc.append(float(r["best_exact_pct"])); wh.append(float(r["smi_net_Wh"]))
        pts.append((depth, float(np.mean(wh)), float(np.mean(acc)),
                    float(np.std(wh, ddof=1)) if len(wh) > 1 else 0.0,
                    float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0, wh, acc))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    ax.plot(xs, ys, "-", color="0.35", lw=1.2, zorder=1)
    for d, x, y, sx, sy, whs, accs in pts:
        col, mk = style[d]
        # Titik per seed + error bar: rata-rata saja membuang sebaran yang justru
        # menentukan apakah jarak antar kedalaman berarti.
        ax.scatter(whs, accs, s=11, color=col, alpha=0.40, marker=mk, zorder=2, linewidth=0)
        ax.errorbar(x, y, xerr=sx, yerr=sy, fmt=mk, ms=6, mfc="white", mec=col, mew=1.3,
                    color=col, ecolor=col, elinewidth=1.0, capsize=2.5, zorder=3)
        # D36 duduk paling bawah: labelnya digeser ke kanan-bawah supaya tidak menimpa
        # cap error bar horizontalnya (yang lebarnya s.d. energi).
        off = {9: (-30, -14), 36: (12, -13)}.get(d, (9, -3))
        ax.annotate(rf"$D_{{{d}}}$", (x, y), textcoords="offset points", xytext=off,
                    fontsize=PANEL6_TICK, color=col)
    # Font dinaikkan lewat konstanta PANEL6_*: pada tick 8 pt lama, hasil cetaknya hanya
    # 6,9 pt, di bawah ambang 7 pt Elsevier.
    ax.set_xlabel("Net training energy (Wh)", fontsize=PANEL6_LABEL)
    ax.set_ylabel("Exact accuracy (%)", fontsize=PANEL6_LABEL)
    ax.tick_params(labelsize=PANEL6_TICK)
    allx = [v for p in pts for v in p[5]]; ally = [v for p in pts for v in p[6]]
    ax.set_xlim(min(allx) - 14, max(allx) + 14)
    # Batas y memperhitungkan ujung error bar, bukan hanya titik seed, lalu diberi
    # padding bawah: tanpa itu titik D36 dan cap-nya menempel garis sumbu. 11% dipakai
    # (bukan 8%) karena label $D_{36}$ digantung 13 pt di bawah titiknya.
    ylo = min(min(ally), min(p[2] - p[4] for p in pts))
    yhi = max(max(ally), max(p[2] + p[4] for p in pts))
    span = yhi - ylo
    ax.set_ylim(ylo - 0.11 * span, yhi + 0.06 * span)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(out_path); plt.close(fig)
    print(f"  tulis {out_path}")


def fig_width_optimum(out_path, data_root):
    """Sumbu lebar pada D_eff=18: optimum di tengah pada h=512."""
    rec = os.path.join(data_root, "recipe_out", "recipe_summary.csv")
    g = {}
    for r in csv.DictReader(open(rec)):
        if r["D_eff"] == "18" and "recipe" in r["tag"]:
            g.setdefault(int(r["hidden"]), []).append(float(r["best_exact_pct"]))
    hs = sorted(g)
    m = [float(np.mean(g[h])) for h in hs]
    sd = [float(np.std(g[h], ddof=1)) for h in hs]
    # Lebar disetel 2,671 in supaya lebar PDF keluaran (setelah bbox="tight") berbanding
    # 0.43/0.50 terhadap panel kiri: dengan begitu kedua panel Gbr. 6 menyusut pada skala
    # cetak yang sama (0,864x) dan fontnya terbaca sama besar, bukan kiri lebih kecil.
    fig, ax = plt.subplots(figsize=(2.671, 2.6))
    ax.errorbar(range(len(hs)), m, yerr=sd, fmt="D-", ms=5.5, mfc="white",
                color=OI["reddish"], ecolor=OI["reddish"], capsize=3, lw=1.3, mew=1.2)
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([str(h) for h in hs])
    ax.tick_params(labelsize=PANEL6_TICK)
    ax.set_xlabel("hidden size $h$", fontsize=PANEL6_LABEL)
    ax.set_ylabel("Exact accuracy (%)", fontsize=PANEL6_LABEL)
    ax.set_xlim(-0.35, len(hs) - 0.65)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(out_path); plt.close(fig)
    print(f"  tulis {out_path}")


def fig_design_plane(out_path, data_root):
    """Bidang rancangan (P, D_eff): titik yang benar-benar dijalankan."""
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    ax.plot([512, 512, 512], [9, 18, 36], "-", color="0.6", lw=0.9, zorder=1)
    ax.plot([256, 512, 768], [18, 18, 18], "-", color="0.6", lw=0.9, zorder=1)
    ax.plot([512] * 3, [9, 18, 36], "o", ms=8, mfc="none", mec=OI["blue"], mew=1.6,
            label=r"depth sweep ($h{=}512$)", zorder=3)
    ax.plot([256, 512, 768], [18] * 3, "s", ms=8, mfc="none", mec=OI["reddish"], mew=1.6,
            label=r"width sweep ($D_{\mathrm{eff}}{=}18$)", zorder=3)
    ax.plot([512], [1], "^", ms=9, mfc="none", mec="0.15", mew=1.6,
            label="non-recursive baseline", zorder=3)
    ax.plot([256, 256], [9, 36], "o", ms=8, mfc="none", mec=OI["blue"], mew=1.6, alpha=0.45,
            zorder=2, label=r"Maze and ARC depth sweep ($h{=}256$)")
    # Titik (256,18) dipakai bersama dengan sapuan lebar. Alih-alih menggesernya ke
    # posisi h yang tidak pernah dijalankan, ia digambar sebagai lingkaran lebih besar
    # yang MELINGKARI kotak sapuan lebar: keduanya terbaca, datanya tetap di tempat.
    ax.plot([256], [18], "o", ms=13, mfc="none", mec=OI["blue"], mew=1.6, alpha=0.45, zorder=2)
    ax.plot([256, 256, 256], [9, 18, 36], color="0.75", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.set_yscale("log"); ax.set_yticks([1, 9, 18, 36])
    ax.set_yticklabels(["1", "9", "18", "36"])
    ax.set_xticks([256, 512, 768])
    ax.set_xlabel(r"hidden size $h$")
    ax.set_ylabel(r"recursion depth $D_{\mathrm{eff}}$")
    ax.set_xlim(175, 880); ax.set_ylim(0.62, 62)
    ax.grid(alpha=0.25, lw=0.5)
    # Legenda ditaruh di luar kanan-atas area data: di dalam plot ia selalu
    # bertabrakan, entah dgn titik D=36, penanda baseline, atau titik D=9.
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0.015, 0.52),
              handletextpad=0.4, borderpad=0.2, labelspacing=0.35)
    fig.savefig(out_path); plt.close(fig)
    print(f"  tulis {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "manuscript", "figures"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    root = os.path.abspath(a.data_root)
    fig_energy_accuracy(os.path.join(out, "fig_energy_accuracy.pdf"), root)
    fig_crosstask(os.path.join(out, "fig_crosstask_depth.pdf"), root)
    fig_learning(os.path.join(out, "fig_learning.pdf"), root)
    fig_iso_accuracy(os.path.join(out, "fig_iso_accuracy.pdf"), root)
    fig_baseline_distance(os.path.join(out, "fig_baseline_distance.pdf"), root)
    fig_protocol(os.path.join(out, "fig_protocol.pdf"))
    fig_sudoku_frontier(os.path.join(out, "fig_sudoku_frontier.pdf"), root)
    fig_width_optimum(os.path.join(out, "fig_width_optimum.pdf"), root)
    fig_design_plane(os.path.join(out, "fig_design_plane.pdf"), root)
    mb = os.path.join(root, "microbench_results.csv")
    if not os.path.exists(mb):
        mb_alt = os.path.join(root, "..", "kalibrasi", "microbench_results.csv")
        root = os.path.dirname(os.path.abspath(mb_alt))
    fig_cost_model(os.path.join(out, "fig_cost_model.pdf"), root)
    joules_to_target_table(os.path.abspath(a.data_root))
    perseed_table(os.path.abspath(a.data_root))
    pilot_vs_faithful(os.path.abspath(a.data_root))
    agreement_floor(os.path.abspath(a.data_root))
    iso_accuracy_energy(os.path.abspath(a.data_root))
    iso_accuracy_energy_arc(os.path.abspath(a.data_root))
    eval_share_table(os.path.abspath(a.data_root))


if __name__ == "__main__":
    main()
