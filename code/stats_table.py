#!/usr/bin/env python3
"""Hitung ulang SEMUA statistik yang dilaporkan naskah dari summary CSV.

Menutup celah telusur: sebelumnya angka Welch t / ANOVA F / Cohen d hanya ada di
prosa naskah dan EXPERIMENT_LOG, bukan dari skrip yang di-commit. Skrip ini membaca
recipe_summary.csv tiap task, menghitung kontras yang dilaporkan, lalu meng-emit
CSV + tabel LaTeX. Stdlib saja (tanpa SciPy) supaya reproducible di env minimal.

Usage: python3 stats_table.py [data_root] [out_prefix]
"""
import csv, math, os, sys, statistics as st

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUTP = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "ablation", "stats_table")


# ---------- distribusi (regularized incomplete beta, Lentz) ----------
def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300: d = 1e-300
        if abs(c) < 1e-300: c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300: d = 1e-300
        if abs(c) < 1e-300: c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """I_x(a,b) teregularisasi."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def welch(x, y):
    """(t, df, p dua-sisi, Cohen d pooled)."""
    n1, n2 = len(x), len(y)
    m1, m2 = st.mean(x), st.mean(y)
    v1, v2 = st.variance(x), st.variance(y)
    se2 = v1 / n1 + v2 / n2
    t = (m1 - m2) / math.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p = betai(df / 2.0, 0.5, df / (df + t * t))
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return t, df, p, (m1 - m2) / sp


def anova(groups):
    """(F, df1, df2, p) satu arah."""
    k = len(groups)
    N = sum(len(g) for g in groups)
    gm = st.mean([v for g in groups for v in g])
    ssb = sum(len(g) * (st.mean(g) - gm) ** 2 for g in groups)
    ssw = sum((v - st.mean(g)) ** 2 for g in groups for v in g)
    df1, df2 = k - 1, N - k
    F = (ssb / df1) / (ssw / df2)
    p = betai(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * F))
    return F, df1, df2, p


# ---------- data ----------
def load(rel, metric):
    """{D_eff atau hidden: [nilai per seed]} dari satu recipe_summary.csv."""
    path = os.path.join(ROOT, rel)
    rows = list(csv.DictReader(open(path)))
    return rows, path


def cells(rows, key, metric, filt=lambda r: True):
    out = {}
    for r in rows:
        if not filt(r):
            continue
        out.setdefault(r[key], []).append(float(r[metric]))
    return out


def main():
    sud, sud_p = load("recipe_out/recipe_summary.csv", "best_exact_pct")
    mz, mz_p = load("maze_depth_out/recipe_summary.csv", "best_token_pct")
    ar, ar_p = load("arc_depth_out/recipe_summary.csv", "best_token_pct")

    sud_depth = cells(sud, "D_eff", "best_exact_pct",
                      lambda r: r["hidden"] == "512" and "recipe" in r["tag"])
    sud_width = cells(sud, "hidden", "best_exact_pct",
                      lambda r: r["D_eff"] == "18" and "recipe" in r["tag"])
    sud_base = cells(sud, "tag", "best_exact_pct", lambda r: "baseline" in r["tag"])
    base_vals = [v for vs in sud_base.values() for v in vs]
    mz_depth = cells(mz, "D_eff", "best_token_pct")
    ar_depth = cells(ar, "D_eff", "best_token_pct")

    FAM = []  # (family, label, kind, payload)
    FAM += [("Sudoku depth", r"$D_9$ vs $D_{18}$", "t", (sud_depth["9"], sud_depth["18"])),
            ("Sudoku depth", r"$D_{18}$ vs $D_{36}$", "t", (sud_depth["18"], sud_depth["36"])),
            ("Sudoku depth", r"$D_9$ vs $D_{36}$", "t", (sud_depth["9"], sud_depth["36"]))]
    FAM += [("Sudoku width", r"$h_{512}$ vs $h_{256}$", "t", (sud_width["512"], sud_width["256"])),
            ("Sudoku width", r"$h_{512}$ vs $h_{768}$", "t", (sud_width["512"], sud_width["768"]))]
    FAM += [("Sudoku baseline", r"$D_9$ vs non-recursive", "t", (sud_depth["9"], base_vals)),
            ("Sudoku baseline", r"$D_{18}$ vs non-recursive", "t", (sud_depth["18"], base_vals)),
            ("Sudoku baseline", r"$D_{36}$ vs non-recursive", "t", (sud_depth["36"], base_vals))]
    # Ketiga sumbu depth memuat SEMUA 3 kontras berpasangan, supaya ambang Bonferroni
    # ditentukan oleh banyaknya kontras yang benar-benar mungkin (3), bukan oleh berapa
    # yang kebetulan kita tabelkan. Sebelumnya Sudoku dapat alpha=0,017 sedangkan ARC
    # dapat 0,050 hanya karena ARC cuma didaftarkan satu baris -- itu tak bisa dibela.
    FAM += [("Maze depth", r"$D_{36}$ vs $D_9$", "t", (mz_depth["36"], mz_depth["9"])),
            ("Maze depth", r"$D_{36}$ vs $D_{18}$", "t", (mz_depth["36"], mz_depth["18"])),
            ("Maze depth", r"$D_{18}$ vs $D_9$", "t", (mz_depth["18"], mz_depth["9"]))]
    FAM += [("ARC depth", r"$D_9$ vs $D_{36}$", "t", (ar_depth["9"], ar_depth["36"])),
            ("ARC depth", r"$D_{18}$ vs $D_{36}$", "t", (ar_depth["18"], ar_depth["36"])),
            ("ARC depth", r"$D_9$ vs $D_{18}$", "t", (ar_depth["9"], ar_depth["18"]))]

    fam_n = {}
    for f, *_ in FAM:
        fam_n[f] = fam_n.get(f, 0) + 1

    rows_out = []
    for fam, lab, _k, (x, y) in FAM:
        t, df, p, d = welch(x, y)
        thr = 0.05 / fam_n[fam]
        rows_out.append(dict(family=fam, contrast=lab, n1=len(x), n2=len(y),
                             mean1=st.mean(x), sd1=st.stdev(x), mean2=st.mean(y), sd2=st.stdev(y),
                             t=t, df=df, p=p, cohen_d=d, bonf_thr=thr, survives=p < thr))

    ANO = [("Sudoku depth", [sud_depth[k] for k in ("9", "18", "36")]),
           ("Sudoku width", [sud_width[k] for k in ("256", "512", "768")]),
           ("Maze depth", [mz_depth[k] for k in ("9", "18", "36")]),
           ("ARC depth", [ar_depth[k] for k in ("9", "18", "36")])]
    ano_out = []
    for fam, gs in ANO:
        F, df1, df2, p = anova(gs)
        ano_out.append(dict(family=fam, F=F, df1=df1, df2=df2, p=p, n_per_cell=len(gs[0])))

    os.makedirs(os.path.dirname(OUTP), exist_ok=True)
    with open(OUTP + "_contrasts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    with open(OUTP + "_anova.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ano_out[0]))
        w.writeheader(); w.writerows(ano_out)


    def pf(p):
        if p >= 0.0005:
            return f"{p:.4f}"
        m, e = f"{p:.1e}".split("e")
        return rf"{m}\times 10^{{{int(e)}}}"


    lines = []
    prev = None
    for r in rows_out:
        if r["family"] != prev:
            a = next((x for x in ano_out if x["family"] == r["family"]), None)
            lines.append(r"\midrule" if prev else "")
            head = (rf"$F({a['df1']},{a['df2']}){{=}}{a['F']:.1f}$, $p{{=}}{pf(a['p'])}$, "
                    rf"$n{{=}}{a['n_per_cell']}$/cell; " if a else "")
            lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{r['family']}}}: {head}"
                         rf"Bonf. $\alpha{{=}}{r['bonf_thr']:.3f}$}} \\")
            prev = r["family"]
        lines.append(rf"\quad {r['contrast']} & {r['mean1']:.2f}$\pm${r['sd1']:.2f} & "
                     rf"{r['mean2']:.2f}$\pm${r['sd2']:.2f} & {r['t']:.2f} & {r['df']:.1f} & "
                     rf"{pf(r['p'])} & {r['cohen_d']:.2f} & {'yes' if r['survives'] else 'no'} \\")
    tex = "\n".join(x for x in lines if x != "" or True)
    open(OUTP + ".tex", "w").write(tex + "\n")

    print(f"sumber: {sud_p}\n        {mz_p}\n        {ar_p}")
    print(f"\n{'family':16s} {'contrast':26s} {'t':>7s} {'df':>5s} {'p':>9s} {'d':>6s} {'bonf':>5s}")
    for r in rows_out:
        print(f"{r['family']:16s} {r['contrast'][:26]:26s} {r['t']:7.2f} {r['df']:5.1f} "
              f"{r['p']:9.5f} {r['cohen_d']:6.2f} {'OK' if r['survives'] else 'no':>5s}")
    print()
    for a in ano_out:
        print(f"ANOVA {a['family']:16s} F({a['df1']},{a['df2']})={a['F']:6.2f}  p={a['p']:.5f}  n={a['n_per_cell']}/sel")
    print(f"\ntulis {OUTP}_contrasts.csv, {OUTP}_anova.csv, {OUTP}.tex")


if __name__ == "__main__":
    main()
