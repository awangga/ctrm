#!/usr/bin/env python3
"""Uji signifikansi tren exact-accuracy vs recursion depth pada replikasi multi-seed.

Membaca progress_h{H}_d{D}_s{seed}.jsonl di budget_out/, mengambil exact_accuracy
final per (D_eff, seed), lalu:
  1. Bootstrap CI (percentile) untuk slope OLS exact_acc ~ log2(D_eff) atas seed.
  2. Permutation test: apakah slope teramati melampaui null (label depth diacak)?
  3. Welch t-test berpasangan antar depth ekstrem (D=9 vs D=36).

Output: report markdown + CSV per-seed terkonsolidasi. Tanpa SciPy/Numpy (stdlib saja)
agar reproducible di env minimal. RNG di-seed tetap supaya hasil deterministik.

Usage: python sig_test.py <budget_out> <report.md> [hidden=128] [n_boot=10000]
"""
import csv, glob, json, math, os, random, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "eksperimen/frontier/budget_out"
REPORT = sys.argv[2] if len(sys.argv) > 2 else "eksperimen/frontier/ablation/significance_report.md"
HIDDEN = int(sys.argv[3]) if len(sys.argv) > 3 else 128
NBOOT = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
random.seed(20260627)


def final_exact(path):
    """exact_accuracy (%) dari eval-record terakhir di satu progress jsonl."""
    last = None
    for ln in open(path):
        ln = ln.strip()
        if '"phase": "eval"' not in ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        v = r.get("all/exact_accuracy")
        if v is not None:
            last = v * 100.0
    return last


def collect():
    """{D_eff: {seed: exact_pct}} dari semua progress_h{H}_d{D}_s{seed}.jsonl."""
    data = {}
    pat = os.path.join(OUT, f"progress_h{HIDDEN}_d*_s*.jsonl")
    for p in sorted(glob.glob(pat)):
        base = os.path.basename(p)
        # progress_h128_d9_s0.jsonl
        try:
            d = int(base.split("_d")[1].split("_s")[0])
            s = int(base.split("_s")[1].split(".")[0])
        except Exception:
            continue
        v = final_exact(p)
        if v is None:
            continue
        data.setdefault(d, {})[s] = v
    return data


def ols_slope(xs, ys):
    """slope regresi linear y~x (least squares)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def mean(v):
    return sum(v) / len(v)


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def welch_t(a, b):
    """Welch t-statistik + dof (Welch-Satterthwaite)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None
    va, vb = stdev(a) ** 2, stdev(b) ** 2
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return None, None
    t = (mean(a) - mean(b)) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df


def main():
    data = collect()
    depths = sorted(data.keys())
    if not depths:
        print("NO multi-seed data found in", OUT)
        sys.exit(1)

    # dataset per-seed long-form (point cloud untuk bootstrap)
    points = []  # (D_eff, seed, exact_pct, log2D)
    for d in depths:
        for s, v in sorted(data[d].items()):
            points.append((d, s, v, math.log2(d)))

    xs = [p[3] for p in points]
    ys = [p[2] for p in points]
    obs_slope = ols_slope(xs, ys)

    # 1) BOOTSTRAP CI slope: resample titik (D,seed) dengan pengembalian
    boots = []
    n = len(points)
    for _ in range(NBOOT):
        samp = [points[random.randrange(n)] for _ in range(n)]
        bx = [p[3] for p in samp]
        by = [p[2] for p in samp]
        # butuh variasi di x; skip degenerate
        if len(set(bx)) < 2:
            continue
        boots.append(ols_slope(bx, by))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    frac_pos = sum(1 for b in boots if b > 0) / len(boots)

    # 2) PERMUTATION test: acak label depth, sebar exact_acc; berapa sering |slope|>=obs
    perm_ge = 0
    NPERM = NBOOT
    yv = list(ys)
    for _ in range(NPERM):
        random.shuffle(yv)
        ps = ols_slope(xs, yv)
        if abs(ps) >= abs(obs_slope):
            perm_ge += 1
    p_perm = (perm_ge + 1) / (NPERM + 1)

    # 3) Welch antar depth ekstrem
    a = list(data[depths[0]].values())
    b = list(data[depths[-1]].values())
    t, df = welch_t(a, b)

    # per-depth ringkas
    rows = []
    for d in depths:
        vals = list(data[d].values())
        rows.append((d, len(vals), mean(vals), stdev(vals), min(vals), max(vals)))

    # tulis CSV per-seed
    seedcsv = os.path.join(os.path.dirname(REPORT), "perseed_exact.csv")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(seedcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D_eff", "seed", "exact_pct", "log2_D"])
        for d, s, v, lg in points:
            w.writerow([d, s, round(v, 4), round(lg, 4)])

    # report
    L = []
    L.append(f"# Uji signifikansi: exact-accuracy vs recursion depth (hidden={HIDDEN})")
    L.append("")
    L.append(f"Sumber: `{OUT}/progress_h{HIDDEN}_d*_s*.jsonl` "
             f"({len(points)} run = {len(depths)} depth x {len(data[depths[0]])} seed). "
             f"Metrik: exact (puzzle) accuracy final, %. Bootstrap/permutation N={NBOOT}, seed RNG tetap.")
    L.append("")
    L.append("## Ringkasan per-depth (mean +- sd antar seed)")
    L.append("")
    L.append("| D_eff | n_seed | mean exact % | sd | min | max |")
    L.append("|---|---|---|---|---|---|")
    for d, n_, m_, sd_, mn_, mx_ in rows:
        L.append(f"| {d} | {n_} | {m_:.3f} | {sd_:.3f} | {mn_:.3f} | {mx_:.3f} |")
    L.append("")
    L.append("## 1. Bootstrap CI slope (exact% per oktaf depth, basis log2 D_eff)")
    L.append("")
    L.append(f"- Slope teramati: **{obs_slope:.3f} %/oktaf**")
    L.append(f"- 95% CI bootstrap: **[{lo:.3f}, {hi:.3f}]**")
    L.append(f"- Fraksi bootstrap dengan slope>0: **{frac_pos*100:.1f}%**")
    ci_excl = (lo > 0) or (hi < 0)
    L.append(f"- CI {'TIDAK memuat 0 (tren signifikan pada 0.05)' if ci_excl else 'MEMUAT 0 (tren TIDAK signifikan pada 0.05)'}")
    L.append("")
    L.append("## 2. Permutation test (H0: depth tak berpengaruh)")
    L.append("")
    L.append(f"- p (two-sided, |slope|): **{p_perm:.4f}** "
             f"({'tolak H0' if p_perm < 0.05 else 'gagal tolak H0'} pada 0.05)")
    L.append("")
    L.append(f"## 3. Welch t-test D={depths[0]} vs D={depths[-1]}")
    L.append("")
    if t is not None:
        L.append(f"- t = {t:.3f}, df ~ {df:.1f}, mean({depths[0]})={mean(a):.3f}%, mean({depths[-1]})={mean(b):.3f}%")
        L.append(f"- |t| {'>' if abs(t) > 2.0 else '<='} ~2.0 -> "
                 f"{'beda terdeteksi (kasar)' if abs(t) > 2.0 else 'beda TIDAK terdeteksi'} "
                 f"(n kecil; perlakukan sebagai indikatif)")
    else:
        L.append("- tak cukup data per kelompok.")
    L.append("")
    L.append("## Vonis")
    L.append("")
    verdict = ("Tren depth->exact **signifikan** secara statistik." if (ci_excl and p_perm < 0.05)
               else "Tren depth->exact **TIDAK signifikan**: selisih antar-depth masih dalam noise seed. "
                    "Sesuai aturan integritas repo, dilaporkan sebagai **null/indikatif**, bukan hukum. "
                    "Butuh lebih banyak seed untuk mempersempit CI.")
    L.append(verdict)
    L.append("")
    open(REPORT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote", REPORT)
    print("wrote", seedcsv)
    # baris ringkas utk notif
    print(f"NOTIF slope={obs_slope:.3f} CI=[{lo:.3f},{hi:.3f}] p_perm={p_perm:.4f} sig={'YES' if (ci_excl and p_perm<0.05) else 'NO'}")


if __name__ == "__main__":
    main()
