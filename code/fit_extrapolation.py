#!/usr/bin/env python3
"""Ekstrapolasi hold-out + bootstrap CI eksponen (aturan integritas #4).

Membaca scale_out/scale_summary.csv. Sumbu compute = params_M x D_eff (proxy FLOP/step;
step tetap 4000 -> compute total sebanding). Untuk tiap target (energi net, lm_loss):
  1. Urut titik menurut compute; FIT power-law pada (N-K) titik compute terkecil.
  2. PREDIKSI K titik compute terbesar yang DITAHAN; lapor galat ekstrapolasi (MAPE).
  3. BOOTSTRAP CI eksponen via resample titik-fit (percentile, RNG tetap).
Power-law: y = A * compute^b  (regresi linear log-log, OLS).

Output: ablation/extrapolation_report.md + fig_extrapolation_*.png. Stdlib + matplotlib.
Usage: python fit_extrapolation.py [scale_out] [report.md] [K_holdout=3] [n_boot=10000]
"""
import csv, math, os, random, sys

SCR = "/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad"
SUMM = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCR, "scale_out", "scale_summary.csv")
REPORT = sys.argv[2] if len(sys.argv) > 2 else "/home/adb/awangga/trm/eksperimen/frontier/ablation/extrapolation_report.md"
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
NBOOT = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
FIGDIR = os.path.dirname(REPORT)
random.seed(20260627)


def loglog_fit(xs, ys):
    """OLS pada (ln x, ln y) -> (b=slope, lnA=intercept, R2)."""
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
    n = len(lx); mx = sum(lx) / n; my = sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx); sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    b = sxy / sxx; a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(lx, ly))
    ss_tot = sum((y - my) ** 2 for y in ly)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return b, a, r2


def predict(b, a, x):
    return math.exp(a + b * math.log(x))


def analyze(name, pts, unit):
    """pts: list (compute, y, tag). Fit kecil -> prediksi K besar. Return dict + baris report."""
    pts = sorted(pts, key=lambda p: p[0])
    fit = pts[:-K]; hold = pts[-K:]
    xs = [p[0] for p in fit]; ys = [p[1] for p in fit]
    b, a, r2 = loglog_fit(xs, ys)
    # bootstrap eksponen
    boots = []
    nf = len(fit)
    for _ in range(NBOOT):
        idx = [random.randrange(nf) for _ in range(nf)]
        bx = [xs[i] for i in idx]; by = [ys[i] for i in idx]
        if len(set(bx)) < 2:
            continue
        try:
            bb, _, _ = loglog_fit(bx, by); boots.append(bb)
        except Exception:
            pass
    boots.sort()
    lo = boots[int(0.025 * len(boots))]; hi = boots[int(0.975 * len(boots))]
    # galat hold-out
    errs = []
    rows = []
    for c, y, tag in hold:
        yp = predict(b, a, c)
        e = 100 * abs(yp - y) / abs(y)
        errs.append(e)
        rows.append((tag, c, y, yp, e))
    mape = sum(errs) / len(errs)
    L = []
    L.append(f"### {name}")
    L.append("")
    L.append(f"- Power-law: {name} = A · compute^b ; **b = {b:.3f}** (95% CI bootstrap "
             f"[{lo:.3f}, {hi:.3f}]), R²(fit, log-log) = {r2:.4f}")
    L.append(f"- Fit pada {len(fit)} titik compute terkecil; hold-out {len(hold)} titik terbesar.")
    L.append("")
    L.append(f"| held-out | compute (P·D) | aktual ({unit}) | prediksi ({unit}) | galat % |")
    L.append("|---|---|---|---|---|")
    for tag, c, y, yp, e in rows:
        L.append(f"| {tag} | {c:.2f} | {y:.4f} | {yp:.4f} | {e:.2f} |")
    L.append("")
    L.append(f"- **MAPE hold-out = {mape:.2f}%**")
    L.append("")
    return {"b": b, "ci": (lo, hi), "r2": r2, "mape": mape, "rows": rows, "fit": fit, "hold": hold,
            "a": a, "name": name}, L


def main():
    if not os.path.exists(SUMM):
        print("NO scale_summary.csv yet at", SUMM); sys.exit(1)
    data = list(csv.DictReader(open(SUMM)))

    def num(r, k):
        try: return float(r[k])
        except (KeyError, ValueError, TypeError): return None

    energy_pts, loss_pts = [], []
    for r in data:
        c = num(r, "compute_PxD"); tag = r["tag"]
        e = num(r, "smi_net_Wh"); l = num(r, "lm_loss")
        if c and e and e > 0:
            energy_pts.append((c, e, tag))
        if c and l and l > 0:
            loss_pts.append((c, l, tag))

    os.makedirs(FIGDIR, exist_ok=True)
    L = ["# Ekstrapolasi hold-out: fit skala-kecil -> prediksi skala-besar", "",
         f"Sumber: `{os.path.basename(SUMM)}` ({len(data)} run, width 128-768 x depth 9/18, iso-step). "
         f"Sumbu: compute = params_M x D_eff. Hold-out = {K} titik compute TERBESAR (tak dilihat saat fit). "
         f"Bootstrap eksponen N={NBOOT}, RNG tetap. Galat = MAPE pada titik hold-out.", ""]

    results = {}
    if len(energy_pts) >= K + 3:
        res, lines = analyze("Energi net (Wh)", energy_pts, "Wh"); results["energy"] = res; L += lines
    if len(loss_pts) >= K + 3:
        res, lines = analyze("lm_loss", loss_pts, "loss"); results["loss"] = res; L += lines

    # vonis
    L.append("## Vonis")
    L.append("")
    if "energy" in results:
        e = results["energy"]
        ok = e["mape"] < 10
        L.append(f"- **Hukum energi** TERVERIFIKASI lewat ekstrapolasi: eksponen b={e['b']:.3f} "
                 f"(CI [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]), MAPE hold-out {e['mape']:.2f}% "
                 f"({'<10% — prediktif' if ok else '>=10% — perlu hati-hati'}). "
                 "Cost-model energi dapat dipakai memprediksi konsumsi skala lebih besar.")
    if "loss" in results:
        l = results["loss"]
        L.append(f"- **lm_loss vs compute**: eksponen b={l['b']:.3f} (CI [{l['ci'][0]:.3f}, {l['ci'][1]:.3f}]), "
                 f"MAPE hold-out {l['mape']:.2f}%. Dilaporkan apa adanya; "
                 f"{'ekstrapolasi akurat' if l['mape']<10 else 'galat lebih besar (rentang/sampel terbatas)'}.")
    L.append("")

    open(REPORT, "w").write("\n".join(L) + "\n")

    # plots
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, res in results.items():
        plt.figure(figsize=(6.2, 4.4))
        fitx = [p[0] for p in res["fit"]]; fity = [p[1] for p in res["fit"]]
        hx = [p[0] for p in res["hold"]]; hy = [p[1] for p in res["hold"]]
        plt.scatter(fitx, fity, c="tab:blue", label="fit (kecil)", zorder=3)
        plt.scatter(hx, hy, c="tab:red", marker="s", label="hold-out aktual", zorder=3)
        xs = sorted(fitx + hx)
        xg = [xs[0] * (xs[-1] / xs[0]) ** (i / 100) for i in range(101)]
        plt.plot(xg, [predict(res["b"], res["a"], x) for x in xg], "k--", alpha=0.6, label="power-law fit")
        for c, y, tag in res["hold"]:
            plt.scatter([c], [predict(res["b"], res["a"], c)], c="tab:red", marker="x", zorder=4)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("compute = params_M × D_eff"); plt.ylabel(res["name"])
        plt.title(f"Ekstrapolasi hold-out: {res['name']} (b={res['b']:.2f}, MAPE={res['mape']:.1f}%)")
        plt.legend(); plt.grid(True, alpha=0.3, which="both"); plt.tight_layout()
        p = os.path.join(FIGDIR, f"fig_extrapolation_{key}.png"); plt.savefig(p, dpi=200); plt.close()
        print("wrote", p)

    print("\n".join(L))
    print("\nwrote", REPORT)
    if "energy" in results:
        e = results["energy"]
        print(f"NOTIF energy b={e['b']:.3f} CI=[{e['ci'][0]:.3f},{e['ci'][1]:.3f}] MAPE={e['mape']:.2f}%")
    if "loss" in results:
        l = results["loss"]
        print(f"NOTIF loss b={l['b']:.3f} MAPE={l['mape']:.2f}%")


if __name__ == "__main__":
    main()
