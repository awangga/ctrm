#!/usr/bin/env python3
"""Konsolidasi dataset ablasi terpadu + learning curves siap-plot (aturan integritas #6).

Menyatukan tiga sumber run:
  frontier_out/  (hidden=256, depth-sweep, budget pilot)
  isoflop_out/   (hidden 128/256/384 x D 9/18/36, iso-FLOP grid)
  budget_out/    (hidden=128, D 9/18/36, budget panjang + replikasi multi-seed)

Output (ke eksperimen/frontier/ablation/):
  ablation_master.csv   - 1 baris/run, self-describing: sumber, config lengkap,
                          metrik final (exact/token/lm_loss/q_halt), energi, wall.
  learning_curves_all.csv - long-form semua eval-checkpoint tiap run (plot-ready).
  fig_*.png             - learning curves + energy-accuracy frontier.
  ablation_report.md    - ringkasan + tabel.

Energi: net_energy_Wh dari summary bila ada; jika tidak, integral trapezoid pw_*.csv.
Config tiap titik diturunkan dari summary CSV (otoritatif) + tag + isi progress jsonl.
"""
import csv, datetime as dt, glob, json, math, os

BASE = "/home/adb/awangga/trm/eksperimen/frontier"
SRCS = {
    "frontier": os.path.join(BASE, "frontier_out"),
    "isoflop":  os.path.join(BASE, "isoflop_out"),
    "budget":   os.path.join(BASE, "budget_out"),
    "scale":    os.path.join(BASE, "scale_out"),
    "maze":     os.path.join(BASE, "maze_out"),
    "converge": os.path.join(BASE, "converge_out"),
}
ABL = os.path.join(BASE, "ablation"); os.makedirs(ABL, exist_ok=True)
IDLE_W = 4.7


def parse_ts(s):
    return dt.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def integrate_pw(pwcsv):
    """(gross_Wh, net_Wh, dur_s, mean_W, n) dari trace nvidia-smi."""
    ts, pw = [], []
    for ln in open(pwcsv):
        p = ln.split(",")
        if len(p) < 2:
            continue
        try:
            ts.append(parse_ts(p[0])); pw.append(float(p[1]))
        except Exception:
            continue
    if len(ts) < 2:
        return None
    g = n = 0.0
    for i in range(1, len(ts)):
        d = ts[i] - ts[i - 1]
        if d <= 0 or d > 30:
            continue
        g += 0.5 * (pw[i] + pw[i - 1]) * d
        n += 0.5 * ((pw[i] - IDLE_W) + (pw[i - 1] - IDLE_W)) * d
    return g / 3600, n / 3600, ts[-1] - ts[0], sum(pw) / len(pw), len(pw)


def parse_tag(tag):
    """Turunkan hidden, D_eff, seed dari tag progress file."""
    h = d = None; seed = "0"
    if tag.startswith("h"):
        try: h = int(tag.split("_")[0][1:])
        except Exception: h = None
    if "_d" in tag:
        seg = tag.split("_d")[1]
        dd = seg.split("_")[0].lstrip("0") or "0"
        # frontier uses d09 -> 9
        try: d = int(seg.split("_")[0])
        except Exception: d = None
    if "_s" in tag:
        try: seed = tag.split("_s")[1].split(".")[0]
        except Exception: seed = "0"
    return h, d, seed


def last_eval(pj):
    """eval-record terakhir (dict metrik) + jumlah eval/train record."""
    last = None; ne = nt = 0
    for ln in open(pj):
        if '"phase"' not in ln:
            continue
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("phase") == "eval":
            ne += 1; last = r
        elif r.get("phase") == "train":
            nt += 1
    return last, ne, nt


def load_summary(path):
    """Map tag(progress filename stem) -> baris summary (config + energi)."""
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        pjn = r.get("progress_jsonl", "")
        if pjn:
            tag = os.path.basename(pjn)[len("progress_"):-len(".jsonl")]
            out[tag] = r
    return out


def main():
    master = []          # 1 row/run
    lc_rows = []         # long-form learning curves
    # ambil energy_xval bila ada (run baru): map tag->gross/net/cc
    xval = {}
    xvp = os.path.join(SRCS["budget"], "energy_xval.csv")
    if os.path.exists(xvp):
        for r in csv.DictReader(open(xvp)):
            xval[r["tag"]] = r

    for src, d in SRCS.items():
        summ = {}
        for sp in glob.glob(os.path.join(d, "*summary*.csv")):
            summ.update(load_summary(sp))
        for pj in sorted(glob.glob(os.path.join(d, "progress_*.jsonl"))):
            tag = os.path.basename(pj)[len("progress_"):-len(".jsonl")]
            h, deff, seed = parse_tag(tag)
            srow = summ.get(tag, {})
            # frontier_out tags d09/d18/d36: hidden dari summary
            if h is None and srow.get("hidden"):
                try: h = int(srow["hidden"])
                except Exception: pass
            if deff is None and srow.get("D_eff"):
                try: deff = int(srow["D_eff"])
                except Exception: pass
            last, ne, nt = last_eval(pj)
            if last is None:
                continue
            # energi: prioritas summary net_energy_Wh, lalu xval, lalu integrate pw
            net = srow.get("net_energy_Wh")
            gross = None; mean_w = None; cc_gpu = None; agree = None
            pw = os.path.join(d, f"pw_{tag}.csv")
            if tag in xval:
                x = xval[tag]
                net = x.get("smi_net_Wh") or net
                gross = x.get("smi_gross_Wh"); mean_w = x.get("smi_mean_W")
                cc_gpu = x.get("cc_gpu_Wh"); agree = x.get("gross_agree_pct")
            elif os.path.exists(pw):
                ip = integrate_pw(pw)
                if ip:
                    gross = round(ip[0], 4); mean_w = round(ip[3], 2)
                    if net is None: net = round(ip[1], 4)
            # learning curves long-form
            for ln in open(pj):
                if '"phase": "eval"' not in ln:
                    continue
                try: r = json.loads(ln)
                except Exception: continue
                lc_rows.append({
                    "source": src, "tag": tag, "hidden": h, "D_eff": deff, "seed": seed,
                    "step": r.get("step"),
                    "exact_pct": round(r.get("all/exact_accuracy", 0) * 100, 4),
                    "token_pct": round(r.get("all/accuracy", 0) * 100, 4),
                    "lm_loss": r.get("all/lm_loss"),
                })
            master.append({
                "source": src, "tag": tag, "hidden": h, "D_eff": deff,
                "L_cycles": (deff // 3 if deff else None), "H_cycles": 3, "seed": seed,
                "params_M": srow.get("params_M"),
                "epochs": srow.get("epochs"), "eval_interval": srow.get("eval_interval"),
                "wall_s": srow.get("wall_s"),
                "n_eval_ckpt": ne, "n_train_step": nt,
                "exact_pct": round(last.get("all/exact_accuracy", 0) * 100, 4),
                "token_pct": round(last.get("all/accuracy", 0) * 100, 4),
                "lm_loss": last.get("all/lm_loss"),
                "q_halt_acc": last.get("all/q_halt_accuracy"),
                "net_energy_Wh": net, "gross_energy_Wh": gross, "mean_W": mean_w,
                "cc_gpu_Wh": cc_gpu, "gross_agree_pct": agree,
                "idle_W": IDLE_W,
            })

    # tulis CSV
    mfields = ["source", "tag", "hidden", "D_eff", "L_cycles", "H_cycles", "seed",
               "params_M", "epochs", "eval_interval", "wall_s", "n_eval_ckpt", "n_train_step",
               "exact_pct", "token_pct", "lm_loss", "q_halt_acc",
               "net_energy_Wh", "gross_energy_Wh", "mean_W", "cc_gpu_Wh", "gross_agree_pct", "idle_W"]
    master.sort(key=lambda x: (x["source"], x["hidden"] or 0, x["D_eff"] or 0, x["seed"]))
    mp = os.path.join(ABL, "ablation_master.csv")
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfields); w.writeheader()
        for r in master: w.writerow({k: r.get(k) for k in mfields})

    lcp = os.path.join(ABL, "learning_curves_all.csv")
    lfields = ["source", "tag", "hidden", "D_eff", "seed", "step", "exact_pct", "token_pct", "lm_loss"]
    with open(lcp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lfields); w.writeheader()
        for r in lc_rows: w.writerow(r)

    # ---- PLOTS ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) learning curve: budget runs (hidden=128) exact% vs step, warna per depth, seed tipis
    plt.figure(figsize=(6.4, 4.4))
    colors = {9: "tab:blue", 18: "tab:orange", 36: "tab:green"}
    seen = set()
    by = {}
    for r in lc_rows:
        if r["source"] != "budget" or r["hidden"] != 128:
            continue
        by.setdefault((r["D_eff"], r["seed"]), []).append((r["step"], r["exact_pct"]))
    for (deff, seed), pts in sorted(by.items()):
        pts.sort()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        lab = f"D={deff}" if deff not in seen else None
        seen.add(deff)
        plt.plot(xs, ys, color=colors.get(deff, "gray"), alpha=0.45, lw=1, marker="o", ms=2, label=lab)
    plt.xlabel("training step (compute)"); plt.ylabel("exact (puzzle) accuracy (%)")
    plt.title("Learning curves: exact-acc vs compute per depth (hidden=128, semua seed)")
    plt.legend(title="recursion depth"); plt.grid(True, alpha=0.3); plt.tight_layout()
    p1 = os.path.join(ABL, "fig_learning_budget_exact.png"); plt.savefig(p1, dpi=200); plt.close()

    # 2) energy-accuracy frontier: scatter net energy vs exact, semua run berenergi
    plt.figure(figsize=(6.4, 4.4))
    mk = {"frontier": "s", "isoflop": "^", "budget": "o", "scale": "D", "maze": "*", "converge": "P"}
    for src in SRCS:
        xs = []; ys = []
        for r in master:
            try: e = float(r["net_energy_Wh"])
            except (TypeError, ValueError): continue
            if r["source"] != src: continue
            xs.append(e); ys.append(r["exact_pct"])
        if xs:
            plt.scatter(xs, ys, marker=mk[src], label=src, alpha=0.7)
    plt.xlabel("net energy (Wh, dikurangi idle)"); plt.ylabel("exact accuracy (%)")
    plt.title("Frontier energi-akurasi (semua run)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    p2 = os.path.join(ABL, "fig_energy_accuracy_frontier.png"); plt.savefig(p2, dpi=200); plt.close()

    # 3) iso-FLOP: token-acc vs depth per hidden (menunjukkan depth merugikan pd budget pendek)
    plt.figure(figsize=(6.4, 4.4))
    iso = {}
    for r in master:
        if r["source"] != "isoflop":
            continue
        iso.setdefault(r["hidden"], []).append((r["D_eff"], r["token_pct"]))
    for h, pts in sorted(iso.items()):
        pts.sort()
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"hidden={h}")
    plt.xlabel("recursion depth D_eff"); plt.ylabel("token accuracy (%)")
    plt.title("Iso-FLOP: efek depth pada akurasi per lebar (budget pendek)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    p3 = os.path.join(ABL, "fig_isoflop_depth.png"); plt.savefig(p3, dpi=200); plt.close()

    # ---- REPORT ----
    nseed = len(set(r["seed"] for r in master if r["source"] == "budget" and r["hidden"] == 128))
    L = []
    L.append("# Dataset ablasi terkonsolidasi + learning curves")
    L.append("")
    L.append(f"Dibangun dari 3 sumber run ({len(master)} run total, {len(lc_rows)} eval-checkpoint). "
             "Tiap baris `ablation_master.csv` self-describing (config penuh + metrik + energi).")
    L.append("")
    L.append("## Artefak")
    L.append("- `ablation_master.csv` — 1 baris/run (source, hidden, D_eff, L/H_cycles, seed, params, "
             "epochs, eval_interval, wall, exact/token/lm_loss/q_halt, net/gross energy, mean_W, cc_gpu_Wh, %agree).")
    L.append("- `learning_curves_all.csv` — long-form semua eval-checkpoint (plot-ready).")
    L.append("- `fig_learning_budget_exact.png`, `fig_energy_accuracy_frontier.png`, `fig_isoflop_depth.png`.")
    L.append("")
    L.append("## Cakupan")
    cov = {}
    for r in master:
        cov.setdefault(r["source"], set()).add((r["hidden"], r["D_eff"]))
    for src in SRCS:
        cells = sorted(c for c in cov.get(src, []))
        L.append(f"- **{src}**: {len([r for r in master if r['source']==src])} run, "
                 f"{len(cells)} sel (hidden×D). budget hidden=128 seed unik: {nseed}.")
    L.append("")
    L.append("## Tabel ringkas (per run)")
    L.append("")
    L.append("| source | hidden | D_eff | seed | exact% | token% | lm_loss | net Wh | gross Wh | %agree |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in master:
        def fmt(v, n=3):
            try: return f"{float(v):.{n}f}"
            except (TypeError, ValueError): return "—"
        L.append(f"| {r['source']} | {r['hidden']} | {r['D_eff']} | {r['seed']} | "
                 f"{fmt(r['exact_pct'])} | {fmt(r['token_pct'])} | {fmt(r['lm_loss'],4)} | "
                 f"{fmt(r['net_energy_Wh'])} | {fmt(r['gross_energy_Wh'])} | {fmt(r['gross_agree_pct'],2)} |")
    L.append("")
    open(os.path.join(ABL, "ablation_report.md"), "w").write("\n".join(L) + "\n")

    print(f"master rows: {len(master)} -> {mp}")
    print(f"lc rows: {len(lc_rows)} -> {lcp}")
    print("figs:", p1, p2, p3)
    print(f"budget hidden=128 unique seeds: {nseed}")


if __name__ == "__main__":
    main()
