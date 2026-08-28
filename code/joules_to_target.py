#!/usr/bin/env python3
"""Metrik utama paper: Joules-to-target-accuracy + Pareto frontier (energi, akurasi).

Untuk tiap run dengan histori daya (pw_*.csv) + eval (progress_*.jsonl): cari checkpoint
PERTAMA yang mencapai target akurasi tau, lalu integrasikan daya (trapezoid, idle-corrected)
dari awal training sampai timestamp checkpoint itu -> energi kumulatif net (Wh & Joule)
untuk mencapai tau. Bila tak pernah mencapai tau -> 'tak tercapai' (itu sendiri = bukti
saturasi). Dilaporkan untuk beberapa tau (token & exact).

Sumber: budget_out (5 seed, ~30k step) + frontier_out + scale_out. Hanya run yg punya
pw_*.csv yang dipakai (energi terukur). Output: ablation/joules_to_target.csv + report + figur.

Usage: python joules_to_target.py
"""
import csv, datetime as dt, glob, json, os, statistics

BASE = "/home/adb/awangga/trm/eksperimen/frontier"
SRCS = ["budget_out", "frontier_out", "scale_out", "maze_out", "converge_out"]
ABL = os.path.join(BASE, "ablation"); os.makedirs(ABL, exist_ok=True)
IDLE_W = 4.7
TAU_TOKEN = [60.0, 65.0]   # %
TAU_EXACT = [5.0, 10.0]    # %


def parse_ts(s):
    return dt.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def load_power(pwcsv):
    ts, pw = [], []
    for ln in open(pwcsv):
        p = ln.split(",")
        if len(p) < 2:
            continue
        try:
            ts.append(parse_ts(p[0])); pw.append(float(p[1]))
        except Exception:
            continue
    return ts, pw


def energy_until(ts, pw, t_end):
    """Wh net (idle-corrected) dari ts[0] sampai t_end, trapezoid."""
    if len(ts) < 2:
        return None
    net = 0.0
    for i in range(1, len(ts)):
        if ts[i] > t_end:
            break
        d = ts[i] - ts[i - 1]
        if d <= 0 or d > 30:
            continue
        net += 0.5 * ((pw[i] - IDLE_W) + (pw[i - 1] - IDLE_W)) * d
    return net / 3600.0  # Wh


def evals(pj):
    """list (step, t, exact%, token%) urut step."""
    out = []
    for ln in open(pj):
        if '"phase": "eval"' not in ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        out.append((r.get("step"), r.get("t"),
                    r.get("all/exact_accuracy", 0) * 100, r.get("all/accuracy", 0) * 100))
    out.sort(key=lambda x: (x[0] is None, x[0]))
    return out


def first_reach(evs, idx, tau):
    """(step, t) checkpoint pertama dgn metrik[idx]>=tau, else None. idx: 2=exact,3=token."""
    for e in evs:
        if e[idx] is not None and e[idx] >= tau:
            return e[0], e[1]
    return None


def cfg_from_tag(tag):
    h = d = None
    if tag.startswith("h"):
        try: h = int(tag.split("_")[0][1:])
        except Exception: pass
    if "_d" in tag:
        try: d = int(tag.split("_d")[1].split("_")[0])
        except Exception: pass
    return h, d


def main():
    rows = []
    for src in SRCS:
        d = os.path.join(BASE, src)
        for pj in sorted(glob.glob(os.path.join(d, "progress_*.jsonl"))):
            tag = os.path.basename(pj)[len("progress_"):-len(".jsonl")]
            pw = os.path.join(d, f"pw_{tag}.csv")
            if not os.path.exists(pw):
                continue
            ts, pwr = load_power(pw)
            if len(ts) < 2:
                continue
            evs = evals(pj)
            if not evs:
                continue
            h, deff = cfg_from_tag(tag)
            rec = {"source": src, "tag": tag, "hidden": h, "D_eff": deff,
                   "best_exact": max((e[2] for e in evs), default=0),
                   "best_token": max((e[3] for e in evs), default=0)}
            for tau in TAU_TOKEN:
                r = first_reach(evs, 3, tau)
                key = f"Wh_token{int(tau)}"
                if r and r[1]:
                    e = energy_until(ts, pwr, r[1])
                    rec[key] = round(e, 4) if e is not None else None
                    rec[f"step_token{int(tau)}"] = r[0]
                else:
                    rec[key] = None; rec[f"step_token{int(tau)}"] = None
            for tau in TAU_EXACT:
                r = first_reach(evs, 2, tau)
                key = f"Wh_exact{int(tau)}"
                if r and r[1]:
                    e = energy_until(ts, pwr, r[1])
                    rec[key] = round(e, 4) if e is not None else None
                    rec[f"step_exact{int(tau)}"] = r[0]
                else:
                    rec[key] = None; rec[f"step_exact{int(tau)}"] = None
            rows.append(rec)

    # CSV
    fields = ["source", "tag", "hidden", "D_eff", "best_exact", "best_token",
              "Wh_token60", "step_token60", "Wh_token65", "step_token65",
              "Wh_exact5", "step_exact5", "Wh_exact10", "step_exact10"]
    csvp = os.path.join(ABL, "joules_to_target.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in fields})

    # frontier per (hidden,D_eff): rata-rata Wh-to-token65 antar seed (budget_out)
    def agg(src, key):
        g = {}
        for r in rows:
            if r["source"] != src or r.get(key) is None:
                continue
            g.setdefault((r["hidden"], r["D_eff"]), []).append(r[key])
        return {k: (statistics.mean(v), len(v)) for k, v in g.items()}

    bud65 = agg("budget_out", "Wh_token65")
    bud_ex5 = agg("budget_out", "Wh_exact5")

    # report
    L = ["# Joules-to-target-accuracy (metrik utama) + Pareto frontier", "",
         "Energi kumulatif net (idle-corrected) untuk MENCAPAI target akurasi tau, dihitung dari "
         "deret-waktu daya (pw_*.csv) di-join ke checkpoint eval (progress). Checkpoint pertama yang "
         "menembus tau menentukan energi. 'tak tercapai' = config tak pernah lewati tau pada budget ini "
         "(bukti saturasi).", "",
         f"Idle={IDLE_W}W. Sumber: {', '.join(SRCS)} (hanya run berdaya-terukur).", "",
         "## Budget_out (h128, ~30k step, 5 seed): energi rata-rata untuk token-acc >=65%", "",
         "| D_eff | Wh-to-token65 (mean) | n seed capai | Wh-to-exact5 (mean) | n capai |",
         "|---|---|---|---|---|"]
    for deff in sorted({k[1] for k in bud65} | {k[1] for k in bud_ex5}):
        a = bud65.get((128, deff)); b = bud_ex5.get((128, deff))
        astr = f"{a[0]:.2f} ({a[1]}/5)" if a else "tak tercapai"
        bstr = f"{b[0]:.2f} ({b[1]}/5)" if b else "tak tercapai"
        an = a[1] if a else 0; bn = b[1] if b else 0
        L.append(f"| {deff} | {a[0]:.2f} |" .replace("nan","-") if a else f"| {deff} | tak tercapai |")
        L[-1] = f"| {deff} | {astr.split(' (')[0] if a else 'tak tercapai'} | {an}/5 | {bstr.split(' (')[0] if b else 'tak tercapai'} | {bn}/5 |"
    L += ["", "## Interpretasi", ""]
    # cari config termurah utk token65
    reach = [(r["tag"], r["D_eff"], r["Wh_token65"]) for r in rows if r.get("Wh_token65") is not None]
    if reach:
        cheap = min(reach, key=lambda x: x[2])
        L.append(f"- Config TERMURAH mencapai token-acc 65%: **{cheap[0]}** "
                 f"(D={cheap[1]}) pada **{cheap[2]:.2f} Wh**.")
        deep_reach = [r for r in reach if r[1] == 36]
        shallow_reach = [r for r in reach if r[1] == 9]
        if deep_reach and shallow_reach:
            ms = statistics.mean([r[2] for r in shallow_reach])
            md = statistics.mean([r[2] for r in deep_reach])
            L.append(f"- Rata-rata Wh-to-token65: D=9 -> {ms:.2f} Wh vs D=36 -> {md:.2f} Wh "
                     f"(**{md/ms:.1f}x** lebih boros utk target sama). Saturasi energi terukur eksplisit.")
    n_ex10 = sum(1 for r in rows if r.get("Wh_exact10") is not None)
    L.append(f"- Target exact-acc 10% dicapai oleh {n_ex10}/{len(rows)} run "
             f"(akurasi rendah = model under-trained; metrik token lebih informatif pada budget ini).")
    L.append("")
    open(os.path.join(ABL, "joules_to_target_report.md"), "w").write("\n".join(L) + "\n")

    # figur Pareto: x=Wh-to-token65, y=best_token, warna per D_eff (budget_out)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.2, 4.4))
    col = {9: "tab:blue", 18: "tab:orange", 36: "tab:green"}
    for r in rows:
        if r["source"] != "budget_out" or r.get("Wh_token65") is None:
            continue
        plt.scatter(r["Wh_token65"], r["best_token"], c=col.get(r["D_eff"], "gray"),
                    label=f"D={r['D_eff']}", alpha=0.7)
    h_, l_ = plt.gca().get_legend_handles_labels()
    by = dict(zip(l_, h_))
    plt.legend(by.values(), by.keys(), title="recursion depth")
    plt.xlabel("Joules-to-target (Wh, net) — energi capai token-acc 65%")
    plt.ylabel("best token accuracy (%)")
    plt.title("Pareto: energi untuk capai target vs akurasi (h128, 5 seed)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    p = os.path.join(ABL, "fig_joules_to_target.png"); plt.savefig(p, dpi=200); plt.close()

    print("\n".join(L))
    print("\nwrote", csvp, "and report + fig")
    if reach:
        print(f"NOTIF cheapest_token65={cheap[0]} {cheap[2]:.2f}Wh; "
              f"reach_token65={len(reach)}/{len(rows)} reach_exact10={n_ex10}/{len(rows)}")


if __name__ == "__main__":
    main()
