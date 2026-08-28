#!/usr/bin/env python3
"""Seed tambahan + cross-validasi energi (CodeCarbon vs nvidia-smi) pada run IDENTIK.

Untuk tiap (depth, seed) di SEEDS x DEPTHS:
  * Sampler nvidia-smi 1 Hz  -> pw_h{H}_d{d}_s{seed}.csv  (timestamp,power,util,mem)
  * CodeCarbon OfflineEmissionsTracker membungkus window training yang sama
    -> gpu_energy (kWh) di emissions_h{H}_d{d}_s{seed}.csv
  * pretrain.py (config sama persis dgn run_replicate.py)
Lalu hitung energi nvidia-smi (trapezoid integral daya x dt) dan bandingkan ke
CodeCarbon gpu_energy pada window yang sama -> energy_xval.csv (gross + net + %agree).

Kenapa run yang sama: dua pembukuan energi atas window fisik identik = cross-validasi
yang adil (aturan integritas #3, target >95% setuju).

Env: SEEDS (default "3 4"), EPOCHS (3840), NEVAL (15), HIDDEN (128), DATA (data/sudoku-pilot).
"""
import csv, datetime as dt, json, os, signal, subprocess, sys, time

SCR = "/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad"
TRM = os.path.join(SCR, "TRM")
OUT = os.path.join(SCR, "budget_out"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DATA", "data/sudoku-pilot")
EPOCHS = int(os.environ.get("EPOCHS", "3840")); NEVAL = int(os.environ.get("NEVAL", "15"))
HIDDEN = int(os.environ.get("HIDDEN", "128")); DEPTHS = [9, 18, 36]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "3 4").split()]
EI = max(1, EPOCHS // NEVAL); EPOCHS = EI * NEVAL
IDLE_W = 4.7  # diukur pada GPU idle (RTX 5060 Ti), konsisten dgn run sebelumnya


def complete(tag):
    pj = os.path.join(OUT, f"progress_{tag}.jsonl")
    return os.path.exists(pj) and sum(1 for l in open(pj) if '"phase": "eval"' in l) >= NEVAL


def parse_ts(s):
    # "2026/06/27 06:30:31.101"
    return dt.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def integrate_power(pwcsv):
    """Trapezoid integral daya(W) atas waktu -> (gross_Wh, net_Wh, dur_s, mean_W, n)."""
    ts, pw = [], []
    for ln in open(pwcsv):
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        try:
            ts.append(parse_ts(parts[0])); pw.append(float(parts[1]))
        except Exception:
            continue
    if len(ts) < 2:
        return None
    gross_j = net_j = 0.0
    for i in range(1, len(ts)):
        dtv = ts[i] - ts[i - 1]
        if dtv <= 0 or dtv > 30:  # buang gap besar
            continue
        gross_j += 0.5 * (pw[i] + pw[i - 1]) * dtv
        net_j += 0.5 * ((pw[i] - IDLE_W) + (pw[i - 1] - IDLE_W)) * dtv
    dur = ts[-1] - ts[0]
    return gross_j / 3600.0, net_j / 3600.0, dur, sum(pw) / len(pw), len(pw)


def cc_gpu_wh(emcsv):
    """gpu_energy (kWh->Wh) + energy_consumed (Wh) dari emissions csv CodeCarbon (baris terakhir)."""
    if not os.path.exists(emcsv):
        return None
    rows = list(csv.DictReader(open(emcsv)))
    if not rows:
        return None
    r = rows[-1]
    def f(k):
        try: return float(r.get(k, "")) * 1000.0  # kWh -> Wh
        except Exception: return None
    return f("gpu_energy"), f("energy_consumed"), f("cpu_energy"), f("ram_energy")


def run(h, d, seed):
    tag = f"h{h}_d{d}_s{seed}"; L = d // 3
    if complete(tag):
        print(f"[{tag}] SKIP complete", flush=True); return None
    pj = os.path.join(OUT, f"progress_{tag}.jsonl"); open(pj, "w").close()
    pw = os.path.join(OUT, f"pw_{tag}.csv"); logf = os.path.join(OUT, f"log_{tag}.txt")
    emcsv = os.path.join(OUT, f"emissions_{tag}.csv")
    if os.path.exists(emcsv):
        os.remove(emcsv)
    print(f"[{tag}] epochs={EPOCHS} ei={EI} seed={seed}", flush=True)
    env = dict(os.environ, WANDB_MODE="disabled", WANDB_SILENT="true", HYDRA_FULL_ERROR="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TRM_PROGRESS_LOG=pj)

    # CodeCarbon offline tracker (no network); ukur window training ini
    from codecarbon import OfflineEmissionsTracker
    tracker = OfflineEmissionsTracker(
        country_iso_code="IDN", measure_power_secs=5, log_level="error",
        save_to_file=True, output_dir=OUT, output_file=os.path.basename(emcsv),
        tracking_mode="machine", project_name=tag)

    pwf = open(pw, "w")
    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-lms", "1000"], stdout=pwf, preexec_fn=os.setsid)
    t0 = time.time()
    tracker.start()
    cmd = ["python", "pretrain.py", "arch=trm", f"data_paths=[{DATA}]", "evaluators=[]",
           f"epochs={EPOCHS}", f"eval_interval={EI}", f"seed={seed}", "lr=1e-4", "puzzle_emb_lr=1e-4",
           "weight_decay=1.0", "puzzle_emb_weight_decay=1.0", "arch.mlp_t=True", "arch.pos_encodings=none",
           "arch.L_layers=2", "arch.H_cycles=3", f"arch.L_cycles={L}", f"arch.hidden_size={h}",
           "global_batch_size=128", f"+run_name={tag}", "ema=False"]
    try:
        with open(logf, "w") as lf:
            subprocess.run(cmd, cwd=TRM, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=21600)
    finally:
        try: tracker.stop()
        except Exception as e: print(f"[{tag}] cc.stop err {e}", flush=True)
        try: os.killpg(os.getpgid(sampler.pid), signal.SIGTERM)
        except Exception: pass
        try: pwf.close()
        except Exception: pass
    wall = time.time() - t0

    em = "{}"
    for ln in open(logf):
        if "EVAL_METRICS_JSON" in ln:
            em = ln[ln.find("{"):ln.rfind("}") + 1] or "{}"
    try: acc = json.loads(em).get("all/exact_accuracy", None)
    except Exception: acc = None

    smi = integrate_power(pw)
    cc = cc_gpu_wh(emcsv)
    rec = {"tag": tag, "hidden": h, "D_eff": d, "seed": seed, "wall_s": round(wall, 1),
           "exact_acc": acc}
    if smi:
        rec.update({"smi_gross_Wh": round(smi[0], 4), "smi_net_Wh": round(smi[1], 4),
                    "smi_dur_s": round(smi[2], 1), "smi_mean_W": round(smi[3], 2), "smi_n": smi[4]})
    if cc:
        rec["cc_gpu_Wh"] = round(cc[0], 4) if cc[0] is not None else None
        rec["cc_total_Wh"] = round(cc[1], 4) if cc[1] is not None else None
    if smi and cc and cc[0]:
        agree = 100.0 * (1 - abs(smi[0] - cc[0]) / ((smi[0] + cc[0]) / 2))
        rec["gross_agree_pct"] = round(agree, 2)
    print(f"[{tag}] done wall={wall:.0f}s exact={acc} "
          f"smi_gross={rec.get('smi_gross_Wh')}Wh cc_gpu={rec.get('cc_gpu_Wh')}Wh "
          f"agree={rec.get('gross_agree_pct')}%", flush=True)
    return rec


def main():
    print(f"XVAL+SEEDS hidden={HIDDEN} depths={DEPTHS} seeds={SEEDS} epochs={EPOCHS}", flush=True)
    recs = []
    for seed in SEEDS:
        for d in DEPTHS:
            try:
                r = run(HIDDEN, d, seed)
                if r: recs.append(r)
            except Exception as e:
                print(f"[h{HIDDEN}_d{d}_s{seed}] FAILED: {str(e)[:160]}", flush=True)
    # tulis/append energy_xval.csv
    xv = os.path.join(OUT, "energy_xval.csv")
    fields = ["tag", "hidden", "D_eff", "seed", "wall_s", "exact_acc", "smi_gross_Wh",
              "smi_net_Wh", "smi_dur_s", "smi_mean_W", "smi_n", "cc_gpu_Wh", "cc_total_Wh",
              "gross_agree_pct"]
    exist = os.path.exists(xv)
    with open(xv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exist: w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k) for k in fields})
    print(f"XVAL RUN DONE n_new={len(recs)} -> {xv}", flush=True)


if __name__ == "__main__":
    main()
