#!/usr/bin/env python3
"""Run konvergensi: 1 config Sudoku dilatih panjang untuk lihat apakah exact-acc
menembus plateau ~15% (budget-resolved depth sebelumnya berhenti ~30k step).

Default h128_d36 (depth terdalam, kandidat terbaik utk exact-acc) ~80k step, 40 checkpoint
-> kurva learning halus utk grafik ablasi. Histori per-step penuh + daya + CodeCarbon.

Env: HIDDEN(128) DEPTH(36) STEPS(80000) NEVAL(40) SEED(0) DATA(data/sudoku-pilot).
"""
import csv, datetime as dt, json, os, signal, subprocess, time

SCR = "/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad"
TRM = os.path.join(SCR, "TRM")
OUT = os.path.join(SCR, "converge_out"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DATA", "data/sudoku-pilot")
HIDDEN = int(os.environ.get("HIDDEN", "128")); DEPTH = int(os.environ.get("DEPTH", "36"))
STEPS = int(os.environ.get("STEPS", "80000")); NEVAL = int(os.environ.get("NEVAL", "40"))
SEED = int(os.environ.get("SEED", "0"))
STEPS_PER_EPOCH = 7.81; IDLE_W = 4.7
EPOCHS = max(NEVAL, round(STEPS / STEPS_PER_EPOCH)); EI = max(1, EPOCHS // NEVAL); EPOCHS = EI * NEVAL


def parse_ts(s):
    return dt.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def integrate_pw(pwcsv):
    ts, pw = [], []
    for ln in open(pwcsv):
        p = ln.split(",")
        if len(p) < 2: continue
        try: ts.append(parse_ts(p[0])); pw.append(float(p[1]))
        except Exception: continue
    if len(ts) < 2: return None
    g = n = 0.0
    for i in range(1, len(ts)):
        d = ts[i] - ts[i - 1]
        if d <= 0 or d > 30: continue
        g += 0.5 * (pw[i] + pw[i - 1]) * d
        n += 0.5 * ((pw[i] - IDLE_W) + (pw[i - 1] - IDLE_W)) * d
    return g / 3600, n / 3600, ts[-1] - ts[0], sum(pw) / len(pw), len(pw)


def cc_gpu_wh(emcsv):
    if not os.path.exists(emcsv): return None
    rows = list(csv.DictReader(open(emcsv)))
    if not rows: return None
    try: return float(rows[-1].get("gpu_energy", "")) * 1000.0
    except Exception: return None


def main():
    h, d, L = HIDDEN, DEPTH, DEPTH // 3
    tag = f"h{h}_d{d}_converge_s{SEED}"
    pj = os.path.join(OUT, f"progress_{tag}.jsonl")
    if os.path.exists(pj) and sum(1 for l in open(pj) if '"phase": "eval"' in l) >= NEVAL:
        print(f"[{tag}] SKIP complete", flush=True); print("CONVERGE RUN DONE", flush=True); return
    open(pj, "w").close()
    pw = os.path.join(OUT, f"pw_{tag}.csv"); logf = os.path.join(OUT, f"log_{tag}.txt")
    emcsv = os.path.join(OUT, f"emissions_{tag}.csv")
    if os.path.exists(emcsv): os.remove(emcsv)
    print(f"CONVERGE [{tag}] epochs={EPOCHS} (~{int(EPOCHS*STEPS_PER_EPOCH)} steps) ei={EI}", flush=True)
    env = dict(os.environ, WANDB_MODE="disabled", WANDB_SILENT="true", HYDRA_FULL_ERROR="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TRM_PROGRESS_LOG=pj)
    from codecarbon import OfflineEmissionsTracker
    tracker = OfflineEmissionsTracker(country_iso_code="IDN", measure_power_secs=5, log_level="error",
                                      save_to_file=True, output_dir=OUT, output_file=os.path.basename(emcsv),
                                      tracking_mode="machine", project_name=tag)
    pwf = open(pw, "w")
    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-lms", "1000"], stdout=pwf, preexec_fn=os.setsid)
    t0 = time.time(); tracker.start()
    cmd = ["python", "pretrain.py", "arch=trm", f"data_paths=[{DATA}]", "evaluators=[]",
           f"epochs={EPOCHS}", f"eval_interval={EI}", f"seed={SEED}", "lr=1e-4", "puzzle_emb_lr=1e-4",
           "weight_decay=1.0", "puzzle_emb_weight_decay=1.0", "arch.mlp_t=True", "arch.pos_encodings=none",
           "arch.L_layers=2", "arch.H_cycles=3", f"arch.L_cycles={L}", f"arch.hidden_size={h}",
           "global_batch_size=128", f"+run_name={tag}", "ema=False"]
    try:
        with open(logf, "w") as lf:
            subprocess.run(cmd, cwd=TRM, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=28800)
    finally:
        try: tracker.stop()
        except Exception: pass
        try: os.killpg(os.getpgid(sampler.pid), signal.SIGTERM)
        except Exception: pass
        try: pwf.close()
        except Exception: pass
    wall = time.time() - t0
    best_ex = 0.0; last = None
    for ln in open(pj):
        if '"phase": "eval"' not in ln: continue
        try: r = json.loads(ln)
        except Exception: continue
        best_ex = max(best_ex, r.get("all/exact_accuracy", 0) * 100); last = r
    smi = integrate_pw(pw); cc = cc_gpu_wh(emcsv)
    sp = os.path.join(OUT, "converge_summary.csv")
    fields = ["tag", "hidden", "D_eff", "seed", "steps_target", "wall_s", "best_exact_pct",
              "final_exact_pct", "final_token_pct", "smi_net_Wh", "cc_gpu_Wh", "gross_agree_pct"]
    rec = {"tag": tag, "hidden": h, "D_eff": d, "seed": SEED, "steps_target": STEPS,
           "wall_s": round(wall, 1), "best_exact_pct": round(best_ex, 4),
           "final_exact_pct": round((last or {}).get("all/exact_accuracy", 0) * 100, 4),
           "final_token_pct": round((last or {}).get("all/accuracy", 0) * 100, 4),
           "smi_net_Wh": round(smi[1], 4) if smi else None,
           "cc_gpu_Wh": round(cc, 4) if cc else None,
           "gross_agree_pct": (round(100 * (1 - abs(smi[0] - cc) / ((smi[0] + cc) / 2)), 2)
                               if (smi and cc) else None)}
    with open(sp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerow(rec)
    print(f"CONVERGE [{tag}] done wall={wall:.0f}s best_exact={best_ex:.1f}% "
          f"final_exact={rec['final_exact_pct']}% net={rec['smi_net_Wh']}Wh agree={rec['gross_agree_pct']}%", flush=True)
    print("CONVERGE RUN DONE", flush=True)


if __name__ == "__main__":
    main()
