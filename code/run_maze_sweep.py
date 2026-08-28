#!/usr/bin/env python3
"""Depth-sweep Maze (task kedua, generalitas lintas-task) dengan replikasi seed.

Menjawab celah 'hanya 1 task': mengulang protokol budget-resolved Sudoku pada
Maze-30x30-hard. Untuk tiap (depth, seed): histori per-step penuh (progress jsonl),
deret-waktu daya (pw csv), CodeCarbon (emissions) pada window training yang sama,
+ baris summary self-describing dengan Joules-to-target & energi cross-val.

Maze ~8.5x lebih lambat dari Sudoku (seq_len 900 vs 81), jadi STEPS lebih kecil (8000).
Robust: skip-if-complete, kill sampler via process-group.

Env: SEEDS ("0 1 2"), STEPS (8000), NEVAL (16), HIDDEN (128), DATA (data/maze-30x30-hard-1k).
"""
import csv, datetime as dt, json, os, signal, subprocess, time

SCR = "/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad"
TRM = os.path.join(SCR, "TRM")
OUT = os.path.join(SCR, "maze_out"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DATA", "data/maze-30x30-hard-1k")
STEPS = int(os.environ.get("STEPS", "8000")); NEVAL = int(os.environ.get("NEVAL", "16"))
HIDDEN = int(os.environ.get("HIDDEN", "128")); DEPTHS = [9, 18, 36]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2").split()]
STEPS_PER_EPOCH = 7.81
IDLE_W = 4.7
EPOCHS = max(NEVAL, round(STEPS / STEPS_PER_EPOCH)); EI = max(1, EPOCHS // NEVAL); EPOCHS = EI * NEVAL


def parse_ts(s):
    return dt.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def integrate_pw(pwcsv):
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


def cc_gpu_wh(emcsv):
    if not os.path.exists(emcsv):
        return None
    rows = list(csv.DictReader(open(emcsv)))
    if not rows:
        return None
    try:
        return float(rows[-1].get("gpu_energy", "")) * 1000.0
    except Exception:
        return None


def complete(tag):
    pj = os.path.join(OUT, f"progress_{tag}.jsonl")
    return os.path.exists(pj) and sum(1 for l in open(pj) if '"phase": "eval"' in l) >= NEVAL


def last_eval(pj):
    last = None
    for ln in open(pj):
        if '"phase": "eval"' not in ln:
            continue
        try:
            last = json.loads(ln)
        except Exception:
            pass
    return last


def run(h, d, seed):
    tag = f"h{h}_d{d}_s{seed}"; L = d // 3
    if complete(tag):
        print(f"[{tag}] SKIP complete", flush=True); return None
    pj = os.path.join(OUT, f"progress_{tag}.jsonl"); open(pj, "w").close()
    pw = os.path.join(OUT, f"pw_{tag}.csv"); logf = os.path.join(OUT, f"log_{tag}.txt")
    emcsv = os.path.join(OUT, f"emissions_{tag}.csv")
    if os.path.exists(emcsv):
        os.remove(emcsv)
    print(f"[{tag}] epochs={EPOCHS} (~{int(EPOCHS*STEPS_PER_EPOCH)} steps) ei={EI} seed={seed}", flush=True)
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
    t0 = time.time()
    tracker.start()
    cmd = ["python", "pretrain.py", "arch=trm", f"data_paths=[{DATA}]", "evaluators=[]",
           f"epochs={EPOCHS}", f"eval_interval={EI}", f"seed={seed}", "lr=1e-4", "puzzle_emb_lr=1e-4",
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
    le = last_eval(pj) or {}
    smi = integrate_pw(pw); cc = cc_gpu_wh(emcsv)
    rec = {"tag": tag, "task": "maze", "hidden": h, "D_eff": d, "seed": seed, "wall_s": round(wall, 1),
           "exact_pct": round(le.get("all/exact_accuracy", 0) * 100, 4),
           "token_pct": round(le.get("all/accuracy", 0) * 100, 4),
           "lm_loss": le.get("all/lm_loss"),
           "smi_gross_Wh": round(smi[0], 4) if smi else None,
           "smi_net_Wh": round(smi[1], 4) if smi else None,
           "mean_W": round(smi[3], 2) if smi else None,
           "cc_gpu_Wh": round(cc, 4) if cc else None,
           "gross_agree_pct": (round(100 * (1 - abs(smi[0] - cc) / ((smi[0] + cc) / 2)), 2)
                               if (smi and cc) else None)}
    print(f"[{tag}] done wall={wall:.0f}s exact={rec['exact_pct']} token={rec['token_pct']} "
          f"net={rec['smi_net_Wh']}Wh agree={rec['gross_agree_pct']}%", flush=True)
    return rec


def main():
    print(f"MAZE sweep hidden={HIDDEN} depths={DEPTHS} seeds={SEEDS} STEPS={STEPS} (~{EPOCHS*STEPS_PER_EPOCH:.0f})", flush=True)
    recs = []
    for seed in SEEDS:
        for d in DEPTHS:
            try:
                r = run(HIDDEN, d, seed)
                if r: recs.append(r)
            except Exception as e:
                print(f"[h{HIDDEN}_d{d}_s{seed}] FAILED: {str(e)[:160]}", flush=True)
    fields = ["tag", "task", "hidden", "D_eff", "seed", "wall_s", "exact_pct", "token_pct",
              "lm_loss", "smi_gross_Wh", "smi_net_Wh", "mean_W", "cc_gpu_Wh", "gross_agree_pct"]
    sp = os.path.join(OUT, "maze_summary.csv")
    exist = os.path.exists(sp)
    with open(sp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exist: w.writeheader()
        for r in recs: w.writerow({k: r.get(k) for k in fields})
    print(f"MAZE RUN DONE n_new={len(recs)} -> {sp}", flush=True)


if __name__ == "__main__":
    main()
