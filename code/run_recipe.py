#!/usr/bin/env python3
"""Smoke resep-setia TRM: uji apakah pipeline bisa keluar dari plateau ~18% bila memakai
resep kanonik `pretrain_mlp_t_sudoku` (config/arch/trm.yaml + README), bukan h128 mungil.

Resep setia: arch=trm hidden=512, H_cycles=3 L_cycles=6 (D_eff=18), mlp_t, pos none, L_layers=2,
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0, EMA=True, ACT halting
default (halt_max_steps=16). Data: sudoku-aug1k (~1M contoh). Batch diset agar muat 16GB.

Emit per run (WAJIB, CLAUDE.md): progress_<tag>.jsonl, pw_<tag>.csv, emissions_<tag>.csv + summary.
Env: HIDDEN(512) DEPTH(18) STEPS(25000) BATCH(192) NEVAL(25) EMA(True) DATA(data/sudoku-aug1k) SEED(0).
"""
import csv, datetime as dt, json, os, signal, subprocess, time

# Durable: TRM env di luar /tmp; artefak ditulis LANGSUNG ke repo (anti-insiden fase L).
TRM = os.environ.get("TRM_DIR", "/home/adb/awangga/trm-env/TRM")
OUT = os.environ.get("OUT_DIR", "/home/adb/awangga/trm/eksperimen/frontier/recipe_out")
os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DATA", "data/sudoku-aug1k")
HIDDEN = int(os.environ.get("HIDDEN", "512")); DEPTH = int(os.environ.get("DEPTH", "18"))
STEPS = int(os.environ.get("STEPS", "25000")); NEVAL = int(os.environ.get("NEVAL", "25"))
BATCH = int(os.environ.get("BATCH", "192")); SEED = int(os.environ.get("SEED", "0"))
EMA = os.environ.get("EMA", "True"); GROUPS = int(os.environ.get("GROUPS", "1000")); IDLE_W = 4.7
ARCH = os.environ.get("ARCH", "trm")   # 'trm' (recursive) atau 'transformers_baseline' (non-recursive)
SPE = max(1.0, GROUPS / BATCH)                       # steps per epoch ~ groups/batch
EPOCHS = max(NEVAL, round(STEPS / SPE)); EI = max(1, EPOCHS // NEVAL); EPOCHS = EI * NEVAL


def parse_ts(s): return dt.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


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
        d = ts[i] - ts[i-1]
        if d <= 0 or d > 30: continue
        g += 0.5*(pw[i]+pw[i-1])*d; n += 0.5*((pw[i]-IDLE_W)+(pw[i-1]-IDLE_W))*d
    return g/3600, n/3600, ts[-1]-ts[0], sum(pw)/len(pw), len(pw)


def cc_gpu_wh(emcsv):
    if not os.path.exists(emcsv): return None
    rows = list(csv.DictReader(open(emcsv)))
    if not rows: return None
    try: return float(rows[-1].get("gpu_energy", ""))*1000.0
    except Exception: return None


def main():
    L = DEPTH // 3
    tag = (f"h{HIDDEN}_d{DEPTH}_recipe_b{BATCH}_s{SEED}" if ARCH == "trm"
           else f"h{HIDDEN}_{ARCH}_b{BATCH}_s{SEED}")
    pj = os.path.join(OUT, f"progress_{tag}.jsonl"); open(pj, "w").close()
    pw = os.path.join(OUT, f"pw_{tag}.csv"); logf = os.path.join(OUT, f"log_{tag}.txt")
    emcsv = os.path.join(OUT, f"emissions_{tag}.csv")
    if os.path.exists(emcsv): os.remove(emcsv)
    print(f"RECIPE [{tag}] hidden={HIDDEN} D={DEPTH} batch={BATCH} ema={EMA} "
          f"epochs={EPOCHS} (~{int(EPOCHS*SPE)} steps) ei={EI} data={DATA}", flush=True)
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
    if ARCH == "trm":
        cmd = ["python", "pretrain.py", "arch=trm", f"data_paths=[{DATA}]", "evaluators=[]",
               f"epochs={EPOCHS}", f"eval_interval={EI}", f"seed={SEED}", "lr=1e-4", "puzzle_emb_lr=1e-4",
               "weight_decay=1.0", "puzzle_emb_weight_decay=1.0", "arch.mlp_t=True", "arch.pos_encodings=none",
               "arch.L_layers=2", "arch.H_cycles=3", f"arch.L_cycles={L}", f"arch.hidden_size={HIDDEN}",
               f"global_batch_size={BATCH}", f"+run_name={tag}", f"ema={EMA}"]
    else:
        # baseline non-recursive (arch=transformers_baseline): tanpa knob rekursi trm-spesifik.
        # Pakai pos_encodings default arch (rope): baseline tak mendukung 'none' (NotImplementedError).
        cmd = ["python", "pretrain.py", f"arch={ARCH}", f"data_paths=[{DATA}]", "evaluators=[]",
               f"epochs={EPOCHS}", f"eval_interval={EI}", f"seed={SEED}", "lr=1e-4", "puzzle_emb_lr=1e-4",
               "weight_decay=1.0", "puzzle_emb_weight_decay=1.0",
               f"arch.hidden_size={HIDDEN}", f"global_batch_size={BATCH}", f"+run_name={tag}", f"ema={EMA}"]
    try:
        with open(logf, "w") as lf:
            subprocess.run(cmd, cwd=TRM, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=43200)
    finally:
        try: tracker.stop()
        except Exception: pass
        try: os.killpg(os.getpgid(sampler.pid), signal.SIGTERM)
        except Exception: pass
        try: pwf.close()
        except Exception: pass
    wall = time.time() - t0
    best = 0.0; last = None
    for ln in open(pj):
        if '"phase": "eval"' not in ln: continue
        try: r = json.loads(ln)
        except Exception: continue
        best = max(best, r.get("all/exact_accuracy", 0)*100); last = r
    smi = integrate_pw(pw); cc = cc_gpu_wh(emcsv)
    sp = os.path.join(OUT, "recipe_summary.csv")
    fields = ["tag","hidden","D_eff","batch","ema","steps_target","wall_s","best_exact_pct",
              "final_exact_pct","final_token_pct","smi_net_Wh","cc_gpu_Wh","gross_agree_pct"]
    rec = {"tag":tag,"hidden":HIDDEN,"D_eff":DEPTH,"batch":BATCH,"ema":EMA,"steps_target":STEPS,
           "wall_s":round(wall,1),"best_exact_pct":round(best,4),
           "final_exact_pct":round((last or {}).get("all/exact_accuracy",0)*100,4),
           "final_token_pct":round((last or {}).get("all/accuracy",0)*100,4),
           "smi_net_Wh":round(smi[1],4) if smi else None,
           "cc_gpu_Wh":round(cc,4) if cc else None,
           "gross_agree_pct":(round(100*(1-abs(smi[0]-cc)/((smi[0]+cc)/2)),2) if (smi and cc) else None)}
    exist = os.path.exists(sp)
    with open(sp,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if not exist: w.writeheader()
        w.writerow(rec)
    print(f"RECIPE [{tag}] done wall={wall:.0f}s best_exact={best:.1f}% "
          f"final_exact={rec['final_exact_pct']}% net={rec['smi_net_Wh']}Wh agree={rec['gross_agree_pct']}%", flush=True)
    print("RECIPE RUN DONE", flush=True)


if __name__ == "__main__":
    main()
