#!/usr/bin/env python3
"""Iso-FLOP 2D frontier runner (P x D). Holds training compute ~constant across configs by setting
per-config steps inversely to per-step cost (FLOP/step ~ params x D_eff ~ measured ms/step) -> iso-wall.
Robust: skips configs already complete (>=5 eval checkpoints); kills the power sampler via process group;
never loses a config on a sampler glitch. Per-progress history via patched pretrain.py.

Usage: python run_isoflop.py        Env: TARGET_WALL_S (default 420), DATA (default data/sudoku-pilot)."""
import csv, json, os, signal, statistics, subprocess, time

TRM = "/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad/TRM"
SCR = "/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad"
MB  = "/home/adb/awangga/trm/eksperimen/kalibrasi/microbench_results.csv"
OUT = os.path.join(SCR, "isoflop_out"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DATA", "data/sudoku-pilot")
TARGET = float(os.environ.get("TARGET_WALL_S", "420"))
STEPS_PER_EPOCH = 7.81
GRID = [(h, d) for h in (128, 256, 384) for d in (9, 18, 36)]

ms = {(int(r["hidden"]), int(r["D_eff"])): float(r["ms_per_step"]) for r in csv.DictReader(open(MB))}
pm = {(int(r["hidden"]), int(r["D_eff"])): float(r["params_M"]) for r in csv.DictReader(open(MB))}

def idle_power():
    try:
        o = subprocess.check_output(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"]).decode()
        return float(o.strip().split("\n")[0])
    except Exception: return 0.0

def complete(tag):
    pj = os.path.join(OUT, f"progress_{tag}.jsonl")
    if not os.path.exists(pj): return False
    return sum(1 for l in open(pj) if '"phase": "eval"' in l) >= 5

def energy_from(pw, wall, idle):
    P=[]
    for ln in open(pw):
        c=ln.split(",")
        if len(c)>=2:
            try: P.append(float(c[1]))
            except: pass
    if not P: return float("nan"), float("nan")
    avg=statistics.mean(P); return avg, max(0.0,(avg-idle))*wall/3600

def final_eval(logf):
    em="{}"
    for ln in open(logf):
        if "EVAL_METRICS_JSON" in ln: em=ln[ln.find("{"):ln.rfind("}")+1] or "{}"
    try: json.loads(em); return em
    except: return "{}"

def run_cfg(h, d, idle):
    tag=f"h{h}_d{d}"; L=d//3
    if complete(tag): print(f"[{tag}] SKIP (already complete)", flush=True); return
    mss=ms[(h,d)]; steps=TARGET/(mss/1000.0); epochs=max(5,round(steps/STEPS_PER_EPOCH))
    ei=max(1,epochs//5); epochs=ei*5
    pj=os.path.join(OUT,f"progress_{tag}.jsonl"); open(pj,"w").close()
    pw=os.path.join(OUT,f"pw_{tag}.csv"); logf=os.path.join(OUT,f"log_{tag}.txt")
    print(f"[{tag}] ms/step={mss} epochs={epochs} (~{int(epochs*STEPS_PER_EPOCH)} steps) ei={ei}", flush=True)
    env=dict(os.environ, WANDB_MODE="disabled", WANDB_SILENT="true", HYDRA_FULL_ERROR="1",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TRM_PROGRESS_LOG=pj)
    pwf=open(pw,"w")
    sampler=subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits","-lms","1000"], stdout=pwf, preexec_fn=os.setsid)
    t0=time.time()
    cmd=["python","pretrain.py","arch=trm",f"data_paths=[{DATA}]","evaluators=[]",
         f"epochs={epochs}",f"eval_interval={ei}","lr=1e-4","puzzle_emb_lr=1e-4","weight_decay=1.0",
         "puzzle_emb_weight_decay=1.0","arch.mlp_t=True","arch.pos_encodings=none","arch.L_layers=2",
         "arch.H_cycles=3",f"arch.L_cycles={L}",f"arch.hidden_size={h}","global_batch_size=128",
         f"+run_name={tag}","ema=False"]
    try:
        with open(logf,"w") as lf: subprocess.run(cmd,cwd=TRM,env=env,stdout=lf,stderr=subprocess.STDOUT,timeout=7200)
    finally:
        try: os.killpg(os.getpgid(sampler.pid), signal.SIGTERM)
        except Exception: pass
        try: pwf.close()
        except Exception: pass
    wall=time.time()-t0
    print(f"[{tag}] training done wall={wall:.0f}s exact_acc={json.loads(final_eval(logf)).get('all/exact_accuracy','?')}", flush=True)

def summarize(idle):
    rows=[]
    for h,d in GRID:
        tag=f"h{h}_d{d}"; pj=os.path.join(OUT,f"progress_{tag}.jsonl")
        pw=os.path.join(OUT,f"pw_{tag}.csv"); logf=os.path.join(OUT,f"log_{tag}.txt")
        if not os.path.exists(pj): continue
        # wall from pw line count (1s sampling) as fallback; else from progress timestamps
        wall=0.0
        try:
            ts=[json.loads(l)["t"] for l in open(pj) if l.strip()]
            if ts: wall=max(ts)-min(ts)
        except Exception: pass
        avg,net=(float("nan"),float("nan"))
        if os.path.exists(pw): avg,net=energy_from(pw,wall,idle)
        em=final_eval(logf) if os.path.exists(logf) else "{}"
        rows.append([h,d,pm.get((h,d),""),pm.get((h,d),0)*d, wall and round(wall,1), round(avg,1) if avg==avg else "",
                     round(net,4) if net==net else "", em, f"progress_{tag}.jsonl", f"pw_{tag}.csv"])
    sp=os.path.join(OUT,"isoflop_summary.csv")
    with open(sp,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["hidden","D_eff","params_M","compute_PxD","wall_s","avg_gpu_W","net_energy_Wh","final_eval_json","progress_jsonl","power_csv"])
        w.writerows(rows)
    print(f"summarized {len(rows)} configs -> {sp}", flush=True)

if __name__=="__main__":
    idle=idle_power(); print(f"ISO-FLOP grid {len(GRID)} cfgs TARGET={TARGET}s idle={idle}W", flush=True)
    for h,d in GRID:
        try: run_cfg(h,d,idle)
        except Exception as e: print(f"[h{h}_d{d}] FAILED: {str(e)[:140]}", flush=True)
    summarize(idle)
    print("ISOFLOP RUN DONE", flush=True)
