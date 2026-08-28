#!/usr/bin/env python3
"""Replication for CI: re-run the budget sweep (fixed width, depths) across seeds.
Tags h{H}_d{d}_s{seed}; skip-if-complete; robust sampler. Env: SEEDS (default "1 2"),
EPOCHS (3840), NEVAL (15), HIDDEN (128), DATA (data/sudoku-pilot)."""
import json, os, signal, subprocess, time

TRM = "/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad/TRM"
SCR = "/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad"
OUT = os.path.join(SCR, "budget_out"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DATA", "data/sudoku-pilot")
EPOCHS = int(os.environ.get("EPOCHS", "3840")); NEVAL = int(os.environ.get("NEVAL", "15"))
HIDDEN = int(os.environ.get("HIDDEN", "128")); DEPTHS = [9, 18, 36]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1 2").split()]
EI = max(1, EPOCHS // NEVAL); EPOCHS = EI * NEVAL

def complete(tag):
    pj = os.path.join(OUT, f"progress_{tag}.jsonl")
    return os.path.exists(pj) and sum(1 for l in open(pj) if '"phase": "eval"' in l) >= NEVAL

def run(h, d, seed):
    tag=f"h{h}_d{d}_s{seed}"; L=d//3
    if complete(tag): print(f"[{tag}] SKIP", flush=True); return
    pj=os.path.join(OUT,f"progress_{tag}.jsonl"); open(pj,"w").close()
    pw=os.path.join(OUT,f"pw_{tag}.csv"); logf=os.path.join(OUT,f"log_{tag}.txt")
    print(f"[{tag}] epochs={EPOCHS} seed={seed}", flush=True)
    env=dict(os.environ, WANDB_MODE="disabled", WANDB_SILENT="true", HYDRA_FULL_ERROR="1",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TRM_PROGRESS_LOG=pj)
    pwf=open(pw,"w")
    s=subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits","-lms","1000"], stdout=pwf, preexec_fn=os.setsid)
    t0=time.time()
    cmd=["python","pretrain.py","arch=trm",f"data_paths=[{DATA}]","evaluators=[]",
         f"epochs={EPOCHS}",f"eval_interval={EI}",f"seed={seed}","lr=1e-4","puzzle_emb_lr=1e-4",
         "weight_decay=1.0","puzzle_emb_weight_decay=1.0","arch.mlp_t=True","arch.pos_encodings=none",
         "arch.L_layers=2","arch.H_cycles=3",f"arch.L_cycles={L}",f"arch.hidden_size={h}",
         "global_batch_size=128",f"+run_name={tag}","ema=False"]
    try:
        with open(logf,"w") as lf: subprocess.run(cmd,cwd=TRM,env=env,stdout=lf,stderr=subprocess.STDOUT,timeout=21600)
    finally:
        try: os.killpg(os.getpgid(s.pid), signal.SIGTERM)
        except Exception: pass
        try: pwf.close()
        except Exception: pass
    em="{}"
    for ln in open(logf):
        if "EVAL_METRICS_JSON" in ln: em=ln[ln.find("{"):ln.rfind("}")+1] or "{}"
    try: acc=json.loads(em).get("all/exact_accuracy","?")
    except: acc="?"
    print(f"[{tag}] done wall={time.time()-t0:.0f}s exact_acc={acc}", flush=True)

if __name__=="__main__":
    print(f"REPLICATE hidden={HIDDEN} depths={DEPTHS} seeds={SEEDS}", flush=True)
    for seed in SEEDS:
        for d in DEPTHS:
            try: run(HIDDEN,d,seed)
            except Exception as e: print(f"[h{HIDDEN}_d{d}_s{seed}] FAILED: {str(e)[:140]}", flush=True)
    print("REPLICATE RUN DONE", flush=True)
