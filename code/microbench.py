"""Micro-benchmark TRM: params + throughput train-step + energi, sweep (hidden, depth).
Mengukur unit per optimizer-step: 1 forward+backward inner (H_cycles*L_cycles aplikasi L_level).
Tanpa data pipeline, tanpa eval. Energi via nvidia-smi power sampler (subprocess)."""
import os, sys, time, csv, subprocess, threading, statistics
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
sys.path.insert(0, "/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad/TRM")
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1 as TRM,
)

DEV = "cuda"
SEQ_LEN = 81          # Sudoku 9x9
VOCAB = 11
BATCH = 128
WARMUP, ITERS = 5, 25

def make_cfg(hidden, L_cycles, H_cycles=3, L_layers=2):
    return dict(
        batch_size=BATCH, seq_len=SEQ_LEN, vocab_size=VOCAB,
        num_puzzle_identifiers=1, puzzle_emb_ndim=hidden, puzzle_emb_len=16,
        H_cycles=H_cycles, L_cycles=L_cycles, H_layers=0, L_layers=L_layers,
        hidden_size=hidden, expansion=4.0, num_heads=8, pos_encodings="none",
        halt_max_steps=16, halt_exploration_prob=0.1,
        forward_dtype="bfloat16", mlp_t=True, no_ACT_continue=True,
    )

class PowerSampler(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.samples=[]; self.run_=True
    def run(self):
        while self.run_:
            try:
                out=subprocess.check_output(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],timeout=2)
                self.samples.append(float(out.decode().strip().split("\n")[0]))
            except Exception: pass
            time.sleep(0.2)
    def stop(self): self.run_=False

def bench(hidden, L_cycles):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = make_cfg(hidden, L_cycles)
    model = TRM(cfg).to(DEV); model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = dict(
        inputs=torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=DEV),
        labels=torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=DEV),
        puzzle_identifiers=torch.zeros(BATCH, dtype=torch.long, device=DEV),
    )
    inner = model.inner
    def step():
        carry = model.inner.empty_carry(BATCH)
        carry = type(carry)(z_H=carry.z_H.to(DEV), z_L=carry.z_L.to(DEV))
        new_carry, output, _ = inner(carry, batch)   # 1 grad segment (H*L L_level apps)
        loss = torch.nn.functional.cross_entropy(
            output.reshape(-1, VOCAB).float(), batch["labels"].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return loss.item()
    for _ in range(WARMUP): step()
    torch.cuda.synchronize()
    ps = PowerSampler(); ps.start(); t0=time.time()
    for _ in range(ITERS): step()
    torch.cuda.synchronize(); dt=time.time()-t0; ps.stop(); ps.join(timeout=1)
    peak_mem = torch.cuda.max_memory_allocated()/1024**2
    sps = ITERS/dt
    avgP = statistics.mean(ps.samples) if ps.samples else float("nan")
    j_per_step = avgP*dt/ITERS
    del model, opt; torch.cuda.empty_cache()
    return dict(hidden=hidden, L_cycles=L_cycles, D_eff=3*L_cycles, params_M=round(n_params/1e6,3),
                steps_per_s=round(sps,2), ms_per_step=round(1000/sps,1),
                avg_power_W=round(avgP,1), J_per_step=round(j_per_step,3), peak_mem_MiB=round(peak_mem))

if __name__ == "__main__":
    print(f"device={torch.cuda.get_device_name(0)} seq_len={SEQ_LEN} batch={BATCH} iters={ITERS}")
    rows=[]
    grid = [(h, lc) for h in (128,256,384,512,768) for lc in (3,6,12)]
    for h, lc in grid:
        try:
            r=bench(h, lc); rows.append(r)
            print(f"hidden={r['hidden']:4d} D_eff={r['D_eff']:3d} | params={r['params_M']:6.3f}M | "
                  f"{r['steps_per_s']:6.2f} step/s ({r['ms_per_step']:6.1f} ms) | "
                  f"{r['avg_power_W']:5.1f}W | {r['J_per_step']:.2f} J/step | mem={r['peak_mem_MiB']}MiB", flush=True)
        except RuntimeError as e:
            print(f"hidden={h} D_eff={3*lc} FAILED: {str(e)[:80]}", flush=True)
            torch.cuda.empty_cache()
    out="/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad/microbench_results.csv"
    if rows:
        with open(out,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("CSV ->", out)
    print("MICROBENCH DONE")
