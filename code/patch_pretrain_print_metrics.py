#!/usr/bin/env python3
"""Idempotent patch: make TRM pretrain.py record PER-PROGRESS history.
Injects two things (TRM logs only to W&B by default):
  (A) train loop: append per-step train metrics to a JSONL (phase=train) via $TRM_PROGRESS_LOG;
  (B) eval block: print "EVAL_METRICS_JSON step=N {...}" and append per-checkpoint eval metrics
      (phase=eval) to the same JSONL.
Gives learning curves (loss/step), accuracy/loss per checkpoint — material for plots & ablation.
Usage: python patch_pretrain_print_metrics.py <path/to/pretrain.py>"""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "pretrain.py"
s = open(p).read()
if "TRM_PROGRESS_LOG" in s:
    print("already patched"); raise SystemExit

# (A) TRAIN loop: anchor includes the progress_bar.update line (distinguishes from eval block)
train_anchor = ("            if RANK == 0 and metrics is not None:\n"
                "                wandb.log(metrics, step=train_state.step)\n"
                "                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore")
train_new = ("            if RANK == 0 and metrics is not None:\n"
             "                import json as _json, time as _time, os as _os\n"
             "                _flat={}\n"
             "                for _k,_v in metrics.items():\n"
             "                    if hasattr(_v,\"items\"):\n"
             "                        for _kk,_vv in _v.items(): _flat[f\"{_k}/{_kk}\"]=float(_vv)\n"
             "                    else:\n"
             "                        try: _flat[_k]=float(_v)\n"
             "                        except: pass\n"
             "                _pl=_os.environ.get(\"TRM_PROGRESS_LOG\")\n"
             "                if _pl:\n"
             "                    with open(_pl,\"a\") as _pf: _pf.write(_json.dumps({\"phase\":\"train\",\"step\":int(train_state.step),\"t\":_time.time(),**_flat})+\"\\n\")\n"
             "                wandb.log(metrics, step=train_state.step)\n"
             "                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore")

# (B) EVAL block: bare anchor (no progress_bar line)
eval_anchor = ("            if RANK == 0 and metrics is not None:\n"
               "                wandb.log(metrics, step=train_state.step)")
eval_new = ("            if RANK == 0 and metrics is not None:\n"
            "                import json as _json2, time as _time2, os as _os2\n"
            "                _ef={}\n"
            "                for _a,_b in metrics.items():\n"
            "                    if hasattr(_b,\"items\"):\n"
            "                        for _bb,_cc in _b.items(): _ef[f\"{_a}/{_bb}\"]=float(_cc)\n"
            "                    else:\n"
            "                        try: _ef[_a]=float(_b)\n"
            "                        except: pass\n"
            "                print(\"EVAL_METRICS_JSON step=\"+str(int(train_state.step)), _json2.dumps(_ef), flush=True)\n"
            "                _pl2=_os2.environ.get(\"TRM_PROGRESS_LOG\")\n"
            "                if _pl2:\n"
            "                    with open(_pl2,\"a\") as _ef2: _ef2.write(_json2.dumps({\"phase\":\"eval\",\"step\":int(train_state.step),\"t\":_time2.time(),**_ef})+\"\\n\")\n"
            "                wandb.log(metrics, step=train_state.step)")

if train_anchor not in s or eval_anchor not in s:
    print("ERROR: expected anchors not found — pretrain.py version differs"); raise SystemExit(1)
s = s.replace(train_anchor, train_new, 1)   # train first (unique due to progress_bar line)
s = s.replace(eval_anchor, eval_new, 1)     # then the remaining bare eval block
open(p, "w").write(s)
print("patched (train per-step + eval per-checkpoint history):", p)
