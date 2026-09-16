#!/usr/bin/env python3
"""Third patch for the vendored TRM tree (phase BM, in response to reviewer requests).

Two changes, both OFF by default so that behaviour without the environment variables is
byte-for-byte the previous behaviour:

1. `TRM_ACCUM=N` (default 1): gradient accumulation. The loader still yields micro-batches, but
   the optimizer steps once every N of them and the loss is scaled by 1/(N*global_batch_size).
   Each micro-pool keeps its own ACT carry, and the carries live side by side, so one optimizer
   step processes N*micro parallel slots, equivalent to a single full batch of N*micro.
   `total_steps` is divided by N so that `train_state.step` still counts OPTIMIZER steps, leaving
   the learning-rate schedule, the evaluation interval and the total budget unchanged in meaning.
   EMA is updated only at an accumulation boundary.

2. `TRM_EVAL_PREDS=<dir>`: writes per-instance token accuracy and an exact flag for every
   evaluation checkpoint to `<dir>/preds_<TRM_RUN_TAG>_step<k>.npz`. This is what allows the
   512-instance evaluation subset to be split into a selection half and a reporting half, which
   is what a reviewer asked for regarding best-checkpoint selection bias. Without the variable
   nothing is written.

Idempotent, and it normalises trailing whitespace in the files it touches so that its anchors
match. Usage: python3 patch_pretrain_accum_preds.py [TRM_DIR]
"""
import os, sys

TRM = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TRM_DIR", "/home/adb/awangga/trm-env/TRM")
PRE = os.path.join(TRM, "pretrain.py")
LOSS = os.path.join(TRM, "models", "losses.py")

# ---------------------------------------------------------------- losses.py
t = "\n".join(l.rstrip() for l in open(LOSS).read().split("\n"))
if "_PER_SEQ_SINK" not in t:
    old = """            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts"""
    assert t.count(old) == 1, "anchor losses.py is_correct"
    new = old + """

            # --- fase BM: sink per-instance untuk analisis held-out (lihat patch 0003)
            if _PER_SEQ_SINK is not None:
                _valid = new_carry.halted & (loss_counts > 0)
                _PER_SEQ_SINK.append((
                    _valid.detach().cpu(),
                    ((is_correct.to(torch.float32) / loss_divisor).sum(-1)).detach().cpu(),
                    (_valid & seq_is_correct).detach().cpu(),
                ))"""
    t = t.replace(old, new)
    anchor = "IGNORE_LABEL_ID = -100"
    assert t.count(anchor) == 1
    t = t.replace(anchor, anchor + "\n\n# fase BM: bila di-set ke list oleh pretrain.evaluate(), tiap panggilan model\n# menambahkan (valid_mask, per_seq_token_acc, per_seq_exact) untuk slot batch ini.\n_PER_SEQ_SINK = None")
    open(LOSS, "w").write(t)
    print("losses.py: sink per-instance ditambahkan")
else:
    print("losses.py: sudah ter-patch")

# ---------------------------------------------------------------- pretrain.py
# Catatan: spasi di ujung baris dinormalkan lebih dulu supaya jangkar cocok apa adanya.
t = "\n".join(l.rstrip() for l in open(PRE).read().split("\n"))
changed = t != open(PRE).read()

if "TRM_ACCUM" not in t:
    # (a) total_steps dihitung dalam LANGKAH OPTIMIZER
    old = "    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)"
    assert t.count(old) == 1, "anchor total_steps"
    t = t.replace(old, "    _accum = max(1, int(os.environ.get(\"TRM_ACCUM\", \"1\")))\n"
                       "    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / (config.global_batch_size * _accum))")

    # (b) train_batch: carry per micro-pool, skala loss, langkah optimizer di batas
    old = """def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()"""
    assert t.count(old) == 1, "anchor train_batch"
    new = """_ACCUM_CARRIES = []


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    # fase BM: TRM_ACCUM=N menjalankan N micro-batch per langkah optimizer. Tiap micro-pool
    # memegang carry ACT sendiri, jadi satu langkah memproses N*micro slot paralel.
    global _ACCUM_CARRIES
    accum = max(1, int(os.environ.get("TRM_ACCUM", "1")))
    micro = getattr(train_state, "_micro_idx", 0)
    is_boundary = (micro + 1) >= accum

    if is_boundary:
        train_state.step += 1
        if train_state.step > train_state.total_steps:  # At most train_total_steps
            return
    elif train_state.step >= train_state.total_steps:
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    if accum > 1:
        if len(_ACCUM_CARRIES) != accum:
            _ACCUM_CARRIES = [None] * accum
        if _ACCUM_CARRIES[micro] is None:
            with torch.device("cuda"):
                _ACCUM_CARRIES[micro] = train_state.model.initial_carry(batch)  # type: ignore
        _ACCUM_CARRIES[micro], loss, metrics, _, _ = train_state.model(
            carry=_ACCUM_CARRIES[micro], batch=batch, return_keys=[])
    else:
        # Init carry if it is None
        if train_state.carry is None:
            with torch.device("cuda"):
                train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

        # Forward
        train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / (global_batch_size * accum)) * loss).backward()"""
    t = t.replace(old, new)

    # (c) optimizer hanya melangkah di batas akumulasi
    old = """    # Apply optimizer
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step

        optim.step()
        optim.zero_grad()"""
    assert t.count(old) == 1, "anchor optimizer"
    new = """    # Apply optimizer (fase BM: hanya di batas akumulasi)
    lr_this_step = None
    if is_boundary:
        for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
            lr_this_step = compute_lr(base_lr, config, train_state)

            for param_group in optim.param_groups:
                param_group['lr'] = lr_this_step

            optim.step()
            optim.zero_grad()
    train_state._micro_idx = 0 if is_boundary else micro + 1"""
    t = t.replace(old, new)

    # (d) EMA hanya setelah langkah optimizer
    old = """            if config.ema:
                ema_helper.update(train_state.model)"""
    assert t.count(old) >= 1, "anchor ema"
    t = t.replace(old, """            if config.ema and getattr(train_state, "_micro_idx", 0) == 0:
                ema_helper.update(train_state.model)""", 1)
    changed = True
    print("pretrain.py: gradient accumulation ditambahkan")

if "TRM_EVAL_PREDS" not in t:
    old = """        carry = None
        processed_batches = 0

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1"""
    assert t.count(old) == 1, "anchor eval loop"
    new = """        carry = None
        processed_batches = 0
        _preds_dir = os.environ.get("TRM_EVAL_PREDS")
        _per_seq_batches = []
        import models.losses as _losses_mod

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if _preds_dir and rank == 0:
                _losses_mod._PER_SEQ_SINK = []"""
    t = t.replace(old, new)

    old = """            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish"""
    assert t.count(old) == 1, "anchor evaluator update"
    new = """            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            if _preds_dir and rank == 0 and _losses_mod._PER_SEQ_SINK is not None:
                # satu rekaman per slot: ambil panggilan saat slot itu halt
                import numpy as _np
                _recs = _losses_mod._PER_SEQ_SINK
                _losses_mod._PER_SEQ_SINK = None
                if len(_recs):
                    _n = _recs[0][0].shape[0]
                    _acc = _np.full(_n, _np.nan, dtype=_np.float32)
                    _ex = _np.zeros(_n, dtype=bool)
                    _seen = _np.zeros(_n, dtype=bool)
                    for _v, _a, _e in _recs:
                        _vm = _v.numpy()
                        _acc[_vm] = _a.numpy()[_vm]
                        _ex[_vm] = _e.numpy()[_vm]
                        _seen |= _vm
                    _per_seq_batches.append((_acc, _ex, _seen))

            del carry, loss, preds, batch, all_finish"""
    t = t.replace(old, new)

    old = """            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])"""
    assert t.count(old) == 1, "anchor metric_values"
    new = old + """

        if _preds_dir and rank == 0 and len(_per_seq_batches):
            import numpy as _np
            os.makedirs(_preds_dir, exist_ok=True)
            _tag = os.environ.get("TRM_RUN_TAG", "run")
            _np.savez_compressed(
                os.path.join(_preds_dir, f"preds_{_tag}_step{int(train_state.step)}.npz"),
                token_acc=_np.concatenate([b[0] for b in _per_seq_batches]),
                exact=_np.concatenate([b[1] for b in _per_seq_batches]),
                seen=_np.concatenate([b[2] for b in _per_seq_batches]))"""
    t = t.replace(old, new)
    changed = True
    print("pretrain.py: logging prediksi per-instance ditambahkan")

if changed:
    if "\nimport os" not in t and "\nimport os," not in t:
        t = t.replace("import torch", "import os\nimport torch", 1)
    open(PRE, "w").write(t)
    print("pretrain.py ditulis")
else:
    print("pretrain.py: sudah ter-patch")
