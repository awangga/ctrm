"""Fallback shim: AdamATan2 -> AdamW (drop-in for calibration).
Only used if the compiled `adam-atan2` package fails to build.
Throughput/energy calibration is insensitive to the exact optimizer math."""
import torch

class AdamATan2(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), **kw):
        kw.pop("eps", None)
        super().__init__(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=1e-8)
