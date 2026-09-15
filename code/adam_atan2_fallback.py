"""Fallback shim: AdamATan2 -> AdamW.
The compiled `adam-atan2` package does not build for sm_120, so EVERY run in the study
(calibration, pilot and faithful-recipe, TRM and baseline alike) uses this shim with the
upstream lr / weight-decay / betas. Applied identically to all configurations, so
comparisons remain matched; absolute accuracies are not those of the upstream recipe."""
import torch

class AdamATan2(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), **kw):
        kw.pop("eps", None)
        super().__init__(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=1e-8)
