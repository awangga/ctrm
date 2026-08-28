# Third-party licenses

The code and analysis written for this study are MIT-licensed (see `LICENSE`). The archive also
redistributes the third-party works below, each under its own license, with the original license text
retained in place.

| Component | Path | Upstream | License | Copyright |
|---|---|---|---|---|
| Tiny Recursive Models | `vendor/TinyRecursiveModels/` | `SamsungSAILMontreal/TinyRecursiveModels` @ `c011037` | MIT | (c) 2025 Samsung Electronics Co., Ltd. |
| ARC-AGI-1 tasks | `vendor/arc-agi-1-raw/` | `fchollet/ARC-AGI` @ `3990304` | Apache-2.0 | (c) Francois Chollet |

Full license texts: `vendor/TinyRecursiveModels/LICENSE` and `vendor/arc-agi-1-raw/LICENSE`.

The vendored trees are unmodified. Changes we make to the Tiny Recursive Models source are kept
separately in `vendor/patches/` and are covered by this package's own MIT license; `vendor/README.md`
describes each one and why it exists.

## Fetched at build time, not redistributed
- **Sudoku-Extreme** (`sapientinc/sudoku-extreme`) and **Maze-Hard**
  (`sapientinc/maze-30x30-hard-1k`), downloaded from Hugging Face by TRM's dataset builders. Neither
  declares a license, so we do not redistribute them.
- **PyTorch** and the Python dependencies in `requirements.txt`, installed from their own indexes under
  their own licenses.
