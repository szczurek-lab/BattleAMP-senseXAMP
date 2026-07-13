# SenseXAMP

Fork of [William-Zhanng/SenseXAMP](https://github.com/William-Zhanng/SenseXAMP), adapted
for integration with the
[battleamp-snakemake](https://github.com/szczurek-lab/battleamp-snakemake) benchmarking
pipeline.

## Supported tasks

AMP classification and AMP regression.

## Reference

Wentao Zhang, Yanchao Xu, Aowen Wang, Gang Chen, Junbo Zhao, Fuse feeds as one: cross-modal framework for general identification of AMPs, Briefings in Bioinformatics, Volume 24, Issue 6, November 2023, bbad336, https://doi.org/10.1093/bib/bbad336

## Model overview

SenseXAMP combines ESM-1b protein language model embeddings with sequence-derived
physicochemical descriptors. The architecture uses a multi-branch design: an
embedding branch with self-attention and a structured-data branch with fully connected
layers, fused for final prediction.

## Benchmark variants

| Variant | Task | Description |
|---|---|---|
| `sensexamp-classifier` | Classification | Balanced AMP/non-AMP classifier |
| `sensexamp-ecoli` | Regression | E. coli MIC prediction |
| `sensexamp-saureus` | Regression | S. aureus MIC prediction |

Each variant uses different pretrained weights (checkpoints in `weights/`).

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages the Python 3.9 environment and lockfile)
- GPU optional — runs on CUDA if available, otherwise CPU (slower; ESM-1b embedding is the bottleneck)
- Model checkpoints in `weights/` (downloaded by `setup.sh` from Google Drive; not committed)
- ESM-1b weights (~2.5 GB, downloaded automatically to the torch hub cache on first run)

The model architecture and pretrained weights are unchanged. The original `Ampmm_base/`
package, `run.py`, training configs, and all model code are preserved as-is.

## Installation

```bash
./setup.sh
```

This runs `uv sync` (builds the environment from the lockfile) and downloads the model
checkpoints into `weights/`. To do it manually: run `uv sync`, then fetch the checkpoints
per [weights/README.md](weights/README.md).

## Usage (seqme)

The plugin is called through seqme's
[`ThirdPartyModel`](https://github.com/szczurek-lab/seqme), which runs the entry point in
this repo's isolated uv environment:

```python
import seqme as sm

model = sm.models.ThirdPartyModel(
    entry_point="predict:predict",
    path="./plugins/thirdparty/sensexamp",
    url="https://github.com/szczurek-lab/BattleAMP-senseXAMP",
)

df = model(sequences=["KLLKLLKKLL", "GIGKFLHSAK"])
```

`predict()` returns a `pandas.DataFrame` with columns `sequence`, `Prediction`,
`Probability_score`, `MIC_ecoli`, `MIC_unit_ecoli`, `MIC_saureus`, `MIC_unit_saureus`.
**Note:** the `MIC_ecoli` / `MIC_saureus` values are **log10(MIC / uM)** — the regression
heads predict MIC on a log10 scale and no inverse transform is applied.

### Per-variant entry points

For seqme's `ID` / `Threshold` metrics, which expect one score per sequence, use the
variant entry points instead. Each returns a 1-D `numpy.ndarray` aligned to the input
order (`NaN` for sequences dropped by validation):

| Entry point | Returns | Objective |
|---|---|---|
| `predict:predict_amp` | AMP classification probability | maximize |
| `predict:predict_ecoli` | E. coli MIC in uM | minimize |
| `predict:predict_ecoli_log10` | E. coli MIC, log10(uM) | minimize |
| `predict:predict_saureus` | S. aureus MIC in uM | minimize |
| `predict:predict_saureus_log10` | S. aureus MIC, log10(uM) | minimize |

**Weights are not committed** (too large; `.gitignore`d), so the auto-clone form fetches
only code. After cloning, run `./setup.sh` in the plugin directory to download the
checkpoints — or pre-provision the clone and pass `path=<dir>` with `url=None`.

## Notes

- Handles sequences of 6-50 amino acids; others are reported as `NaN`. (The 50-aa cap is
  a resource guard, not a model limit — the original SenseXAMP imposes no length cap.)
- ESM-1b embedding generation is the main bottleneck for large datasets.
- The classifier uses the balanced model by default.
- Regression output is **log10(MIC / uM)** (the `*_uM` entry points return linear uM).

## License

Same as the original SenseXAMP repository.