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

- Python 3.8
- conda (for environment creation by the pipeline)
- NVIDIA GPU
- Model checkpoints in `weights/` (download from Google Drive, see original README)
- ESM-1b weights (~2.5 GB, downloaded automatically during setup)

The model architecture and pretrained weights are unchanged. The original `Ampmm_base/` package, `run.py`, training configs, and all model code are
preserved as-is.

## Installation

```bash
TODO setting up enviroment with uv
```

## Usage within the pipeline

```bash
TODO show usage with SEQME https://github.com/szczurek-lab/seqme-thirdparty
```

The pipeline handles variant selection automatically based on the task configuration.

## Notes

- Cannot handle sequences shorter than 6 amino acids.
- ESM-1b embedding generation is the main bottleneck for large datasets.
- The classifier uses the balanced model by default.
- Regression output is MIC in micromolar.

## License

Same as the original SenseXAMP repository.