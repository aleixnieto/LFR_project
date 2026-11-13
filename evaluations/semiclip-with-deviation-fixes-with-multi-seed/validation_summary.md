# Joint validation evaluation summary

This report combines the three-seed held-out validation evaluations for the
supervised OpenCLIP and SemiCLIP caption-policy runs. Each model was evaluated
zero-shot on the same 219-image validation subset derived from the RSICD `val`
split. The subset was fixed with `split_seed: 42`; training seeds were 42, 123,
and 2026. Values below are means and sample standard deviations across those
three training seeds.

## Results

| Model family | Variant | Accuracy | Weighted F1 | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| Supervised OpenCLIP | `sup1` (random-one captions) | 88.58% ± 1.65% | 87.36% ± 1.15% | 87.36% ± 1.45% |
| Supervised OpenCLIP | `supAll` (all-unique captions) | 92.09% ± 0.26% | 91.24% ± 0.22% | 91.74% ± 0.36% |
| SemiCLIP | `semiCLIP1` (random-one captions) | 90.72% ± 0.95% | 90.38% ± 1.02% | 90.44% ± 1.19% |
| SemiCLIP | `semiCLIPall` (all-unique captions) | 90.41% ± 2.28% | 89.78% ± 2.40% | 89.85% ± 2.61% |

The standard deviation is the sample SD (`ddof=1`). The caption-policy
comparisons are paired by seed and validation split. `sup1` and `supAll` are
supervised OpenCLIP fine-tuning; `semiCLIP1` and `semiCLIPall` are paper-aligned
SemiCLIP Stage-2 checkpoints.

## Detailed reports

- [Supervised OpenCLIP validation aggregate](sup-deterministic-multi-seed/aggregate/report.md)
- [SemiCLIP validation aggregate](semiclip-with-deviation-fixes-with-multi-seed/aggregate/report.md)

Per-seed predictions and metrics are under the corresponding evaluation
folders, in `seed42`, `seed123`, and `seed2026`.
