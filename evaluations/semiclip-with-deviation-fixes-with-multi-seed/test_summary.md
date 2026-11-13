# Joint test-set evaluation summary

This report combines the three-seed test-set evaluations for the supervised
OpenCLIP and SemiCLIP caption-policy runs. Each model was evaluated on the
full 1,093-image RSICD `test` split. Values are means and sample standard
deviations across training seeds 42, 123, and 2026.

## Results

| Model family | Variant | Accuracy | Weighted F1 | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| Supervised OpenCLIP | `sup1` (random-one captions) | 82.37% ± 0.14% | 81.09% ± 0.28% | 82.34% ± 0.19% |
| Supervised OpenCLIP | `supAll` (all-unique captions) | 85.03% ± 1.09% | 83.67% ± 1.27% | 85.14% ± 1.20% |
| SemiCLIP | `semiCLIP1` (random-one captions) | 83.35% ± 0.48% | 82.57% ± 0.56% | 83.96% ± 0.59% |
| SemiCLIP | `semiCLIPall` (all-unique captions) | 82.95% ± 0.74% | 81.50% ± 0.62% | 83.05% ± 0.75% |

The standard deviation is the sample SD (`ddof=1`).

## Detailed reports

- [Supervised OpenCLIP test aggregate](../sup-deterministic-multi-seed/test/aggregate/report.md)
- [SemiCLIP test aggregate](test/aggregate/report.md)

Per-seed predictions and metrics are in each family's `test/seed42`,
`test/seed123`, and `test/seed2026` folders.

## Image-to-text retrieval

Retrieval checks whether a query image's captions appear among the top-(k)
globally ranked unique captions from the full test split. Values are mean ±
sample SD across seeds 42, 123, and 2026.

| Model family | Variant | Accuracy@1 | Accuracy@3 | Accuracy@5 | Accuracy@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Supervised OpenCLIP | `sup1` | 10.49% ± 0.61% | 22.32% ± 1.20% | 31.23% ± 1.23% | 47.00% ± 1.44% |
| Supervised OpenCLIP | `supAll` | 11.38% ± 1.37% | 25.71% ± 2.38% | 35.86% ± 2.62% | 51.20% ± 1.66% |
| SemiCLIP | `semiCLIP1` | 9.24% ± 0.84% | 22.17% ± 0.97% | 30.92% ± 0.27% | 46.48% ± 0.84% |
| SemiCLIP | `semiCLIPall` | 12.47% ± 1.06% | 25.71% ± 0.99% | 34.46% ± 0.79% | 49.19% ± 0.69% |

Detailed retrieval reports:

- [Supervised OpenCLIP retrieval aggregate](../sup-deterministic-multi-seed/retrieval_test/aggregate/report.md)
- [SemiCLIP retrieval aggregate](retrieval_test/aggregate/report.md)

### Paired McNemar tests for R@10

For each seed, McNemar's exact two-sided test compares whether each of the
same 1,093 images is retrieved in the top 10 by SemiCLIP and its corresponding
supervised OpenCLIP model. `SemiCLIP better` and `Supervised better` count the
discordant image pairs in each direction. P-values are unadjusted and are
reported per seed; results are not pooled across seeds because they reuse the
same test images.

| Comparison | Seed | Sup R@10 | SemiCLIP R@10 | SemiCLIP better | Sup better | Exact p (McNemar) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `semiCLIP1` vs `sup1` | 42 | 45.38% | 46.66% | 184 | 170 | 0.489656 |
| `semiCLIP1` vs `sup1` | 123 | 48.12% | 45.56% | 157 | 185 | 0.144182 |
| `semiCLIP1` vs `sup1` | 2026 | 47.48% | 47.21% | 168 | 171 | 0.913521 |
| `semiCLIPall` vs `supAll` | 42 | 52.42% | 48.40% | 152 | 196 | 0.021031 |
| `semiCLIPall` vs `supAll` | 123 | 51.88% | 49.68% | 138 | 162 | 0.184117 |
| `semiCLIPall` vs `supAll` | 2026 | 49.31% | 49.50% | 162 | 160 | 0.955570 |

The [per-image paired-test report](retrieval_test/mcnemar/retrieval_mcnemar_r10.md)
contains discordant counts and a machine-readable JSON result.
