# Paired test-set retrieval R@10 comparisons

Exact two-sided McNemar tests compare per-image R@10 hits for each seed.
Positive direction is SemiCLIP: `semiclip_better` counts images only SemiCLIP retrieves in the top 10; `sup_better` counts images only supervised OpenCLIP retrieves in the top 10.
Seeds are tested separately because they reuse the same test images.

| Comparison | Seed | Sup R@10 | SemiCLIP R@10 | SemiCLIP better | Sup better | Discordant | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semiCLIP1 vs sup1 | 42 | 0.4538 | 0.4666 | 184 | 170 | 354 | 0.489656 |
| semiCLIP1 vs sup1 | 123 | 0.4812 | 0.4556 | 157 | 185 | 342 | 0.144182 |
| semiCLIP1 vs sup1 | 2026 | 0.4748 | 0.4721 | 168 | 171 | 339 | 0.913521 |
| semiCLIPall vs supAll | 42 | 0.5242 | 0.4840 | 152 | 196 | 348 | 0.0210306 |
| semiCLIPall vs supAll | 123 | 0.5188 | 0.4968 | 138 | 162 | 300 | 0.184117 |
| semiCLIPall vs supAll | 2026 | 0.4931 | 0.4950 | 162 | 160 | 322 | 0.95557 |
