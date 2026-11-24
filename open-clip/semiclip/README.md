# SemiCLIP Evaluations

* [Random-caption variant **test-set** evaluation](semiclip-random-caption-test-set-eval.md)
* [All-unique-captions variant **test-set** evaluation](semiclip-all-unique-caption-test-set-eval.md)
* [Random-caption and all-unique-captions comparison on **validation data**](semiclip-random-vs-all-unique-heldout-validation-eval.md)

## Hyperparameter sensitivity analysis (Section 4.3)

Evaluated on the 219-image held-out partition of the labeled `val` split. All sweeps use the paper-faithful single-GPU recipe (AdamW, `5e-5`, seed 42) as baseline (`selection_ratio=0.3`, `concept_loss_top_k=3`).

### Selection ratio sweep

| Selection ratio | Accuracy | Weighted F1 | Errors |
| ---: | ---: | ---: | ---: |
| 0.1 | 90.87% | 90.66% | 20 |
| 0.3 (baseline) | 89.95% | 89.72% | 22 |
| 0.5 | 91.78% | 91.39% | 18 |
| 0.7 | 90.41% | 90.17% | 21 |
| 0.9 | 91.78% | 91.70% | 18 |
| 1.0 | 91.32% | 91.16% | 19 |

Accuracy varies by at most 1.8 pp across ratios. No pairwise difference is significant (exact two-sided McNemar, all p > 0.12), confirming low sensitivity to this parameter.

### Concept loss top-k and learning rate

| Setting | Value | Accuracy | Weighted F1 |
| --- | ---: | ---: | ---: |
| concept_loss_top_k | 1 | 90.41% | 90.29% |
| concept_loss_top_k | 3 (baseline) | 89.95% | 89.72% |
| concept_loss_top_k | 5 | 91.32% | 90.81% |
| learning_rate | 1e-5 | 91.78% | 90.81% |
| learning_rate | 5e-5 (baseline) | 89.95% | 89.72% |
| learning_rate | 1e-4 | 89.95% | 89.75% |

### Artifacts

- [`configs/zeroshot_selection_ratio_heldout.yaml`](configs/zeroshot_selection_ratio_heldout.yaml) defines the selection ratio evaluation on the held-out split.
- [`configs/zeroshot_ablations_heldout.yaml`](configs/zeroshot_ablations_heldout.yaml) defines the top-k and learning rate evaluation.
- [`evaluations/semiclip_selection_ratio_heldout`](../../evaluations/semiclip_selection_ratio_heldout) contains metrics, predictions, and analysis plots.
- [`evaluations/semiclip_ablations_heldout`](../../evaluations/semiclip_ablations_heldout) contains metrics and predictions for top-k and learning rate ablations.

## Failure case analysis (Section 4.4)

### Playfields/Playground class merge diagnostic

Merging *playfields* and *playground* into a single *RecreationalArea* class raises accuracy from 93.05% to 95.61% on the test split, confirming that confusion between these two semantically overlapping classes is the primary error source.

### Qualitative failure figure

[`evaluations/semiclip_qualitative_figure/failure_cases.png`](../../evaluations/semiclip_qualitative_figure/failure_cases.png) shows three representative misclassified test images with their true and predicted labels.

### Artifacts

- [`merge_playground_playfields.py`](merge_playground_playfields.py) computes the merged-class diagnostic.
- [`generate_qualitative_figure.py`](generate_qualitative_figure.py) produces the qualitative failure figure.
- [`evaluations/semiclip_qualitative_figure`](../../evaluations/semiclip_qualitative_figure) contains the qualitative failure figure.
- [`evaluations/semiclip_failure_analysis_baseline_run4`](../../evaluations/semiclip_failure_analysis_baseline_run4) contains the confusion matrix and playground/playfields failure visualization.

### Running the evaluations

All commands are run from the project root (`LFR_project/`).

Selection ratio sweep on held-out split:
```bash
cd open-clip && python main_zeroshot_compare_subset.py \
  --models-config semiclip/configs/zeroshot_selection_ratio_heldout.yaml \
  --subset-json semiclip/results/semiclip_ablation/baseline/concepts.json \
  --subset-key labeled_validation_filenames \
  --output-dir ../evaluations/semiclip_selection_ratio_heldout
```

Ablations (top-k, learning rate, loss terms) on held-out split:
```bash
cd open-clip && python main_zeroshot_compare_subset.py \
  --models-config semiclip/configs/zeroshot_ablations_heldout.yaml \
  --subset-json semiclip/results/semiclip_ablation/baseline/concepts.json \
  --subset-key labeled_validation_filenames \
  --output-dir ../evaluations/semiclip_ablations_heldout
```

Analysis and McNemar tests:
```bash
cd open-clip/semiclip && python analyze_selection_ratio_heldout.py
```

Playfields/playground merge diagnostic:
```bash
cd open-clip/semiclip && python merge_playground_playfields.py
```

Qualitative failure figure:
```bash
cd open-clip/semiclip && python generate_qualitative_figure.py
```