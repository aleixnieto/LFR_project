## All-unique-caption SemiCLIP: fixed-test evaluation

These results were produced on 2026-10-23 with seed 42. Every classification
result uses the fixed 1,093-image, 31-class RSICD `test` split, identical class
prompts, preprocessing, and zero-shot evaluator. This is final reporting for
the predefined paper-faithful single-GPU recipe, not a hyperparameter-selection
result.

The principal run uses
[`configs/semiclip_rsicd_paper_1gpu_all_unique_seed42.yaml`](configs/semiclip_rsicd_paper_1gpu_all_unique_seed42.yaml):
`ViT-B-32 / laion2b_s34b_b79k`, 875 labeled and 8,734 unlabeled images,
all token-distinct labeled captions, 25 Stage-1 epochs, 15 Stage-2 epochs,
AdamW at `5e-5`, weight decay `5e-4`, ten warm-up steps, and cosine decay.

The matched supervised control uses
[`configs/supervised_all_unique_control_paper_1gpu.yaml`](configs/supervised_all_unique_control_paper_1gpu.yaml).
It retains the principal run's labeled-data side and all-unique/per-caption
objective, but removes concept supervision and unlabeled images.

### Zero-shot classification

| Model | Checkpoint selection | Accuracy | Weighted precision | Weighted recall | Weighted F1 | Correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original LAION OpenCLIP | Pretrained | 62.67% | 61.60% | 62.67% | 59.09% | 685/1,093 |
| Controlled supervised, all unique | Best caption-validation loss | 75.30% | 76.94% | 75.30% | 74.01% | 823/1,093 |
| Controlled supervised, all unique | Final epoch 25 | 81.79% | 84.02% | 81.79% | 80.37% | 894/1,093 |
| All-unique SemiCLIP Stage 1 | Final epoch 25 | 83.71% | 85.95% | 83.71% | 82.31% | 915/1,093 |
| All-unique SemiCLIP Stage 2 | Final epoch 15 | **87.65%** | **88.57%** | **87.65%** | **87.20%** | **958/1,093** |

### Controlled contribution breakdown

The final Stage-1 checkpoint is the Stage-2 input, and final checkpoints are
therefore used for the controlled transitions below. P-values are exact
two-sided paired McNemar/binomial tests over the fixed test predictions.

| Transition | Accuracy change | Weighted-F1 change | Prediction changes | Exact paired p-value |
| --- | ---: | ---: | --- | ---: |
| Controlled supervised final → SemiCLIP Stage-1 final | +1.92 pp | +1.94 pp | 64 corrected, 43 regressed | 0.0527 |
| SemiCLIP Stage-1 final → SemiCLIP Stage-2 final | **+3.93 pp** | **+4.89 pp** | 71 corrected, 28 regressed | **1.83e-5** |
| Controlled supervised final → complete SemiCLIP | **+5.86 pp** | **+6.35 pp** | 98 corrected, 34 regressed | **2.23e-8** |

The all-unique recipe's larger gain is supplied by Stage 2. The test split must
not be used to choose Stage-1 or Stage-2 settings; those choices belong to the
held-out labeled-validation subset.

### Artifacts

- [`evaluations/semiclip_all_unique_functionality_rerun`](../../evaluations/semiclip_all_unique_functionality_rerun) contains per-model metrics, predictions, and confusion matrices.
- [`results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun`](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun) contains the principal run.
- [`results/open_clip_supervised_all_unique_control_paper_1gpu_seed42_functionality_rerun`](../../results/open_clip_supervised_all_unique_control_paper_1gpu_seed42_functionality_rerun) contains the matched supervised control.

### Plots

- Principal training: [Stage-1 losses](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun/plots/stage1_losses.png), [Stage-2 losses](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun/plots/stage2_losses.png), [learning rate](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun/plots/learning_rate.png), [selection counts](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun/plots/selection_counts.png), and [logit scales](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun/plots/logit_scales.png).
- Supervised-control training: [losses](../../results/open_clip_supervised_all_unique_control_paper_1gpu_seed42_functionality_rerun/plots/supervised_losses.png), [learning rate](../../results/open_clip_supervised_all_unique_control_paper_1gpu_seed42_functionality_rerun/plots/learning_rate.png), and [logit scales](../../results/open_clip_supervised_all_unique_control_paper_1gpu_seed42_functionality_rerun/plots/logit_scales.png).
- Test evaluation: [model comparison](../../evaluations/semiclip_all_unique_functionality_rerun/model_comparison.png) and [combined confusion matrices](../../evaluations/semiclip_all_unique_functionality_rerun/confusion_matrices_comparison.png); [per-model confusion matrices](../../evaluations/semiclip_all_unique_functionality_rerun) are stored alongside the predictions.
