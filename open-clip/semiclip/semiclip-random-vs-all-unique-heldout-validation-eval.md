## Final Stage-2 caption-policy comparison: held-out validation

This comparison uses only the 219-image held-out partition of the labeled RSICD
`val` split (`seed: 42`, 80/20 split). Its exact filenames are read from the
random-caption workflow's `concepts.json` artifact. Neither model was trained
on these images.

Only the final Stage-2 random-caption and all-unique-caption checkpoints are
compared. Supervised controls and Stage-1 checkpoints are intentionally
excluded.

| Final Stage-2 model | Accuracy | Weighted precision | Weighted recall | Weighted F1 | Correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random-caption SemiCLIP | 90.41% | **90.91%** | 90.41% | **90.17%** | 198/219 |
| All-unique SemiCLIP | 90.41% | 90.42% | 90.41% | 89.77% | 198/219 |

The models have identical accuracy. All-unique corrects seven random-caption
errors but regresses on seven other images; the exact two-sided paired test is
`p = 1.0`. This validation result provides no accuracy evidence for selecting
one caption policy over the other. The small weighted-F1 difference favors the
random-caption checkpoint descriptively.

### Artifacts

- [`configs/zeroshot_compare_random_vs_all_unique_final_stage2_heldout_validation.yaml`](configs/zeroshot_compare_random_vs_all_unique_final_stage2_heldout_validation.yaml) defines the exact model and held-out-subset inputs.
- [`evaluations/semiclip_random_vs_all_unique_final_stage2_heldout_validation`](../../evaluations/semiclip_random_vs_all_unique_final_stage2_heldout_validation) contains the metrics and per-image predictions.

### Plots

- Random-caption training: [training plots](../../results/open_clip_semiclip_paper_1gpu_seed42/plots).
- All-unique training: [Stage-1 losses](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun/plots/stage1_losses.png), [Stage-2 losses](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun/plots/stage2_losses.png), [learning rate](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun/plots/learning_rate.png), [selection counts](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun/plots/selection_counts.png), and [logit scales](../../results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun/plots/logit_scales.png).
- Held-out validation evaluation: [model comparison](../../evaluations/semiclip_random_vs_all_unique_final_stage2_heldout_validation/model_comparison.png), [combined confusion matrices](../../evaluations/semiclip_random_vs_all_unique_final_stage2_heldout_validation/confusion_matrices_comparison.png), [random-caption confusion matrix](../../evaluations/semiclip_random_vs_all_unique_final_stage2_heldout_validation/semiclip_random_caption_stage2_final/confusion_matrix.png), and [all-unique confusion matrix](../../evaluations/semiclip_random_vs_all_unique_final_stage2_heldout_validation/semiclip_all_unique_stage2_final/confusion_matrix.png).
