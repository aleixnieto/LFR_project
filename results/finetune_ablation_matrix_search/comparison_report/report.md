# Matrix-ablation comparison report

Generated: 2026-10-19T17:02:10.547021+00:00

## Completion

- Expected runs: 30
- Runs with metrics: 30
- Missing runs: 0
- Runs with classification metrics: 30
- Missing classification runs: 0

## Comparison rules

All runs use the same seed-42 875/219 train/validation split and the same all-unique/per-caption validation objective. Best epoch means minimum validation.selection_loss.

Loss values are compared only within a fixed batch size. Cross-batch selection uses zero-shot accuracy and macro F1 on the same held-out validation images.

## Batch size 16

| Learning rate | Adafactor loss | Peak epoch | AdamW loss | Peak epoch |
| ---: | ---: | ---: | ---: | ---: |
| 5e-6 | 2.028692 | 8 | 1.156699 | 5 |
| 1e-5 | 1.724831 | 8 | 1.118015 | 3 |
| 3e-5 | 1.361083 | 8 | **1.079904** | 7 |
| 5e-5 | 1.265864 | 8 | 1.116115 | 8 |
| 1e-4 | 1.187046 | 8 | 1.237155 | 8 |

## Batch size 32

| Learning rate | Adafactor loss | Peak epoch | AdamW loss | Peak epoch |
| ---: | ---: | ---: | ---: | ---: |
| 5e-6 | 2.777504 | 8 | 1.507355 | 3 |
| 1e-5 | 2.411205 | 8 | 1.466397 | 3 |
| 3e-5 | 1.884376 | 8 | **1.452718** | 3 |
| 5e-5 | 1.735214 | 8 | 1.454627 | 4 |
| 1e-4 | 1.602528 | 8 | 1.553960 | 7 |

## Batch size 64

| Learning rate | Adafactor loss | Peak epoch | AdamW loss | Peak epoch |
| ---: | ---: | ---: | ---: | ---: |
| 5e-6 | 3.453922 | 8 | 1.846049 | 7 |
| 1e-5 | 3.103115 | 8 | 1.775819 | 4 |
| 3e-5 | 2.461927 | 8 | 1.750826 | 4 |
| 5e-5 | 2.243061 | 8 | **1.748169** | 4 |
| 1e-4 | 2.038338 | 8 | 1.820190 | 4 |

## Peak timing

| Run | Best epoch | Completed epochs | Best validation loss | Last validation loss |
| --- | ---: | ---: | ---: | ---: |
| bs16_adafactor_lr_5e-6 | 8 | 8/8 | 2.028692 | 2.028692 |
| bs16_adamw_lr_5e-6 | 5 | 8/8 | 1.156699 | 1.180836 |
| bs16_adafactor_lr_1e-5 | 8 | 8/8 | 1.724831 | 1.724831 |
| bs16_adamw_lr_1e-5 | 3 | 8/8 | 1.118015 | 1.139298 |
| bs16_adafactor_lr_3e-5 | 8 | 8/8 | 1.361083 | 1.361083 |
| bs16_adamw_lr_3e-5 | 7 | 8/8 | 1.079904 | 1.132220 |
| bs16_adafactor_lr_5e-5 | 8 | 8/8 | 1.265864 | 1.265864 |
| bs16_adamw_lr_5e-5 | 8 | 8/8 | 1.116115 | 1.116115 |
| bs16_adafactor_lr_1e-4 | 8 | 8/8 | 1.187046 | 1.187046 |
| bs16_adamw_lr_1e-4 | 8 | 8/8 | 1.237155 | 1.237155 |
| bs32_adafactor_lr_5e-6 | 8 | 8/8 | 2.777504 | 2.777504 |
| bs32_adamw_lr_5e-6 | 3 | 8/8 | 1.507355 | 1.565923 |
| bs32_adafactor_lr_1e-5 | 8 | 8/8 | 2.411205 | 2.411205 |
| bs32_adamw_lr_1e-5 | 3 | 8/8 | 1.466397 | 1.598613 |
| bs32_adafactor_lr_3e-5 | 8 | 8/8 | 1.884376 | 1.884376 |
| bs32_adamw_lr_3e-5 | 3 | 8/8 | 1.452718 | 1.491046 |
| bs32_adafactor_lr_5e-5 | 8 | 8/8 | 1.735214 | 1.735214 |
| bs32_adamw_lr_5e-5 | 4 | 8/8 | 1.454627 | 1.522293 |
| bs32_adafactor_lr_1e-4 | 8 | 8/8 | 1.602528 | 1.602528 |
| bs32_adamw_lr_1e-4 | 7 | 8/8 | 1.553960 | 1.569907 |
| bs64_adafactor_lr_5e-6 | 8 | 8/8 | 3.453922 | 3.453922 |
| bs64_adamw_lr_5e-6 | 7 | 8/8 | 1.846049 | 1.849459 |
| bs64_adafactor_lr_1e-5 | 8 | 8/8 | 3.103115 | 3.103115 |
| bs64_adamw_lr_1e-5 | 4 | 8/8 | 1.775819 | 1.850139 |
| bs64_adafactor_lr_3e-5 | 8 | 8/8 | 2.461927 | 2.461927 |
| bs64_adamw_lr_3e-5 | 4 | 8/8 | 1.750826 | 1.928290 |
| bs64_adafactor_lr_5e-5 | 8 | 8/8 | 2.243061 | 2.243061 |
| bs64_adamw_lr_5e-5 | 4 | 8/8 | 1.748169 | 1.872952 |
| bs64_adafactor_lr_1e-4 | 8 | 8/8 | 2.038338 | 2.038338 |
| bs64_adamw_lr_1e-4 | 4 | 8/8 | 1.820190 | 2.012557 |

## Shared-split zero-shot classification

| Run | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| bs16_adafactor_lr_5e-6 | 0.689498 | 0.641647 | 0.664509 |
| bs16_adamw_lr_5e-6 | 0.885845 | 0.859627 | 0.864584 |
| bs16_adafactor_lr_1e-5 | 0.739726 | 0.696237 | 0.710828 |
| bs16_adamw_lr_1e-5 | 0.844749 | 0.827407 | 0.829514 |
| bs16_adafactor_lr_3e-5 | 0.789954 | 0.745900 | 0.760085 |
| bs16_adamw_lr_3e-5 | **0.922374** | **0.921422** | 0.914824 |
| bs16_adafactor_lr_5e-5 | 0.826484 | 0.791763 | 0.801403 |
| bs16_adamw_lr_5e-5 | 0.899543 | 0.888294 | 0.888989 |
| bs16_adafactor_lr_1e-4 | 0.844749 | 0.812282 | 0.821591 |
| bs16_adamw_lr_1e-4 | 0.867580 | 0.855474 | 0.859016 |
| bs32_adafactor_lr_5e-6 | 0.689498 | 0.635315 | 0.659028 |
| bs32_adamw_lr_5e-6 | 0.844749 | 0.816064 | 0.822446 |
| bs32_adafactor_lr_1e-5 | 0.721461 | 0.673545 | 0.691213 |
| bs32_adamw_lr_1e-5 | 0.881279 | 0.858002 | 0.867849 |
| bs32_adafactor_lr_3e-5 | 0.785388 | 0.739854 | 0.752404 |
| bs32_adamw_lr_3e-5 | 0.853881 | 0.825875 | 0.824301 |
| bs32_adafactor_lr_5e-5 | 0.812785 | 0.771304 | 0.781599 |
| bs32_adamw_lr_5e-5 | 0.876712 | 0.865600 | 0.863710 |
| bs32_adafactor_lr_1e-4 | 0.844749 | 0.811564 | 0.820901 |
| bs32_adamw_lr_1e-4 | 0.858447 | 0.835403 | 0.839827 |
| bs64_adafactor_lr_5e-6 | 0.680365 | 0.625138 | 0.649128 |
| bs64_adamw_lr_5e-6 | 0.867580 | 0.832611 | 0.839575 |
| bs64_adafactor_lr_1e-5 | 0.689498 | 0.633749 | 0.657089 |
| bs64_adamw_lr_1e-5 | 0.881279 | 0.856209 | 0.860476 |
| bs64_adafactor_lr_3e-5 | 0.762557 | 0.718914 | 0.731092 |
| bs64_adamw_lr_3e-5 | 0.872146 | 0.870055 | 0.869391 |
| bs64_adafactor_lr_5e-5 | 0.785388 | 0.739782 | 0.752087 |
| bs64_adamw_lr_5e-5 | 0.885845 | 0.868150 | 0.881479 |
| bs64_adafactor_lr_1e-4 | 0.826484 | 0.788348 | 0.798924 |
| bs64_adamw_lr_1e-4 | 0.849315 | 0.845383 | 0.846022 |

## Conclusion

Completed conclusions based on 30/30 runs:

- Batch size 16: adamw at 3e-5 had the lowest validation loss (1.079904) at epoch 7.
- Batch size 32: adamw at 3e-5 had the lowest validation loss (1.452718) at epoch 3.
- Batch size 64: adamw at 5e-5 had the lowest validation loss (1.748169) at epoch 4.
- The within-batch winner changes with batch size, indicating an interaction between the matrix dimensions.
- 13 runs peaked before their final completed epoch; 17 were still best at their final completed epoch.

**No global batch-size winner is selected from contrastive loss.** Its scale changes with the number of in-batch negatives.

- Accuracy leader(s): bs16_adamw_lr_3e-5 (0.922374).
- Macro-F1 leader(s): bs16_adamw_lr_3e-5 (0.921422).
- **Global validation winner: bs16_adamw_lr_3e-5.** It leads both zero-shot accuracy and macro F1 on the shared split.

## Artifacts

- summary.csv
- validation_loss_by_learning_rate.png
- best_epoch_by_learning_rate.png
- classification_metrics_by_learning_rate.png (when available)
