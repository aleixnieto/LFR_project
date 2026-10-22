# Supervised fine-tuning ablations

This directory contains the staged ablation configurations for full-parameter
fine-tuning of OpenCLIP `ViT-B-32` with the
`laion2b_s34b_b79k` checkpoint.

## Fixed reference setup

Unless a stage explicitly changes a field, every run uses:

- RSICD JSON `val` split, as required by the assignment-aligned workflow;
- deterministic 80/20 split with seed 42 (875 train and 219 validation images);
- batch size 32, four data-loader workers, five epochs, and 10% warmup;
- Adafactor and linear learning-rate decay, which are currently fixed in
  `main_finetune.py`;
- a separate output directory for every run.

At batch size 32, a full run has 28 training batches per epoch and 140 optimizer
updates over five epochs. Do not pass `--max-train-batches` or
`--max-validation-batches` for an ablation run.

## Stage 1: learning rate

Stage 1 is fully runnable. It changes only the peak learning rate:

| Config | Learning rate | Role |
| --- | ---: | --- |
| `stage1_lr_1e-5.yaml` | `1e-5` | conservative |
| `stage1_lr_3e-5.yaml` | `3e-5` | reference |
| `stage1_lr_5e-5.yaml` | `5e-5` | CLIP-RSICD-like GPU setting |
| `stage1_lr_1e-4.yaml` | `1e-4` | aggressive/current-project setting |

Run one configuration with:

```bash
python open-clip/main_finetune.py \
  --config open-clip/configs/ablations/stage1_lr_3e-5.yaml \
  --device cuda
```

Run all four sequentially with:

```bash
for config in open-clip/configs/ablations/stage1_lr_*.yaml; do
  python open-clip/main_finetune.py --config "$config" --device cuda || break
done
```

Select the learning rate using held-out validation results, not the official
test split. Use validation contrastive loss together with validation zero-shot
accuracy and macro F1 once those classification metrics are available.

Stage 1 is complete. The held-out classification results confirm the ordering
suggested by validation loss:

| Learning rate | Validation loss | Accuracy | Weighted F1 | Macro F1 |
| ---: | ---: | ---: | ---: | ---: |
| `1e-5` | 2.6731 | 0.6895 | 0.6580 | 0.6344 |
| `3e-5` | 2.0826 | 0.7534 | 0.7192 | 0.7058 |
| `5e-5` | 1.8776 | 0.7854 | 0.7495 | 0.7367 |
| `1e-4` | **1.6856** | **0.8265** | **0.7969** | **0.7883** |

The `1e-4` run is the Stage 1 winner used in Stage 2.

## Stage 2: image batch size

Using the Stage 1 learning-rate winner, compare image batch sizes 16, 32, and 64
while keeping the winning `1e-4` learning rate fixed:

| Config/result | Batch size | Role |
| --- | ---: | --- |
| `stage2_bs16_lr_1e-4.yaml` | 16 | smaller contrastive batch |
| Stage 1 `stage1_lr_1e-4` result | 32 | reused reference |
| `stage2_bs64_lr_1e-4.yaml` | 64 | larger contrastive batch |

Run the two new configurations sequentially with:

```bash
for config in open-clip/configs/ablations/stage2_bs*.yaml; do
  python open-clip/main_finetune.py --config "$config" --device cuda || break
done
```

Use batch 64 only if it fits in GPU memory. Gradient accumulation is not
equivalent because it does not enlarge the contrastive similarity matrix.

Stage 2 is complete. Because contrastive-loss magnitude depends on the number
of in-batch negatives, batch sizes were compared on the same 219-image held-out
subset rather than by raw validation loss:

| Batch size | Accuracy | Weighted F1 | Macro F1 |
| ---: | ---: | ---: | ---: |
| 16 | **0.8493** | **0.8242** | **0.8161** |
| 32 | 0.8265 | 0.7969 | 0.7883 |
| 64 | 0.7945 | 0.7642 | 0.7534 |

For context, the original pretrained model scored 0.6575 accuracy, 0.6350
weighted F1, and 0.6128 macro F1 on that subset. Batch 16 is the Stage 2 winner.

## Stage 3: duration

The batch-16 validation loss was still improving at epoch five, so compare its
existing five-epoch result with a new eight-epoch run:

| Config/result | Epochs | Role |
| --- | ---: | --- |
| Stage 2 `stage2_bs16_lr_1e-4` result | 5 | reused reference |
| `stage3_ep8_bs16_lr_1e-4.yaml` | 8 | longer schedule |

Run the new configuration with:

```bash
python open-clip/main_finetune.py \
  --config open-clip/configs/ablations/stage3_ep8_bs16_lr_1e-4.yaml \
  --device cuda
```

The scheduler horizon changes with `epochs`, so this duration variant starts
again from the original pretrained checkpoint rather than resuming the
five-epoch run. Compare it with the five-epoch model on the same held-out
classification metrics used in Stage 2.

Stage 3 is complete. Eight epochs achieved validation loss 1.1829, compared
with 1.2284 for the five-epoch schedule. On the held-out classification set,
the difference is small:

| Epochs | Accuracy | Weighted F1 | Macro F1 |
| ---: | ---: | ---: | ---: |
| 5 | 0.8493 | 0.8242 | 0.8161 |
| 8 | **0.8539** | **0.8295** | **0.8177** |

Eight epochs is the provisional Stage 3 winner, but the accuracy gain is only
one image out of 219 and should not be treated as conclusive without seed
replication.

## Stage 4: optimizer recipe

Optimizer and weight decay are now configurable from YAML. Compare the winning
eight-epoch Adafactor run against these AdamW recipes:

| Config/result | Optimizer | Learning rate | Weight decay |
| --- | --- | ---: | ---: |
| Stage 3 `stage3_ep8_bs16_lr_1e-4` result | Adafactor | `1e-4` | `0.0` |
| `stage4_adamw_lr_1e-5_wd_0.1.yaml` | AdamW | `1e-5` | `0.1` |
| `stage4_adamw_lr_5e-6_wd_0.1.yaml` | AdamW | `5e-6` | `0.1` |

Run the two new configurations sequentially with:

```bash
for config in open-clip/configs/ablations/stage4_adamw_*.yaml; do
  python open-clip/main_finetune.py --config "$config" --device cuda || break
done
```

AdamW uses betas `(0.9, 0.98)` and epsilon `1e-6`; bias, normalization,
scalar, and logit-scale parameters are excluded from weight decay. Compare all
three recipes on the same held-out classification metrics.

Stage 4 is complete. Metrics below use each run's best-validation-loss
checkpoint:

| Recipe | Best epoch | Validation loss | Accuracy | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adafactor, `1e-4` | 8 | 1.1829 | 0.8539 | 0.8295 | 0.8177 |
| AdamW, `1e-5` | 5 | **1.0771** | **0.8813** | 0.8521 | **0.8509** |
| AdamW, `5e-6` | 4 | 1.1385 | 0.8676 | **0.8573** | 0.8489 |

Both AdamW runs began overfitting after their best checkpoint. AdamW at `1e-5`
is the provisional winner by validation loss, accuracy, and macro F1. The
`5e-6` recipe remains a close second and has the highest weighted F1.

## Confirmation seeds

The two AdamW recipes were rerun with seeds 123 and 2026; the seed-42 Stage 4
runs provide the first replicate:

| Config | Recipe | Seed |
| --- | --- | ---: |
| `confirm_adamw_lr_1e-5_wd_0.1_seed123.yaml` | AdamW, `1e-5` | 123 |
| `confirm_adamw_lr_1e-5_wd_0.1_seed2026.yaml` | AdamW, `1e-5` | 2026 |
| `confirm_adamw_lr_5e-6_wd_0.1_seed123.yaml` | AdamW, `5e-6` | 123 |
| `confirm_adamw_lr_5e-6_wd_0.1_seed2026.yaml` | AdamW, `5e-6` | 2026 |

The command used to run all four sequentially was:

```bash
for config in open-clip/configs/ablations/confirm_adamw_*.yaml; do
  python open-clip/main_finetune.py --config "$config" --device cuda || break
done
```

For each checkpoint, classification evaluation reconstructed its held-out
subset using that run's seed. The official test split remained untouched during
model selection.

The confirmation runs are complete. Each row below evaluates the checkpoint
selected by validation loss on the 219-image held-out subset generated by that
run's seed:

| Recipe | Seed | Best epoch | Validation loss | Accuracy | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW, `1e-5` | 42 | 5 | 1.0771 | 0.8813 | 0.8521 | 0.8509 |
| AdamW, `1e-5` | 123 | 4 | 1.1424 | 0.8858 | 0.8809 | 0.8917 |
| AdamW, `1e-5` | 2026 | 4 | 1.0414 | 0.8676 | 0.8552 | 0.8564 |
| AdamW, `5e-6` | 42 | 4 | 1.1385 | 0.8676 | 0.8573 | 0.8489 |
| AdamW, `5e-6` | 123 | 4 | 1.1688 | 0.8630 | 0.8502 | 0.8353 |
| AdamW, `5e-6` | 2026 | 4 | 1.0910 | 0.8539 | 0.8429 | 0.8385 |

Aggregate results are mean ± sample standard deviation across seeds 42, 123,
and 2026:

| Recipe | Validation loss | Accuracy | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| AdamW, `1e-5` | **1.0870 ± 0.0512** | **0.8782 ± 0.0095** | **0.8627 ± 0.0158** | **0.8663 ± 0.0221** |
| AdamW, `5e-6` | 1.1328 ± 0.0392 | 0.8615 ± 0.0070 | 0.8501 ± 0.0072 | 0.8409 ± 0.0071 |

The selected recipe is therefore OpenCLIP `ViT-B-32` initialized from
`laion2b_s34b_b79k`, fully fine-tuned with batch size 16, AdamW at `1e-5`,
weight decay 0.1, eight configured epochs, 10% linear warmup, and checkpoint
selection by held-out contrastive validation loss. Across these runs the
selected checkpoint occurred at epoch four or five. Because the seed changes
both training randomness and the deterministic 80/20 split, the variability
reported here includes both effects. With only three replicates, treat the
comparison as confirmation of the observed trend rather than a formal
significance test.

The canonical selected configuration is `final_outcome.yaml`. It uses seed 42
as the reproducible default and writes to
`results/finetune_ablations/final_outcome`. Run it with:

```bash
python open-clip/main_finetune.py \
  --config open-clip/configs/ablations/final_outcome.yaml \
  --device cuda
```

## Checkpoint limitation

The trainer currently retains only `ckp_best.pth` according to validation loss
and `ckp_last.pth`. Before using downstream validation accuracy to select an
epoch, add optional per-epoch checkpoint retention; otherwise intermediate
models cannot be evaluated after training.
