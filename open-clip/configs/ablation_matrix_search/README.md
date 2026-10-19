# Full-factorial optimizer, learning-rate, and batch-size ablation

### Parameters

**Varied Parameters:**

| Dimension | Values |
| --- | --- |
| Learning rate | `5e-6`, `1e-5`, `3e-5`, `5e-5`, `1e-4` |
| Optimizer | `adafactor`, `adamw` |
| Image batch size | `16`, `32`, `64` |

The complete matrix contains 5 × 2 × 3 = *30 runs*. Every combination has a
separate YAML file and output directory.

**Fixed Parameters:**

The following controls are identical across all runs (fixed assumptions):

| Control | Fixed value |
| --- | --- |
| Model initialization | OpenCLIP `ViT-B-32 / laion2b_s34b_b79k` |
| Caption policy | `all_unique` |
| Text-to-image weighting | `per_caption` |
| Source data | RSICD JSON `val` split |
| Derived split | deterministic 80% train / 20% validation |
| Split and training seed | `42` |
| Weight decay | `0.1` for both optimizers |
| Scheduler | linear |
| Warmup fraction | `0.1` |
| Data-loader workers | `4` |
| Configured epochs | `8` |

### Other Details

Every comparison in this matrix uses the same held-out validation images. Each config takes the captioned RSICD JSON `val` subset and applies the
same deterministic 80/20 image-level partition with seed 42, producing 875 training images and 219 validation images. 

Each configuration writes beneath the new output root:
```text
results/finetune_ablation_matrix_search/
├── bs16_adafactor_lr_5e-6/
├── ...
└── bs64_adamw_lr_1e-4/
```

Each run stores its copied config, `metrics.json`, `train.log`,
`ckp_best.pth`, and `ckp_last.pth`.


## Automated workflow

From the repository root, run training, validation, and report generation in one command:
```bash
source lfr-clip.venv/bin/activate
python open-clip/configs/ablation_matrix_search/run_study.py --stage all --device cuda
```

The stages can instead be run separately:
```bash
# Train all 30 configurations
python open-clip/configs/ablation_matrix_search/run_study.py --stage train --device cuda

# Evaluate every available ckp_best.pth on the shared 219-image validation split
python open-clip/configs/ablation_matrix_search/run_study.py --stage evaluate --device cuda

# Generate the comparison report from the training and evaluation results
python open-clip/configs/ablation_matrix_search/run_study.py --stage report
```

The final report selects a global validation winner only when one configuration leads both zero-shot accuracy and macro F1. 