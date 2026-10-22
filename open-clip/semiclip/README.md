# SemiCLIP on RSICD

## Benchmark results

These results were produced on 2026-10-21 with seed 42. Every classification result uses the same 1,093-image, 31-class RSICD `test` split, class prompts, preprocessing, and zero-shot evaluator. Values are from a single training seed unless stated otherwise.

The principal run uses [`configs/semiclip_rsicd_paper_1gpu.yaml`](configs/semiclip_rsicd_paper_1gpu.yaml): `ViT-B-32 / laion2b_s34b_b79k`, 875 labeled images, 8,734 unlabeled images, labeled/unlabeled half-batches of 32, 25 Stage-1 epochs, 15 Stage-2 epochs, AdamW at `5e-5`, paper weight decay `5e-4` (this differs from the generic configuration's released-code weight decay of `0.2`), ten warmup steps, and cosine decay.

The controlled supervised run uses [`configs/supervised_random_control_paper_1gpu.yaml`](configs/supervised_random_control_paper_1gpu.yaml). It matches the configured labeled-data side of SemiCLIP Stage 1: the split and seed, random-caption policy, transforms, 32-image training batch, 64-image validation batch, dropped incomplete training batch, shuffled loading, 25 epochs, optimizer, learning rate, weight decay, warmup, and scheduler. It omits concept supervision and never loads the 8,734 unlabeled images.

*These settings intentionally follow the released code when it differs from or is more specific than the details provided by the paper text.*

### Zero-shot classification

| Model | Checkpoint selection | Accuracy | Weighted precision | Weighted recall | Weighted F1 | Correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original LAION OpenCLIP | Pretrained (no fine-tuning) | 62.67% | 61.60% | 62.67% | 59.09% | 685/1,093 |
| Existing supervised, random caption (from [here!](open-clip/configs/experiment_caption_strategy/README.md)) | Best caption-validation loss | 82.25% | 84.40% | 82.25% | 81.35% | 899/1,093 |
| Existing supervised, all unique captions (also from [here!](open-clip/configs/experiment_caption_strategy/README.md)) | Best caption-validation loss | **86.28%** | **88.35%** | **86.28%** | 84.96% | **943/1,093** |
| Controlled supervised, random caption | Best caption-validation loss, epoch 3 | 73.47% | 76.88% | 73.47% | 72.29% | 803/1,093 |
| Controlled supervised, random caption | Final, epoch 25 | 82.07% | 82.46% | 82.07% | 80.60% | 897/1,093 |
| SemiCLIP Stage 1 | Best caption-validation loss, epoch 3 | 71.09% | 77.10% | 71.09% | 70.21% | 777/1,093 |
| SemiCLIP Stage 1 | Final, epoch 25 | 84.54% | 85.94% | 84.54% | 84.45% | 924/1,093 |
| SemiCLIP Stage 2 | Final, epoch 15 | 85.64% | 86.58% | 85.64% | **85.31%** | 936/1,093 |

### Controlled contribution breakdown

Final checkpoints provide the meaningful controlled comparison because Stage 2 is initialized from `stage1_final.pt`, and the released workflow reports the final Stage-2 model.

| Transition | Accuracy change | Weighted-F1 change | Prediction changes | Exact paired p-value |
| --- | ---: | ---: | --- | ---: |
| Controlled supervised final → SemiCLIP Stage-1 final | +2.47 pp | +3.85 pp | 80 corrected, 53 regressed | 0.0238 |
| SemiCLIP Stage-1 final → SemiCLIP Stage-2 final | +1.10 pp | +0.86 pp | 54 corrected, 42 regressed | 0.2615 |
| Controlled supervised final → complete SemiCLIP | **+3.57 pp** | **+4.71 pp** | 89 corrected, 50 regressed | **0.00119** |

These are exact two-sided paired McNemar/binomial tests over the fixed test predictions. Stage 1 supplies most of the observed gain. Stage 2 is positive for this seed, but its incremental gain is not independently significant at the 0.05 level.

Against the existing random-caption checkpoint, complete SemiCLIP gains 3.39 accuracy points and 3.96 weighted-F1 points. It corrects 91 examples that the supervised model misses while regressing on 54 examples (`p = 0.00267`). Against the all-captions supervised model, SemiCLIP is 0.64 points lower in accuracy but 0.34 points higher in weighted F1; their accuracy difference is not significant in the paired test (`p = 0.592`).

The largest SemiCLIP recall gains over the existing random-caption model are Church (+33.3 pp), Bridge (+30.4 pp), RailwayStation (+23.1 pp), Center (+19.2 pp), and playfields (+18.2 pp). The largest regressions are River (-19.5 pp), Playground (-18.9 pp), Resort (-10.3 pp), and Commercial (-8.6 pp). Consult the saved per-class metrics and confusion matrices before drawing conclusions from classes with only 24–66 test examples.

### Training and validation behavior

Concept mining produces 128 concepts. Of the 875 labeled training images, 856 have at least one retained concept and 19 have none. Images have a mean of 3.60 retained concepts, a median of 3, and a maximum of 12; retained concept frequencies range from 5 to 129 images.

| Run | Best validation loss | Best epoch | Final validation loss | Selected checkpoint used downstream |
| --- | ---: | ---: | ---: | --- |
| Controlled supervised | 2.02783 | 3 | 2.25437 | Final epoch 25 for the matched comparison |
| SemiCLIP Stage 1 | 1.97226 | 3 | 2.56068 | Final epoch 25, as required by Stage 2 |
| SemiCLIP Stage 2 | 1.88540 | 1 | 2.01390 | Final epoch 15 |

Both controlled and SemiCLIP Stage-1 validation losses reach their minima at epoch 3, while their final checkpoints perform much better on zero-shot classification. For example, the controlled best/final accuracies are 73.47%/82.07%, and the Stage-1 best/final accuracies are 71.09%/84.54%. Caption-retrieval validation loss is therefore not a reliable proxy for downstream zero-shot classification in these runs. Both best and final results are reported to avoid selecting a checkpoint using the test set.

Stage-1 training loss falls from 5.90555 to 1.32471 across 25 epochs. Stage 2 finishes with a total training loss of 2.12341. Every Stage-2 update uses 32 labeled and 32 unlabeled images and retains 9 unlabeled image–surrogate pairs through the batch-local top-30% rule.

### Comparison with the paper

The paper reports the following RSICD-CLS results. Its values are means over three seeds, while this repository currently has one seed. The dataset construction and pretraining are also different: the paper trains on RS-ALL (10,911 images drawn from RSICD, UCM, and Sydney), whereas this run uses RSICD alone; this repository uses LAION-pretrained ViT-B-32 rather than the paper's OpenAI-pretrained model.

| ViT-B-32 result | Original | Supervised | SemiCLIP | SemiCLIP minus supervised |
| --- | ---: | ---: | ---: | ---: |
| Paper, three-seed mean | 56.2% | 73.6% | 81.3% | +7.7 pp |
| This repository, seed 42, all-captions comparator | 62.67% | 86.28% | 85.64% | -0.64 pp |
| This repository, seed 42, matched random-caption control | 62.67% | 82.07% | 85.64% | **+3.57 pp** |

The paper's separate single-GPU experiment uses ViT-B-16 and reports 59.2% original, 75.3% supervised, and 83.5% SemiCLIP accuracy, an 8.2-point gain. It is informative about scale but is not architecture-matched to this run.

The absolute SemiCLIP result here, 85.64%, is above the paper's reported ViT-B-32 and single-GPU values. The reproduced incremental benefit is smaller: +3.57 points against the matched random-caption control, compared with +7.7 points in the paper's ViT-B-32 table. SemiCLIP does not beat this repository's stronger all-captions supervised model in accuracy, although it has slightly higher weighted F1.

### Saved benchmark artifacts

- [`evaluations/semiclip_controlled_baseline_comparison`](../../evaluations/semiclip_controlled_baseline_comparison) contains the main metrics, predictions, per-class reports, confusion matrices, and comparison plots.
- [`evaluations/semiclip_stage1_checkpoint_comparison`](../../evaluations/semiclip_stage1_checkpoint_comparison) contains the Stage-1 best/final evaluation.
- [`results/open_clip_semiclip_paper_1gpu_seed42`](../../results/open_clip_semiclip_paper_1gpu_seed42) contains the complete SemiCLIP run.
- [`results/open_clip_supervised_random_control_paper_1gpu_seed42`](../../results/open_clip_supervised_random_control_paper_1gpu_seed42) contains the matched supervised control.

