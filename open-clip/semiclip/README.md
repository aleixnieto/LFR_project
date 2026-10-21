# SemiCLIP on RSICD

This workflow implements the two-stage method from the [SemiCLIP ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/2058b90b1e0699770840471fc294d1e7-Paper-Conference.pdf), while matching output-affecting behavior in the [released implementation](https://github.com/Gank0078/SemiCLIP). It targets zero-shot RSICD classification with the repository's `ViT-B-32 / laion2b_s34b_b79k` backbone.

## Setup

Concept mining requires NLTK's English tokenizer and tagger data. Install them once in the active environment:

```bash
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
```

The workflow validates these resources at startup and reports this command if they are missing. It never downloads them automatically.

The default recipe is defined in [`configs/semiclip_rsicd.yaml`](configs/semiclip_rsicd.yaml). Paths in that file are resolved relative to the YAML file.

## Training

Run both stages with the default 25 Stage-1 epochs and 15 Stage-2 epochs:

```bash
python open-clip/semiclip/main.py \
  --stage all \
  --device cuda
```

The default output directory is `results/open_clip_semiclip`. Use `--output-dir` to keep separate experiments.

Run Stage 1 only:

```bash
python open-clip/semiclip/main.py \
  --stage 1 \
  --device cuda
```

Stage 1 mines nouns only from the 875 labeled-training images, trains the labeled CLIP and concept objectives, writes best and final checkpoints, and caches fixed weak-view concept logits for the configured unlabeled images. Stage 2 intentionally consumes `stage1_final.pt`, not `stage1_best.pt`.

Run Stage 2 later from the artifacts in the configured output directory:

```bash
python open-clip/semiclip/main.py \
  --stage 2 \
  --device cuda
```

Stage 2 restores the final Stage-1 model and classifier, initializes its learnable prompt after restoration, and creates a fresh optimizer and cosine scheduler `stage2_epochs` is the total target epoch count, not the number of additional epochs.

Use a different configuration with `--config path/to/config.yaml`.

## Bounded smoke run

This exercises both stages and every artifact handoff while limiting data and updates:

```bash
python open-clip/semiclip/main.py \
  --stage all \
  --device cuda \
  --stage1-epochs 1 \
  --stage2-epochs 1 \
  --max-stage1-train-batches 2 \
  --max-stage2-train-batches 2 \
  --max-validation-batches 1 \
  --max-unlabeled-samples 64 \
  --output-dir results/open_clip_semiclip_smoke
```

Use `--device cpu` for debugging when CUDA is unavailable. `--max-unlabeled-samples` limits the ordered image-only dataset consistently for the Stage-1 cache and Stage-2 loader; captions remain unused.

## Artifacts

The run directory contains:

| Artifact | Contents |
| --- | --- |
| `config.yaml`, `resolved_config.json` | Source configuration and fully resolved effective settings. |
| `concepts.json` | Ordered vocabulary, frequencies, per-image labels, labeled split filenames, and split hashes. |
| `stage1_best.pt` | Lowest-validation-loss Stage-1 state. |
| `stage1_final.pt` | Final Stage-1 model, classifier, optimizer, scheduler, vocabulary, configuration, metrics, and RNG state used for Stage 2. |
| `stage1_unlabeled_logits.pt` | Fixed concept logits keyed by unlabeled filename plus exposure hash and Stage-1 provenance. |
| `stage1_metrics.json` | Per-epoch CLIP/concept losses, validation, learning rate, scales, and exposure hashes. |
| `stage2_last.pt` | Resumable checkpoint from the last completed Stage-2 epoch. |
| `stage2_final.pt` | Final OpenCLIP `model_state_dict` plus classifier, prompt, optimizer, scheduler, vocabulary, configuration, provenance, metrics, and RNG state. |
| `stage2_metrics.json` | Per-epoch four-part loss, validation, learning rate, selections, concept confidence, scales, and exposure hashes. |

`stage2_final.pt` uses the same wrapped `model_state_dict` format accepted by `main_zeroshot_compare.py` with `checkpoint_type: finetune`.

## Plotting

Create the Stage-1 loss, Stage-2 loss, learning-rate, selection-count, and logit-scale charts:

```bash
python open-clip/semiclip/plot.py results/open_clip_semiclip \
  --output-dir results/open_clip_semiclip/plots
```

The plotting utility is headless and reads `stage1_metrics.json` and `stage2_metrics.json` from the run directory.

## Zero-shot comparison

The provided [`zeroshot_compare_config.yaml`](zeroshot_compare_config.yaml) lists the original OpenCLIP baseline, the supervised fine-tuned checkpoint, and `stage2_final.pt`. All are evaluated only on the configured RSICD `test` split:

```bash
python open-clip/main_zeroshot_compare.py \
  --models-config open-clip/semiclip/configs/zeroshot_compare_config.yaml \
  --device cuda \
  --output-dir evaluations/semiclip_zeroshot_comparison
```

The existing evaluator writes accuracy, weighted precision/recall/F1, per-class metrics, predictions, individual and aggregate confusion matrices, and comparison plots.

## Benchmark results

These results were produced on 2026-10-21 with seed 42. Every classification result uses the same 1,093-image, 31-class RSICD `test` split, class prompts, preprocessing, and zero-shot evaluator. Values are from a single training seed unless stated otherwise.

The principal run uses [`configs/semiclip_rsicd_paper_1gpu.yaml`](configs/semiclip_rsicd_paper_1gpu.yaml): `ViT-B-32 / laion2b_s34b_b79k`, 875 labeled images, 8,734 unlabeled images, labeled/unlabeled half-batches of 32, 25 Stage-1 epochs, 15 Stage-2 epochs, AdamW at `5e-5`, paper weight decay `5e-4`, ten warmup steps, and cosine decay. This differs from the generic configuration's released-code weight decay of `0.2`.

### Zero-shot classification

| Model | Checkpoint selection | Accuracy | Weighted precision | Weighted recall | Weighted F1 | Correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original LAION OpenCLIP | Pretrained | 62.67% | 61.60% | 62.67% | 59.09% | 685/1,093 |
| Existing supervised, random caption | Best caption-validation loss | 82.25% | 84.40% | 82.25% | 81.35% | 899/1,093 |
| Controlled supervised, random caption | Best caption-validation loss, epoch 3 | 73.47% | 76.88% | 73.47% | 72.29% | 803/1,093 |
| Controlled supervised, random caption | Final, epoch 25 | 82.07% | 82.46% | 82.07% | 80.60% | 897/1,093 |
| SemiCLIP Stage 1 | Best caption-validation loss, epoch 3 | 71.09% | 77.10% | 71.09% | 70.21% | 777/1,093 |
| SemiCLIP Stage 1 | Final, epoch 25 | 84.54% | 85.94% | 84.54% | 84.45% | 924/1,093 |
| SemiCLIP Stage 2 | Final, epoch 15 | 85.64% | 86.58% | 85.64% | **85.31%** | 936/1,093 |
| Existing supervised, all unique captions | Best caption-validation loss | **86.28%** | **88.35%** | **86.28%** | 84.96% | **943/1,093** |

The controlled supervised run uses [`configs/supervised_random_control_paper_1gpu.yaml`](configs/supervised_random_control_paper_1gpu.yaml). It matches the configured labeled-data side of SemiCLIP Stage 1: the split and seed, random-caption policy, transforms, 32-image training batch, 64-image validation batch, dropped incomplete training batch, shuffled loading, 25 epochs, optimizer, learning rate, weight decay, warmup, and scheduler. It omits concept supervision and never loads the 8,734 unlabeled images.

The existing random-caption run and controlled final run are nearly tied: 82.25% versus 82.07% accuracy. The SemiCLIP improvement therefore cannot be explained by the changed optimizer recipe alone.

### Controlled contribution breakdown

Final checkpoints provide the meaningful controlled comparison because Stage 2 is initialized from `stage1_final.pt`, and the released workflow reports the final Stage-2 model.

| Transition | Accuracy change | Weighted-F1 change | Prediction changes | Exact paired p-value |
| --- | ---: | ---: | --- | ---: |
| Controlled supervised final → SemiCLIP Stage-1 final | +2.47 pp | +3.85 pp | 80 corrected, 53 regressed | 0.0238 |
| SemiCLIP Stage-1 final → SemiCLIP Stage-2 final | +1.10 pp | +0.86 pp | 54 corrected, 42 regressed | 0.2615 |
| Controlled supervised final → complete SemiCLIP | **+3.57 pp** | **+4.71 pp** | 89 corrected, 50 regressed | **0.00119** |

These are exact two-sided paired McNemar/binomial tests over the fixed test predictions. They measure paired test-set differences for this run; they do not account for training-seed variance. Stage 1 supplies most of the observed gain. Stage 2 is positive for this seed, but its incremental gain is not independently significant at the 0.05 level.

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

The detailed prediction files are the source for the paired tests above. The JSON metric files retain full precision; README percentages are rounded to two decimals.

## Released-code compatibility choices

These settings intentionally follow the released code when it differs from or is more specific than the paper text:

- semantic pseudo-labels use the top 3 cached concepts;
- surrogate captions use the top 4 cached concepts;
- each top-4 concept receives three learnable context vectors initialized from `a photo includes`;
- selection retains `floor(0.3 × unlabeled_batch_size)` pairs with the highest current weak-image/surrogate cosine similarity within each batch;
- Stage 2 predicts concepts from the strong view but uses the current weak view for pair selection and trapezoidal consistency;
- the Stage-1 cached weak-view logits remain fixed throughout Stage 2;
- AdamW uses weight decay `0.2`, betas `(0.9, 0.98)`, epsilon `1e-6`, ten warmup steps, and cosine decay;
- all four Stage-2 objectives have unit weight by default;
- Stage 2 starts from the final Stage-1 checkpoint with a fresh optimizer.

Prompt construction derives token placement and embedding width from the configured tokenizer and model. It does not retain the released code's literal token IDs or fixed embedding dimensions.

The current implementation covers zero-shot classification. Retrieval evaluation remains out of scope.
