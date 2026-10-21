# SemiCLIP on RSICD

This workflow implements the two-stage method from the [SemiCLIP ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/2058b90b1e0699770840471fc294d1e7-Paper-Conference.pdf), while matching output-affecting behavior in the [released implementation](https://github.com/Gank0078/SemiCLIP). It targets zero-shot RSICD classification with the repository's `ViT-B-32 / laion2b_s34b_b79k` backbone.

## Data boundaries

The default configuration uses these mutually exclusive RSICD partitions:

| Source split | Images | Use |
| --- | ---: | --- |
| `val` | 1,094 | Deterministically split with seed 42 into 875 labeled training and 219 validation images. |
| `train` | 8,734 | Unlabeled images for Stage 1 caching and Stage 2 consistency training. |
| `test` | 1,093 | Reserved for zero-shot evaluation only. |

The unlabeled paths instantiate `RSICDImageOnlyDataset`. They return transformed images and filenames and have no tokenizer or caption access. The shared JSON parser necessarily parses the metadata document containing the caption fields, but SemiCLIP never reads, tokenizes, returns, or trains on the `train` captions.

## Setup

Install the repository requirements and place the manually downloaded OpenCLIP checkpoint at:

```text
weights/open_clip/ViT-B-32-laion2b_s34b_b79k.bin
```

Concept mining requires NLTK's English tokenizer and tagger data. Install them once in the active environment:

```bash
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
```

The workflow validates these resources at startup and reports this command if they are missing. It never downloads them automatically.

The default recipe is defined in [`semiclip_rsicd.yaml`](semiclip_rsicd.yaml). Paths in that file are resolved relative to the YAML file.

## Training

Run both stages with the default 25 Stage-1 epochs and 15 Stage-2 epochs:

```bash
python open-clip/main_semiclip.py \
  --stage all \
  --device cuda
```

The default output directory is `results/open_clip_semiclip`. Use `--output-dir` to keep separate experiments.

Run Stage 1 only:

```bash
python open-clip/main_semiclip.py \
  --stage 1 \
  --device cuda
```

Stage 1 mines nouns only from the 875 labeled-training images, trains the labeled CLIP and concept objectives, writes best and final checkpoints, and caches fixed weak-view concept logits for the configured unlabeled images. Stage 2 intentionally consumes `stage1_final.pt`, not `stage1_best.pt`.

Run Stage 2 later from the artifacts in the configured output directory:

```bash
python open-clip/main_semiclip.py \
  --stage 2 \
  --device cuda
```

Stage 2 restores the final Stage-1 model and classifier, initializes its learnable prompt after restoration, and creates a fresh optimizer and cosine scheduler.

Resume an interrupted Stage-2 run from its last completed epoch:

```bash
python open-clip/main_semiclip.py \
  --stage 2 \
  --device cuda \
  --resume-stage2 results/open_clip_semiclip/stage2_last.pt
```

`stage2_epochs` is the total target epoch count, not the number of additional epochs. Resume validates the concept vocabulary and hashes of the exact Stage-1 checkpoint and cache before restoring model, prompt, optimizer, scheduler, and RNG state. Resume is accepted only with `--stage 2`.

Use a different configuration with `--config path/to/config.yaml`. The `--stage1-epochs` and `--stage2-epochs` options override their respective configured values for the current run.

## Bounded smoke run

This exercises both stages and every artifact handoff while limiting data and updates:

```bash
python open-clip/main_semiclip.py \
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
python open-clip/plot_semiclip.py results/open_clip_semiclip \
  --output-dir results/open_clip_semiclip/plots
```

The plotting utility is headless and reads `stage1_metrics.json` and `stage2_metrics.json` from the run directory.

## Zero-shot comparison

The provided [`zeroshot_compare_config.yaml`](zeroshot_compare_config.yaml) lists the original OpenCLIP baseline, the supervised fine-tuned checkpoint, and `stage2_final.pt`. All are evaluated only on the configured RSICD `test` split:

```bash
python open-clip/main_zeroshot_compare.py \
  --models-config open-clip/configs/semiclip/zeroshot_compare_config.yaml \
  --device cuda \
  --output-dir evaluations/semiclip_zeroshot_comparison
```

The existing evaluator writes accuracy, weighted precision/recall/F1, per-class metrics, predictions, individual and aggregate confusion matrices, and comparison plots.

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
