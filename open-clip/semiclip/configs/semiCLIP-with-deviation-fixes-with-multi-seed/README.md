# SemiCLIP paper/code-deviation variants

This directory defines reproducible one-GPU experiments that separate two
implementation discrepancies between the SemiCLIP paper and its released code:

1. the number of concepts used for Stage-2 pseudo-targets and surrogate
   captions; and
2. the presence and treatment of a separate concept-classifier logit scale.

All variants use the project RSICD split and `ViT-B-32 / laion2b_s34b_b79k`.
They are therefore not direct reproductions of the paper's full RS-ALL,
four-GPU, ViT-B-16/OpenAI experiment. Their purpose is a controlled comparison
of paper-aligned and release-aligned SemiCLIP behavior under one shared local
setup.

## Shared training setup

Every SemiCLIP YAML uses the following settings unless its batch-geometry
suffix says otherwise.

| Setting | Value | Motivation |
| --- | --- | --- |
| Seeds | 42, 123, 2026 | Report variation over three deterministic runs. Each seed controls initialization, caption sampling, shuffle order, and augmentation randomness. |
| Labelled split | Fixed with `split_seed: 42` | All runs use the same derived labelled train/validation filenames, preserving a paired comparison across training seeds. |
| Backbone | OpenCLIP ViT-B/32, `laion2b_s34b_b79k` | Matches the project-wide backbone and checkpoint. |
| Data | 875 labelled images from the derived `val` subset; 8,734 image-only `train` images | Preserves the project data protocol and keeps the test split evaluation-only. |
| Caption policy | `random_one` and `all_unique` | The all-unique family changes only caption selection; it uses the same per-seed split and training setup. |
| Optimizer | AdamW, LR \(5\cdot10^{-5}\), weight decay \(5\cdot10^{-4}\), betas \((0.9,0.98)\), epsilon \(10^{-6}\) | Uses the paper/reference SemiCLIP recipe. |
| Schedule | 25 Stage-1 epochs, then 15 Stage-2 epochs; fresh cosine schedule with 10 warm-up steps per stage | Matches the SemiCLIP reference schedule. |
| Loss weights | all one | Preserves the released-code default weighting. |

### Batch-geometry comparison

The new batch suffixes create the direct one-GPU comparison requested after
the memory smoke test. `total_batch_size` is split evenly into labelled
\(B_L\) and unlabelled \(B_U\) images.

| Suffix | `total_batch_size` | \(B_L\) | \(B_U\) | \(S=\lfloor0.3B_U\rfloor\) | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `b32` | 64 | 32 | 32 | 9 | Original one-GPU/local batch geometry. |

The former `effective4gpu` configurations (128 labelled plus 128 unlabelled
images per update) were removed because they cannot fit on this GPU with the
current direct implementation. They remain a reference geometry in the
comparison below, not runnable local configurations.

#### What `b32` changes relative to the released four-GPU code and paper

The released implementation was run with four processes, each receiving 32
labelled and 32 unlabelled images. It gathers features across processes before
the Stage-2 consistency calculation. Consequently, one released-code update
has \(B_L=B_U=128\), selects \(S=\lfloor0.3\cdot128\rfloor=38\)
pseudo-pairs, and evaluates the consistency matrices on \(128+38=166\) rows.
`b32` has no cross-GPU gather: an update has \(B_L=B_U=32\), selects only
\(S=9\), and its matrices have \(32+9=41\) rows.

| Aspect | One-GPU `b32` | Released code on four GPUs | Paper formulation |
| --- | --- | --- | --- |
| Images contributing to one update | 32 labelled + 32 unlabelled | 128 + 128 after cross-process gathering | Defines the losses over the labelled set and selected pseudo-pair set; it does not prescribe this one-GPU sampling geometry. |
| Pseudo-pair selection | Top 9 from the current 32 unlabelled images | Top 38 from the four-process gathered batch of 128 | Top 30% selected from the whole \(D_U\), rather than from a mini-batch. |
| Consistency support | 41 by 41 similarity matrices | 166 by 166 gathered similarity matrices | Eq. (5) is written over the full labelled-plus-pseudo-pair set, not a particular mini-batch. |
| Code scaling of consistency terms | Multiplies their entrywise mean by \(B_L=32\) | Multiplies by global \(B_L=128\) | Does not specify the released implementation's batch-level \(B_L\) multiplier. |
| Updates at the same epoch count | Approximately four times as many, because the global batch is four times smaller | One update per gathered 128+128 images | Does not resolve this one-GPU versus distributed update-frequency choice. |

Thus `b32` is a valid, deterministic **one-GPU approximation**, but it is not
equivalent to either reference: its pseudo-pairs compete only locally, its
consistency gradients are based on fewer relationships and have a four-times
smaller explicit \(B_L\) multiplier, and its optimiser/scheduler see about
four times as many steps per epoch. The differing top-k and classifier-scale
choices documented below are independent of this batch-geometry difference.

## Determinism

Both SemiCLIP and the Stage-1 control entry points seed Python, NumPy, PyTorch,
and CUDA. They use seeded DataLoader generators, enable PyTorch deterministic
algorithms, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disable cuDNN benchmarking
and TF32, and enable deterministic cuDNN operations. This makes a repeated
run with the same seed reproducible on the same hardware and software stack.

Different seeds intentionally produce different split assignments and
training trajectories. Exact bitwise agreement is not promised across GPU
models, driver/CUDA/PyTorch versions, or changed data files.

## Used SemiCLIP config

### `paper_aligned_semiclip_*`

These are the practical paper-aligned runs:

```yaml
concept_loss_top_k: 4
surrogate_top_k: 4
classifier_logit_scale_mode: fixed_one
```

The paper uses one top-\(k\) definition for pseudo-labels and surrogate
captions, with default \(k=4\). It does not specify a separate learned
classifier scale. `fixed_one` initializes the classifier log-scale to zero,
does not optimize it, and therefore implements \(h_\psi(v)=Wv\). The CLIP
logit scale remains learned and clamped to preserve normal pretrained-OpenCLIP
fine-tuning; this is a practical paper alignment, not the more disruptive
literal interpretation that fixes CLIP’s own temperature to one.

## Outputs

Writes to a unique subdirectory of
`results/semiclip-with-deviation-fixes-with-multi-seed/`. Random-one runs write to
`paper_aligned_b32_seed{42,123,2026}` and all-unique runs write to
`paper_aligned_all_unique_b32_seed{42,123,2026}`.

The YAML files are one directory deeper than the normal SemiCLIP config
directory. Their `checkpoint_path`, `image_dir`, and `json_path` therefore use
`../../../../` so they continue to resolve to the repository-level resources.

## Running a configuration

For example, run the seed-42 practical paper-aligned `b32` variant with:

```bash
python open-clip/semiclip/main.py \
  --config open-clip/semiclip/configs/deviation-fixes-with-multi-seed/paper_aligned_semiclip_rsicd_b32_seed42.yaml \
  --device cuda
```
