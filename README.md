# LFR_project

## LFR Project Code

All updated project code is located under `LFR_project/open-clip/`.

All files under `LFR_project/former_project/` are kept unchanged.

## UV venv - Initial Setup

Here we use [UV](https://docs.astral.sh/uv/pip/environments/) for handeling the venv, but you can also use your alternative setup of choise.

1. Install a uv-managed Python if 3.12.9 is not already available: `uv python install 3.12.9`

2. Create the venv: `uv venv --python 3.12.9 lfr-clip.venv`

3. Activate the venv: `source lfr-clip.venv/bin/activate`

4. Install requirements: `uv pip install -r requirements.txt`

After initial setup is done, you only need to make sure that you have your venv activated when running the project.


## Dataset

We assume that we have the following resources obtrained from dataset.zip organized as follows:

```text
LFR_project/
├── former_project/
├── RSICD_images/
├── txtclasses_rsicd/
└── dataset_rsicd.json
```

`dataset_split:` option in config-files controls what data-source is used for the workflow. This source might be used as the source for forming new train/val splits according to the config.

## Running main_zeroshot.py

The evaluator `main_zeroshot.py` reads both its model selection and local RSICD resource locations from `zeroshot_config.yaml`. The configuration binds:

- `model_name`: the OpenCLIP architecture;
- `pretrained_tag`: its registered preprocessing configuration;
- `checkpoint_path`: manually downloaded local weights.
- `image_dir`, `json_path`, and `class_dir`: the local RSICD resources, resolved relative to the YAML file.

To run the zero-shot evaluation, start by downloading a checkpoint into the weights directory.

1. Create the weights directory the first time: `mkdir -p weights/open_clip`

2. Download the checkpoint do the created folder and configure `zeroshot_config.yaml` according to the table below (model dependent). By default, the repo is set up to run wiht the `ViT-B-32` / `laion2b_s34b_b79k` model.

| Model / checkpoint | YAML configuration values | Manual download command |
| --- | --- | --- |
| `ViT-B-32` / `laion2b_s34b_b79k` (default) | `model_name: ViT-B-32`<br>`pretrained_tag: laion2b_s34b_b79k`<br>`checkpoint_path: ../../weights/open_clip/ViT-B-32-laion2b_s34b_b79k.bin` | `curl -L https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin -o weights/open_clip/ViT-B-32-laion2b_s34b_b79k.bin` |

> TODO: Add validated model/tag/configuration/download entries for additional OpenCLIP architectures together with code support for more preprocessing options.

3. We can run the evaluation with some different settings:
    * One-image smoke test: `python open-clip/main_zeroshot.py --max-samples 1`
    * Subset of samples: `python open-clip/main_zeroshot.py --max-samples 32`
    * Full RSICD test split: `python open-clip/main_zeroshot.py`

Results are written to `results/open_clip_zeroshot/` by default; use `--output-dir` to choose another destination.

Runing example below:
```text
(lfr-clip.venv) casko@soro-casko:~/phd-code/LFR_project$ python open-clip/main_zeroshot.py --max-samples 5
Model: ViT-B-32 / laion2b_s34b_b79k
Evaluated samples: 5
Accuracy: 0.8000
Weighted precision: 1.0000
Weighted recall: 0.8000
Weighted F1: 0.8889
Results: results/open_clip_zeroshot
```

Use `python open-clip/main_zeroshot.py --list-pretrained` to list model/tag pairs supported by the installed OpenCLIP version. Listed pairs still need matching local weights and preprocessing suport.

## Image-to-text retrieval (in progress)

### Comparing retrieval checkpoints

Run `python open-clip/main_retrieval_compare.py --device cpu` to evaluate the
models in `open-clip/configs/retrieval_compare_config.yaml` on the same images
and caption pool. All selected checkpoint files must exist locally; missing files
are reported before evaluation starts. Each model uses its own preprocessing and
re-encodes the captions with its own text encoder.

For a baseline smoke test:

```bash
python open-clip/main_retrieval_compare.py --model baseline --max-samples 2 --device cpu
```

The terminal table reports the Accuracy@k cutoffs configured in the YAML with
text-membership matching. Configure the evaluation and example logging using:

```yaml
ks: [1, 3, 5, 10]
num_captions: 10
log_every: 50
```

`ks` is a non-empty list of distinct positive integer cutoffs. `num_captions`
sets how many ranked captions are logged per example, and `log_every` selects
every Nth evaluated image (both must be positive integers). Ranking retains
enough candidates for the largest cutoff and the requested example count,
limited by the available caption pool.
These settings are also supported in the single-model retrieval configuration.
Logs include the examples and accuracy summaries without timestamps:
`comparison.log` in the comparison output directory and `retrieval.log` in the
single-model output directory. Set `--output-dir` to change those directories.

### Single-model retrieval

The entry point is `open-clip/main_retrieval.py`, configured by
`open-clip/configs/retrieval_config.yaml` with `dataset_split: test`.
Paths in the YAML resolve relative to the configuration file.

Run `python open-clip/main_retrieval.py` to encode images and captions using the
local pretrained OpenCLIP checkpoint. Both retrieval entry points remove exact
duplicate caption strings globally from the selected images' candidate pool
before encoding, preserving first-occurrence order. Captions shared by multiple
images appear only once in the ranking; each image's ground-truth caption list
is unchanged.
Each image ranks captions by cosine similarity. A match is
correct when the retrieved text is in the query image's ground-truth caption list.
The script prints each configured Accuracy@k (default: 1, 3, 5, 10): the fraction of images with at least one
matching caption among the top K candidates (using all candidates if fewer than K
are available).

### Comparing models (zero-shot)

`main_zeroshot_compare.py` evaluates and compares the models listed in `open-clip/configs/zeroshot_compare_config.yaml` on the configured test dataset.

`checkpoint_type: openclip` is for a raw OpenCLIP pretrained weights file (such as the baseline `.bin`). `checkpoint_type: finetune` is for this project’s wrapped
fine-tuning checkpoint (such as `ckp_best.pth`).

The output contains `summary.json` and an aggregate `model_comparison.png`, plus per-model `metrics.json`, `predictions.json`, per-class F1 values, individual `confusion_matrix.png` files, and a shared `confusion_matrices_comparison.png`.

The config looks like this:
```yaml
image_dir: ../../RSICD_images
json_path: ../../dataset_rsicd.json
class_dir: ../../txtclasses_rsicd

models:
  - label: baseline
    model_name: ViT-B-32
    pretrained_tag: laion2b_s34b_b79k
    checkpoint_type: openclip
    checkpoint_path: ../../weights/open_clip/ViT-B-32-laion2b_s34b_b79k.bin

  - label: fine_tuned
    model_name: ViT-B-32
    pretrained_tag: laion2b_s34b_b79k
    checkpoint_type: finetune
    checkpoint_path: ../../results/my_finetune_run/ckp_best.pth
```

Run the configured comparison:

```bash
python open-clip/main_zeroshot_compare.py --device cuda --output-dir evaluations/open_clip_zeroshot_comparison
```

Additional Options:
* To evaluate a subset of the listed models, add flags  `--model baseline --model fine_tuned`, one for each model

## Running main_finetune.py

`open-clip/main_finetune.py` fine-tunes the configured model on RSICD **image-caption pairs**. Technical details of the implementation is found below. It uses the same manually downloaded checkpoint described in the zero-shot section, as the baseline policy for fine-tuning.

Caption-less RSICD images are supported by the separate SemiCLIP workflow, see more details below.

The fine-tuning configuration `open-clip/configs/finetune_config.yaml` is organized as follows.

| Field | Purpose |
| --- | --- |
| `model_name`, `pretrained_tag`, `checkpoint_path` | OpenCLIP architecture, its preprocessing configuration, and the local initialization checkpoint. |
| `image_dir`, `json_path`, `dataset_split` | Local RSICD resources and the source JSON split. Fine-tuning uses `val`; the selected source split is divided deterministically into new train/validation subsets. |
| `seed`, `supervised_train_fraction` | Deterministic split settings. The default creates an 80/20 train/validation split with seed 42. |
| `caption_policy` | Training caption selection: `all_unique` or one deterministic, seed-controlled `random_one` sample per image per epoch. |
| `text_to_image_weighting` | Normalize text-to-image loss `per_caption` or `per_image`. |
| `batch_size`, `num_workers`, `epochs`, `learning_rate`, `warmup_fraction` | Training and linear-schedule settings. |
| `optimizer`, `weight_decay` | Optional optimizer recipe. Supported optimizers are `adafactor` (the backward-compatible default) and `adamw`; weight decay defaults to zero. |
| `output_dir` | Default directory for metrics and checkpoints. |

Run a bounded one-epoch smoke test, preferably into a separate output directory:

```bash
python open-clip/main_finetune.py \
  --device cuda \
  --batch-size 1 \
  --max-train-batches 2 \
  --max-validation-batches 1 \
  --output-dir results/open_clip_finetune_smoke
```

Additional options:
* Use `--device cpu` when CUDA is unavailable, or for debugging.
* The `--max-train-batches` and `--max-validation-batches` flags are optional; omit them to use the full configured subsets.
* Use a different config file with `--config path/to/finetune_config.yaml`

To run unrestricted supervised training with the values in the default configuration:

```bash
python open-clip/main_finetune.py --device cuda
```

Current scope is supervised image-caption fine-tuning only. Mixed supervised/unsupervised training, SimCLR, and checkpoint-resume command-line support are deferred.

### Implementation Details

**Loss:**

For a batch of $B$ images and $N$ captions, let $g(n)\in\{1,\ldots,B\}$ map caption $n$ to its source image. The image and text encoders produce L2-normalized features $v_i$ and $t_n$, respectively. With $s=\exp(\texttt{model.logit\_scale})$, the logits are $z_{ni}=s\,t_n^\top v_i$. Let $P_i=\{n\mid g(n)=i\}$ be all captions belonging to image $i$.

With `per_caption`, the caption-to-image term uses the source image as its single target and averages over captions:

$$
\mathcal{L}_{t\rightarrow i}^{caption} = -\frac{1}{N}\sum_{n=1}^{N}
\log\frac{\exp(z_{n,g(n)})}{\sum_{j=1}^{B}\exp(z_{nj})}.
$$

With `per_image`, caption losses are first averaged within each image and then across images:

$$
\mathcal{L}_{t\rightarrow i}^{image} = -\frac{1}{B}\sum_{i=1}^{B}
\frac{1}{|P_i|}\sum_{n\in P_i}
\log\frac{\exp(z_{n,i})}{\sum_{j=1}^{B}\exp(z_{nj})}.
$$

The image-to-caption term assigns equal target probability to every caption of that image:

$$
\mathcal{L}_{i\rightarrow t} = -\frac{1}{B}\sum_{i=1}^{B}
\sum_{n\in P_i}\frac{1}{|P_i|}
\log\frac{\exp(z_{ni})}{\sum_{m=1}^{N}\exp(z_{mi})}.
$$

The optimized supervised loss is:

$$
\mathcal{L}=\frac{1}{2}\left(\mathcal{L}_{t\rightarrow i}+\mathcal{L}_{i\rightarrow t}\right).
$$

**Captions:** Each image's reconstructed caption strings are deduplicated while preserving first-seen order, then tokenized. With `all_unique`, the collate function flattens every retained caption and emits the caption-to-image index map \(g\). With `random_one`, it returns one deterministic sample keyed by seed, epoch, and image index. For example, if RSICD lists captions as:
```
1. "an airport with runways"
2. "aerial view of an airport"
3. "an airport with runways"   # duplicate
4. "airport surrounded by roads"
```
the loader keeps
```
1. "an airport with runways"
2. "aerial view of an airport"
3. "airport surrounded by roads"
```

**Optimization:** All model parameters remain trainable. The default Adafactor recipe uses `scale_parameter=True`, `relative_step=False`, `warmup_init=False`, and the configured learning rate. Selecting `adamw` uses betas `(0.9, 0.98)` and epsilon `1e-6`; bias, normalization, scalar, and logit-scale parameters are excluded from weight decay. The linear scheduler has `floor(warmup_fraction * total_training_steps)` warmup steps; `total_training_steps` accounts for `epochs` and any `--max-train-batches` limit. After every optimizer update, the logit-scale parameter is clamped to \([0, 4.6052]\), i.e. its exponential is at most approximately 100.

**Metrics:** `metrics.json` records the configured training and validation strategies; directional and combined losses; image, caption, batch, and optimizer-step counts; captions per image; and SHA-256 hashes of processed image order. `ckp_best.pth` is selected only by the common all-unique/per-caption `validation.selection_loss`.

**Seeding:** Python, NumPy, PyTorch, and available CUDA RNGs are seeded from `seed`.

**Checkpoint:** `ckp_best.pth` is selected by validation loss, while `ckp_last.pth` is always written. Both include model, optimizer, scheduler, epoch, metrics, and RNG states; the run verifies that the final checkpoint can be restored into a fresh matching model.

### Plotting fine-tuning progress

Use `open-clip/plotting_utils.py` to create headless PNG charts from the
fine-tuning `metrics.json`. It can inspect a completed run or a run still
training, since metrics are updated after every epoch:

```bash
python open-clip/plotting_utils.py results/open_clip_finetune_smoke \
  --output-dir results/open_clip_finetune_smoke/plots
```

The command writes loss, positive text-to-image and image-to-text logits,
logit scale, and learning-rate charts. To compare runs, pass additional run
directories (or direct `metrics.json` paths) and repeat `--label`:

```bash
python open-clip/plotting_utils.py results/run_a results/run_b \
  --label baseline --label tuned --output-dir results/run_comparison_plots
```

## Running SemiCLIP

This workflow implements the two-stage method from the [SemiCLIP ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/2058b90b1e0699770840471fc294d1e7-Paper-Conference.pdf), while matching output-affecting behavior in the [released implementation](https://github.com/Gank0078/SemiCLIP). It targets zero-shot RSICD classification with the repository's `ViT-B-32 / laion2b_s34b_b79k` backbone.

### Extra dependencies
This workflows concept mining requires NLTK's English tokenizer and tagger data. Install them once in the active environment:
```bash
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
```

The workflow validates these resources at startup and reports this command if they are missing. The default recipe is defined in [`configs/semiclip_rsicd.yaml`](configs/semiclip_rsicd.yaml). Paths in that file are resolved relative to the YAML file.

### Run semiCLIP main.py
*Stage 1* mines nouns only from the labeled-training images, trains the labeled CLIP and concept objectives, writes best and final checkpoints, and caches fixed weak-view concept logits for the configured unlabeled images. 

*Stage 2* intentionally consumes `stage1_final.pt`, not `stage1_best.pt`. Stage 2 restores the final Stage-1 model and classifier, initializes its learnable prompt after restoration, and creates a fresh optimizer and cosine scheduler `stage2_epochs` is the total target epoch count, not the number of additional epochs.

**Smoke Run:**
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

**Full Run**

Run both stages with the default 25 Stage-1 epochs and 15 Stage-2 epochs:

```bash
python open-clip/semiclip/main.py \
  --stage all \
  --device cuda
```

Run Stage 1 only:

```bash
python open-clip/semiclip/main.py \
  --stage 1 \
  --device cuda
```

Run Stage 2 from the  Stage 1 artifacts in the configured output directory:

```bash
python open-clip/semiclip/main.py \
  --stage 2 \
  --device cuda
```

**Additional options:**
*  The default output directory is `results open_clip_semiclip`. Use `--output-dir` to keep separate experiments.
* Use a different configuration with `--config path/to/config.yaml`.
* `--max-unlabeled-samples` limits the ordered image-only dataset consistently for the Stage-1 cache and Stage-2 loader
* Use `--device cpu` for debugging when CUDA is unavailable

### semiCLIP Artefacts

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

### Plotting the training progress

Create the Stage-1 loss, Stage-2 loss, learning-rate, selection-count, and logit-scale charts:

```bash
python open-clip/semiclip/plot.py results/open_clip_semiclip \
  --output-dir results/open_clip_semiclip/plots
```

The plotting utility is headless and reads `stage1_metrics.json` and `stage2_metrics.json` from the run directory.

### Implementation Details

**Concept mining:** Concepts are mined only from the labeled-training partition. NLTK tokenizes and part-of-speech tags every caption, after which the workflow keeps nouns, removes the released implementation's generic/noisy noun list, and filters by `concept_min_frequency` and `concept_max_frequency_ratio`. Each image receives a multi-hot target over the retained vocabulary. A concept classifier is initialized from normalized OpenCLIP text embeddings of `"a photo includes {concept}."`; its weights and its own logit scale remain trainable.

**Stage-1 loss:** Each labeled image is paired with one deterministic, seed- and epoch-dependent random caption. For a batch of \(B\) normalized image features \(v_i\) and caption features \(t_i\), with \(s=\exp(\texttt{model.logit\_scale})\), Stage 1 uses the ordinary symmetric one-to-one CLIP loss

$$
\mathcal{L}_{\mathrm{CLIP}}=
\frac{1}{2B}\sum_{i=1}^{B}\left[
-\log\frac{\exp(sv_i^\top t_i)}{\sum_j\exp(sv_i^\top t_j)}
-\log\frac{\exp(st_i^\top v_i)}{\sum_j\exp(st_i^\top v_j)}
\right].
$$

Let \(a_{ic}\) be the concept-classifier logit and let \(y_{ic}\) be the image's multi-hot concept target normalized to sum to one (an image with no retained concepts contributes zero to this term). The semantic concept loss is

$$
\mathcal{L}_{\mathrm{concept}}=-\frac{1}{B}\sum_{i=1}^{B}\sum_{c=1}^{C}
y_{ic}\log\operatorname{softmax}(a_i)_c.
$$

The optimized Stage-1 objective is `clip_loss_weight` \(\mathcal{L}_{\mathrm{CLIP}}\) plus `concept_loss_weight` \(\mathcal{L}_{\mathrm{concept}}\). After the final Stage-1 epoch, the model computes and stores filename-keyed concept logits for every unlabeled image using the fixed weak OpenCLIP view. Stage 2 always uses this cache and `stage1_final.pt`; the validation-selected `stage1_best.pt` is retained only as a separate artifact.

**Stage-2 pseudo-labels and surrogate captions:** Every update pairs one labeled half-batch with one unlabeled half-batch. For each unlabeled image, the top `concept_loss_top_k` classes in its cached Stage-1 logits form a uniform pseudo-target for the strong augmented view. Independently, the top `surrogate_top_k` cached concepts are placed into a surrogate caption. Three trainable context vectors, initialized from the text embedding of `"a photo includes"`, precede each concept; the resulting token embeddings are passed through the OpenCLIP text tower. The prompt, concept classifier, and all OpenCLIP parameters are trainable.

**Stage-2 loss:** The labeled half-batch retains the Stage-1 CLIP term, while strong unlabeled views receive the pseudo-concept cross-entropy term. Unlabeled weak-view features and their surrogate-caption features are ranked by cosine similarity, and only the batch-local top \(\lfloor\texttt{selection\_ratio}\,U\rfloor\) pairs are used for consistency learning. If \(V\) and \(T\) concatenate the labeled features with those selected unlabeled image and surrogate features, the implementation adds an intramodal penalty aligning \(sVV^\top\) with \(sTT^\top\), and a crossmodal penalty making \(sVT^\top\) symmetric. The complete objective is

$$
\mathcal{L}_{\mathrm{stage2}}=
\lambda_{\mathrm{clip}}\mathcal{L}_{\mathrm{CLIP}}+
\lambda_{\mathrm{concept}}\mathcal{L}_{\mathrm{concept}}+
\lambda_{\mathrm{intra}}\mathcal{L}_{\mathrm{intra}}+
\lambda_{\mathrm{cross}}\mathcal{L}_{\mathrm{cross}},
$$

where the four weights are the corresponding `*_loss_weight` configuration values. The consistency terms divide out \(s^2\) and use the labeled half-batch size as the released implementation's scale factor.

**Augmentation and batching:** Labeled training uses the normal OpenCLIP preprocessing. Each unlabeled Stage-2 sample supplies that weak view plus a strong view consisting of resize, reflected padded random crop, random horizontal flip, and the released FixMatch-style RandAugment. Incomplete labeled and Stage-2 unlabeled batches are dropped; the unlabeled iterator is restarted if the labeled loader is longer.

**Optimization, validation, and reproducibility:** Both stages use AdamW with the configured betas and epsilon, excluding bias, normalization, scalar, and logit-scale parameters from weight decay. Each stage creates its own optimizer and cosine schedule with the configured warmup; Stage 2 never reuses Stage-1 optimizer state. Model and classifier logit scales are clamped after every update so their exponentials are at most approximately 100. Validation uses every unique caption and the symmetric multi-positive CLIP loss. Metrics include component losses, learning rate, scales, processed counts, selection statistics, and SHA-256 hashes of image order. Checkpoints include model/module, optimizer, scheduler, configuration, history, provenance, and RNG states; Python, NumPy, PyTorch, and available CUDA RNGs are seeded from `seed`.

## Retrieval evaluation

### Image-to-text retrieval


Run `python open-clip/main_retrieval_compare.py --device cuda` to evaluate the
models in `open-clip/configs/retrieval_compare_config.yaml` on the same images
and caption pool. All selected checkpoint files must exist locally; missing files
are reported before evaluation starts. 

```bash
python open-clip/main_retrieval_compare.py --model baseline --device cuda
```

The terminal table reports the configured Accuracy@k cutoffs (currently 1, 3, 5,
and 10) with text-membership matching.

The latest full test run used 1,093 images and 3,331 globally unique captions.
Accuracy values are fractions; the best value in each column is bold.

| Model | Accuracy@1 | Accuracy@3 | Accuracy@5 | Accuracy@10 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0631 | 0.1583 | 0.2260 | 0.3541 |
| supervised_all_unique | 0.1006 | 0.2516 | **0.3614** | **0.5380** |
| semiclip | **0.1134** | **0.2543** | 0.3568 | 0.5142 |

See [image-to-text retrieval results](open-clip/retrieval-results.md) for the
comparison table, metric definition, and findings.


## Experiments

See separate readme files under each config folder.

Current experiments:
1. Ablation study for fine-tuning hyper-parameters: [Readme here!](open-clip/configs/ablations/README.md)
2. Evaluation of caption usage strategies: [Readme here!](open-clip/configs/experiment_caption_strategy/README.md)
3. Two-stage SemiCLIP training with unlabeled RSICD images: [Readme here!](open-clip/semiclip/README.md)
4. Image-to-text retrieval comparison: [Results here](open-clip/retrieval-results.md)
