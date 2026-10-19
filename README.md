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

### Comparing selected zero-shot models

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

> TODO: Implement and document a fine-tuning setup that can make use of the caption-less images for fine-tuning.

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

## Experiments

See separate readme files under each config folder.