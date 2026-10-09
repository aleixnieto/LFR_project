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

## Running main_zeroshot.py

The evaluator `open-clip/main_zeroshot.py` reads both its model selection and local RSICD resource locations from `open-clip/zeroshot_config.yaml`. The configuration binds:

- `model_name`: the OpenCLIP architecture;
- `pretrained_tag`: its registered preprocessing configuration;
- `checkpoint_path`: manually downloaded local weights.
- `image_dir`, `json_path`, and `class_dir`: the local RSICD resources, resolved relative to the YAML file.

To run the zero-shot evaluation, start by downloading a checkpoint into the weights directory.

1. Create the weights directory the first time: `mkdir -p weights/open_clip`

2. Download the checkpoint do the created folder and configure the `zeroshot_config.yaml` according to the table below (model dependent). By default, the repo is set up to run wiht the `ViT-B-32` / `laion2b_s34b_b79k` model.

| Model / checkpoint | YAML configuration values | Manual download command |
| --- | --- | --- |
| `ViT-B-32` / `laion2b_s34b_b79k` (default) | `model_name: ViT-B-32`<br>`pretrained_tag: laion2b_s34b_b79k`<br>`checkpoint_path: ../weights/open_clip/ViT-B-32-laion2b_s34b_b79k.bin` | `curl -L https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin -o weights/open_clip/ViT-B-32-laion2b_s34b_b79k.bin` |

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
