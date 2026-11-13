# Supervised OpenCLIP: three seeds, fixed split

These six configs extend the OpenCLIP caption-strategy fine-tuning setup to
seeds 42, 123, and 2026 for both `random_one` and `all_unique` caption
policies. All runs use `split_seed: 42`, so they share the same derived 80/20
train/validation partition of the RSICD `val` source split while retaining
different training seeds.

The training recipe matches the existing full caption-strategy configs:
OpenCLIP `ViT-B-32 / laion2b_s34b_b79k`, batch size 16, eight epochs, AdamW
with learning rate `3e-5` and weight decay `0.1`, and warmup fraction `0.1`.
Both caption policies use per-caption text-to-image weighting and validate
with the all-unique caption objective.

Run one config from the repository root with:

```bash
python open-clip/main_finetune.py \
  --config open-clip/configs/sup-deterministic-multi-seed/supervised_random_one_seed42.yaml \
  --device cuda
```

Run the all-unique counterpart by using `supervised_all_unique_seed42.yaml`.
Each config writes to a unique folder under `results/sup-deterministic-multi-seed/`.
