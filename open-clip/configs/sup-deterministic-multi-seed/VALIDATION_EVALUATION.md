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

## Validation evaluation

The `validation_seed{42,123,2026}.yaml` files each compare the matching
`sup1` (random-one) and `supAll` (all-unique) checkpoints on the same 219-image derived
validation partition. They use the `labeled_validation_filenames` list from
the seed-42 SemiCLIP `concepts.json`; all configs use `split_seed: 42`.
Run them after the supervised checkpoints and seed-42 SemiCLIP Stage-1
concept artifact are available:

```bash
for seed in 42 123 2026; do
  python open-clip/main_zeroshot_compare_config_subset.py \
    --models-config "open-clip/configs/sup-deterministic-multi-seed/validation_seed${seed}.yaml" \
    --device cuda \
    --output-dir "evaluations/sup-deterministic-multi-seed/seed${seed}"
done

python open-clip/configs/sup-deterministic-multi-seed/aggregate_validation.py \
  --results-root evaluations/sup-deterministic-multi-seed \
  --output-dir evaluations/sup-deterministic-multi-seed/aggregate
```

The aggregate contains mean and sample standard deviation across the three
seeds for accuracy, weighted precision/recall/F1, and macro F1.

## Test-set evaluation

The `test_seed{42,123,2026}.yaml` files evaluate `sup1` and `supAll` on the
original RSICD `test` split. Use these only for final reporting after
validation-based choices are frozen.

```bash
for seed in 42 123 2026; do
  python open-clip/main_zeroshot_compare.py \
    --models-config "open-clip/configs/sup-deterministic-multi-seed/test_seed${seed}.yaml" \
    --device cuda \
    --output-dir "evaluations/sup-deterministic-multi-seed/test/seed${seed}"
done

python open-clip/configs/sup-deterministic-multi-seed/aggregate_validation.py \
  --results-root evaluations/sup-deterministic-multi-seed/test \
  --output-dir evaluations/sup-deterministic-multi-seed/test/aggregate \
  --split-name test
```
