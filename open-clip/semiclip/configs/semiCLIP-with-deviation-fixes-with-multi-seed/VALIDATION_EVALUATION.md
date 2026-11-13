# SemiCLIP validation evaluation

The `validation_seed{42,123,2026}.yaml` files compare the `semiCLIP1`
(random-one) and `semiCLIPall` (all-unique) Stage-2 final checkpoints for each training seed. Every
config evaluates the same 219-image derived validation partition using the
seed-42 `labeled_validation_filenames` from `concepts.json`.

Run these after all required `stage2_final.pt` checkpoints and the seed-42
concept artifact are present:

```bash
for seed in 42 123 2026; do
  python open-clip/main_zeroshot_compare_config_subset.py \
    --models-config "open-clip/semiclip/configs/semiCLIP-with-deviation-fixes-with-multi-seed/validation_seed${seed}.yaml" \
    --device cuda \
    --output-dir "evaluations/semiclip-with-deviation-fixes-with-multi-seed/seed${seed}"
done

python open-clip/configs/sup-deterministic-multi-seed/aggregate_validation.py \
  --results-root evaluations/semiclip-with-deviation-fixes-with-multi-seed \
  --output-dir evaluations/semiclip-with-deviation-fixes-with-multi-seed/aggregate
```

The aggregate reports mean and sample standard deviation across seeds for
accuracy, weighted precision/recall/F1, and macro F1. The evaluation configs
point at the final Stage-2 checkpoints, so any run that needs repair must be
completed before evaluating that seed.

## Test-set evaluation

The `test_seed{42,123,2026}.yaml` files evaluate `semiCLIP1` and
`semiCLIPall` on the original RSICD `test` split. Use these only for final
reporting after validation-based choices are frozen.

```bash
for seed in 42 123 2026; do
  python open-clip/main_zeroshot_compare.py \
    --models-config "open-clip/semiclip/configs/semiCLIP-with-deviation-fixes-with-multi-seed/test_seed${seed}.yaml" \
    --device cuda \
    --output-dir "evaluations/semiclip-with-deviation-fixes-with-multi-seed/test/seed${seed}"
done

python open-clip/configs/sup-deterministic-multi-seed/aggregate_validation.py \
  --results-root evaluations/semiclip-with-deviation-fixes-with-multi-seed/test \
  --output-dir evaluations/semiclip-with-deviation-fixes-with-multi-seed/test/aggregate \
  --split-name test
```
