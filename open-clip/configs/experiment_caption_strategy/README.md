### Running the caption-strategy experiment

The experiment uses three full-run configurations that differ in caption selection and text-to-image weighting.

The **all-unique/per-caption baseline** encodes every unique caption for an image and averages the text-to-image loss over captions. It exposes the model to all available descriptions, but images with more unique captions contribute more text-to-image terms. 

**Random-one/per-caption** instead uses one deterministic, seed-controlled caption sample per image per epoch. This keeps image exposure unchanged while reducing text-encoder work and giving every image one text-to-image term; the same caption may be sampled in consecutive epochs.

**All-unique/per-image** encodes the same captions as the baseline, but first averages caption losses within each image so every image has equal text-to-image weight. All three variants use the same all-unique/per-caption validation objective.

| Configuration | Training captions | Text-to-image weighting | Question tested |
| --- | --- | --- | --- |
| `finetune_full_config.yaml` | All unique captions | Per caption | Current multi-caption baseline |
| `finetune_full_random_one_config.yaml` | One seed-controlled sample per image per epoch | Per caption | Effect of using all captions; samples may repeat across epochs |
| `finetune_full_per_image_config.yaml` | All unique captions | Per image | Effect of caption-count weighting |

Run the three configurations from the repository root:

```bash
for config in open-clip/configs/experiment_caption_strategy/finetune_full*_config.yaml; do
  python open-clip/main_finetune.py --config "$config" --device cuda || break
done
```

They write to these separate directories:

```text
results/experiment_caption_strategy/open_clip_finetune_all_unique_per_caption
results/experiment_caption_strategy/open_clip_finetune_random_one_per_caption
results/experiment_caption_strategy/open_clip_finetune_all_unique_per_image
```

### Plotting training metrics

Plot all three training histories together with a separate labeled curve for
each model by running this command from the repository root:

```bash
python open-clip/plotting_utils.py \
  results/experiment_caption_strategy/open_clip_finetune_all_unique_per_caption \
  results/experiment_caption_strategy/open_clip_finetune_random_one_per_caption \
  results/experiment_caption_strategy/open_clip_finetune_all_unique_per_image \
  --label all_unique_per_caption \
  --label random_one_per_caption \
  --label all_unique_per_image \
  --output-dir results/experiment_caption_strategy/training_metric_plots
```

The command writes loss, positive text-to-image and image-to-text logits,
logit-scale, and learning-rate plots. Each plot compares all three models over
their training epochs.

### Zero-shot evaluation

After all three training runs finish, compare their best validation-selected
checkpoints with the original non-fine-tuned OpenCLIP policy on the held-out
RSICD test split. The model list and dataset paths are defined in
`zeroshot_compare_config.yaml`.

Run the comparison from the repository root:

```bash
python open-clip/main_zeroshot_compare.py \
  --models-config open-clip/configs/experiment_caption_strategy/zeroshot_compare_config.yaml \
  --device cuda \
  --output-dir evaluations/open_clip_caption_strategy_comparison
```

The evaluation writes an aggregate `summary.json`, comparison plots, and
per-model metrics, predictions, and confusion matrices beneath
`evaluations/open_clip_caption_strategy_comparison/`. The
`non_finetuned_baseline` entry uses the original local OpenCLIP checkpoint;
the other entries use each training run's `ckp_best.pth`.

The fixed budget is equal image exposure rather than equal text-encoder compute.
All configurations use the same source and derived split, seed, image batch
size, eight epochs, `drop_last=False`, optimizer-step count, and shuffled image
order. The all-unique configurations encode more captions than `random_one`.
The per-image configuration isolates loss normalization; it is not a
compute-matched alternative.

For every epoch, compare these entries in each run's `metrics.json`:

- `train.image_order_sha256`, `train.images`, and `train.optimizer_steps` must
  match across runs.
- `train.captions` and `train.captions_per_image` show the text-compute
  difference.
- `train.text_to_image_loss`, `train.image_to_text_loss`, and
  `train.objective_loss` describe the configured training objective.
- `validation.selection_loss` is the common all-unique/per-caption metric used
  for checkpoint selection. It is intentionally not the random-one run's
  training objective.
