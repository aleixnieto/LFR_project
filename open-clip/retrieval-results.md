# Image-to-text retrieval results

Results reported from the comparison configured in
[retrieval_compare_config.yaml](configs/retrieval_compare_config.yaml).
The configuration selects the RSICD `test` split.

## Metric

Accuracy@K is the fraction of image queries for which at least one of the top K
retrieved caption texts appears in that image's ground-truth caption list.
For instance: 100 images evaluated, 34 images have at least one correct caption in their top 5 -> Accuracy@5 = 34/100 = 34%.

Duplicate captions are removed within each image, but identical text from
different images remains in the candidate pool. 

## Results

Accuracy values are fractions, not percentages. Best scores are bold.

| Model | Accuracy@1 | Accuracy@3 | Accuracy@5 |
| --- | ---: | ---: | ---: |
| baseline | 0.0631 | 0.1565 | 0.2214 |
| supervised_all_unique | 0.1006 | **0.2342** | **0.3394** |
| semiclip | **0.1134** | 0.2315 | 0.3239 |

Both adapted models outperform the pretrained baseline at all three cutoffs.
SemiCLIP achieves the highest Accuracy@1, while supervised all-unique-caption
fine-tuning achieves the highest Accuracy@3 and Accuracy@5.

## Running the comparison

From the project root, with the configured checkpoints available locally:

```bash
python open-clip/main_retrieval_compare.py --device cuda
```

The [comparison script](main_retrieval_compare.py) evaluates every model on the
same image records and caption pool, recomputing embeddings for each checkpoint.
It saves a summary and per-model metrics and predictions under
`evaluations/open_clip_retrieval_comparison`.
