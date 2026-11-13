"""Compare paired test-set retrieval R@10 outcomes across model families."""

import argparse
import json
import math
from pathlib import Path


SEEDS = (42, 123, 2026)
PAIRS = (("sup1", "semiCLIP1"), ("supAll", "semiCLIPall"))


def load_predictions(path: Path) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"Expected prediction list in {path}")
    indexed = {row["filename"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate filenames in {path}")
    return indexed


def exact_mcnemar_p(corrected: int, regressed: int) -> float:
    discordant = corrected + regressed
    if discordant == 0:
        return 1.0
    lower = min(corrected, regressed)
    lower_tail = sum(math.comb(discordant, count) for count in range(lower + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def compare_pair(
    reference_path: Path, candidate_path: Path, reference_label: str,
    candidate_label: str, seed: int,
) -> dict[str, object]:
    reference = load_predictions(reference_path)
    candidate = load_predictions(candidate_path)
    if set(reference) != set(candidate):
        raise ValueError(f"Prediction filenames differ for seed {seed}, {reference_label}/{candidate_label}")

    semi_better = sup_better = both_hit = both_miss = 0
    for filename, sup_row in reference.items():
        semi_row = candidate[filename]
        sup_hit = sup_row.get("correct_at_10")
        semi_hit = semi_row.get("correct_at_10")
        if not isinstance(sup_hit, bool) or not isinstance(semi_hit, bool):
            raise ValueError(f"Missing boolean correct_at_10 for {filename}")
        if semi_hit and not sup_hit:
            semi_better += 1
        elif sup_hit and not semi_hit:
            sup_better += 1
        elif sup_hit:
            both_hit += 1
        else:
            both_miss += 1

    n = len(reference)
    sup_r10 = (sup_better + both_hit) / n
    semi_r10 = (semi_better + both_hit) / n
    return {
        "seed": seed,
        "supervised_variant": reference_label,
        "semiclip_variant": candidate_label,
        "num_images": n,
        "sup_r10": sup_r10,
        "semiclip_r10": semi_r10,
        "semiclip_better": semi_better,
        "sup_better": sup_better,
        "discordant_pairs": semi_better + sup_better,
        "both_hit": both_hit,
        "both_miss": both_miss,
        "p_value_exact_two_sided_mcnemar": exact_mcnemar_p(semi_better, sup_better),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openclip-results-root", type=Path, required=True)
    parser.add_argument("--semiclip-results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    openclip_root = args.openclip_results_root.resolve()
    semiclip_root = args.semiclip_results_root.resolve()
    results = []
    for sup_label, semi_label in PAIRS:
        for seed in SEEDS:
            results.append(compare_pair(
                openclip_root / f"seed{seed}" / sup_label / "predictions.json",
                semiclip_root / f"seed{seed}" / semi_label / "predictions.json",
                sup_label,
                semi_label,
                seed,
            ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "retrieval_mcnemar_r10.json").open("w", encoding="utf-8") as file:
        json.dump({"test": "exact two-sided McNemar", "metric": "R@10", "results": results}, file, indent=2)
        file.write("\n")
    lines = [
        "# Paired test-set retrieval R@10 comparisons", "",
        "Exact two-sided McNemar tests compare per-image R@10 hits for each seed.",
        "Positive direction is SemiCLIP: `semiclip_better` counts images only SemiCLIP retrieves in the top 10; `sup_better` counts images only supervised OpenCLIP retrieves in the top 10.",
        "Seeds are tested separately because they reuse the same test images.", "",
        "| Comparison | Seed | Sup R@10 | SemiCLIP R@10 | SemiCLIP better | Sup better | Discordant | Exact p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['semiclip_variant']} vs {row['supervised_variant']} | {row['seed']} | "
            f"{row['sup_r10']:.4f} | {row['semiclip_r10']:.4f} | "
            f"{row['semiclip_better']} | {row['sup_better']} | "
            f"{row['discordant_pairs']} | {row['p_value_exact_two_sided_mcnemar']:.6g} |"
        )
    (args.output_dir / "retrieval_mcnemar_r10.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote paired R@10 McNemar comparisons to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
