#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

principal_config="open-clip/semiclip/configs/semiclip_rsicd_paper_1gpu_all_unique_seed42.yaml"
control_config="open-clip/semiclip/configs/supervised_all_unique_control_paper_1gpu.yaml"
principal_output="results/open_clip_semiclip_paper_1gpu_all_unique_seed42_functionality_rerun"
control_output="results/open_clip_supervised_all_unique_control_paper_1gpu_seed42_functionality_rerun"

for output_dir in "$principal_output" "$control_output"; do
    if [[ -e "$output_dir" ]]; then
        echo "Refusing to overwrite existing output: $output_dir" >&2
        exit 1
    fi
done

mkdir -p "$principal_output" "$control_output"

python open-clip/semiclip/main.py \
    --stage all \
    --config "$principal_config" \
    --device cuda \
    --output-dir "$principal_output" 2>&1 | tee "$principal_output/train.log"

python open-clip/main_finetune.py \
    --config "$control_config" \
    --device cuda \
    --output-dir "$control_output" 2>&1 | tee "$control_output/train.log"
