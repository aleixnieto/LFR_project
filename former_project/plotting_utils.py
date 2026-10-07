import os
import yaml
import matplotlib.pyplot as plt

# === Setup ===
results_dir = "results/finetune_soft_labels_lower_lr"
name = results_dir.split("/")[-1]
output_dir = os.path.join("images", name)
os.makedirs(output_dir, exist_ok=True)

# List experiment folders like baseline_one_cap, baseline_two_cap, etc.
experiment_folders = [f for f in os.listdir(results_dir) if f.startswith("baseline_")]

# Define which metrics to plot
metrics_to_plot = {
    "train_loss": "Train Loss",
    "val_loss": "Validation Loss",
    "train_Sim_Img2Txt": "Train Similarity (Img→Txt)",
    "train_Sim_Txt2Img": "Train Similarity (Txt→Img)",
    "val_Sim_Img2Txt": "Validation Similarity (Img→Txt)",
    "val_Sim_Txt2Img": "Validation Similarity (Txt→Img)",
    "logit_scale": "Logit Scale",
    "lr_rate": "Learning Rate",
}

# Initialize container: {metric_name: {label: (epochs, values)}}
plots = {metric: {} for metric in metrics_to_plot}

# === Load data ===
for folder in sorted(experiment_folders):
    metrics_path = os.path.join(results_dir, folder, "metrics_log.yaml")
    if not os.path.exists(metrics_path):
        continue

    with open(metrics_path, "r") as f:
        log = yaml.safe_load(f)

    cap_count = folder.split("_")[1]
    label = f"{cap_count} caption(s)"

    epochs = log["epoch"]
    for metric in metrics_to_plot:
        if metric in log:
            plots[metric][label] = (epochs, log[metric])

# === Plotting ===
for metric, label_dict in plots.items():
    plt.figure(figsize=(10, 6))
    for label, (epochs, values) in label_dict.items():
        plt.plot(epochs, values, label=label)

    plt.title(f"{metrics_to_plot[metric]} vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(metrics_to_plot[metric])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}.png")
    plt.savefig(save_path)
    plt.close()

print(f"✅ All plots saved in: {output_dir}")
