import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_training_history(history, probe_history, output_dir):
    """Plot DINO training loss and linear-probe validation accuracy."""

    epochs = [entry["epoch"] for entry in history]
    losses = [entry["train_loss"] for entry in history]

    probe_epochs = [entry["epoch"] for entry in probe_history]
    probe_accuracies = [entry["accuracy"] * 100 for entry in probe_history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: DINO loss
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_title("DINO Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)

    # Right: Linear-probe performance
    axes[1].plot(probe_epochs, probe_accuracies, marker="o")
    axes[1].set_title("Linear Probe Validation Accuracy")
    axes[1].set_xlabel("DINO Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("DINO Domain Adaptation on RSICD")
    fig.tight_layout()

    output_path = output_dir / "training_history.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Training plots saved to {output_path}")
