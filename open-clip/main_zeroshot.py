import os
import yaml
import clip
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import seaborn as sns
from dataloader import rsicd_loader
from utils import *
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# NOTE, do we need to test it "normally as well, on test c? so not zero-shot, just normal supervised"

def calculate_metrics(gts, preds, class_names):
    acc = accuracy_score(gts, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(gts, preds, average='weighted', zero_division=0)
    f1_per_class = dict(zip(class_names, precision_recall_fscore_support(gts, preds, labels=class_names, zero_division=0)[2]))

    conf_matrix = confusion_matrix(gts, preds, labels=class_names)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f1_per_class": f1_per_class,
        "confusion_matrix": conf_matrix
    }
    
def plot_confusion_matrix(conf_matrix, class_names, save_path="conf_matrix.png"):
    title = save_path.split('/')[-1].replace('.png', '')
    fig, ax = plt.subplots(figsize=(14, 12))  # Slightly larger for better label spacing
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    # Improve x-axis label appearance
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def build_text_features(class_names, model, device, templates=None):
    if templates is None:
        # templates = [
        #     "A satellite image of {}",
        #     "A photo of a {}",
        #     "A close-up of {}",
        #     "An aerial view of {}",
        #     "A remote sensing image of {}"
        # ]
        
        # templates = [
        #     "A satellite image of {}"]

        # Some random chat gpt templates
        templates = [
            "An overhead view of a {}",
            "A top-down satellite shot of a {}",
            "A high-resolution image showing a {}",
            "A geospatial view depicting a {}",
            "An image taken from above showing a {}"
        ]


    all_prompts = [template.format(cls) for cls in class_names for template in templates]
    print(f"📝 Number of templates used per class: {len(templates)}")
    print(f"📝 Total number of prompts: {len(all_prompts)}")

    text_inputs = clip.tokenize(all_prompts).to(device)

    num_classes = len(class_names)
    num_templates = len(templates)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(num_classes, num_templates, -1).mean(dim=1)

    return text_features


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Setup ===
    results_root = "results_subset/finetune_soft_labels_lower_lr"
    json_path = "/proj/cvl/datasets/data/dataset_rsicd.json"
    image_dir = "/proj/cvl/datasets/data/RSICD_images"
    txt_classes = "/proj/cvl/datasets/data/txt_classes_rsicd"

    img_folder = results_root.split("/")[1]
    img_folder = f'images/{img_folder}'

    os.makedirs(img_folder, exist_ok=True)


    class_names = [f.stem for f in Path(txt_classes).iterdir() if f.is_file()]

    # === Dataset ===
    test_set = rsicd_loader(
        image_dir=image_dir,
        json_path=json_path,
        text_class=txt_classes,
        split="C",
        num_captions=None,
        tokenizer=None
    )
    test_dt = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=12)

    # === Load CLIP model ===
    model, _ = clip.load("ViT-B/32", device=device, jit=False)

    # === Loop over experiments ===
    experiment_folders = [f for f in Path(results_root).iterdir() if f.is_dir()]
    experiment_folders.sort()



    for exp_dir in experiment_folders:
        best_ckpt = exp_dir / "ckp_best.pth"
        if not best_ckpt.exists():
            print(f"❌ No best checkpoint found in {exp_dir}")
            continue

        print(f"\n🔍 Running zero-shot for {exp_dir.name}")
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # === Build Text Features
        text_features = build_text_features(class_names, model, device)

        # === Run Evaluation
        all_preds = []
        all_gts = []

        model.eval()
        with torch.no_grad():
            for imgs, label in tqdm(test_dt, desc="Zero-Shot"):
                imgs = imgs.to(device)
                label = label[0]  # from tuple

                image_rep = model.encode_image(imgs)
                image_rep = image_rep / image_rep.norm(dim=-1, keepdim=True)

                logits = image_rep @ text_features.T
                pred = logits.argmax(dim=-1).item()

                all_preds.append(class_names[pred])
                all_gts.append(label)

        metrics = calculate_metrics(all_gts, all_preds, class_names)
        print(f"✅ Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"🎯 Weighted F1-score: {metrics['f1_score']:.4f}")

        # Print top-5 worst F1-score classes
        print("\n📉 Worst performing classes (F1):")
        sorted_f1 = sorted(metrics["f1_per_class"].items(), key=lambda x: x[1])
        for cls, f1 in sorted_f1[:5]:
            print(f"{cls}: {f1:.3f}")

        # Save confusion matrix
        plot_confusion_matrix(metrics["confusion_matrix"], class_names, save_path=f"{img_folder}/{exp_dir.name}_conf_matrix.png")


