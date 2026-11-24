import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

IMAGE_DIR = Path("../../RSICD_images")
OUTPUT = Path("evaluations/qualitative_figure")
OUTPUT.mkdir(exist_ok=True)

PRED_FILE = "evaluations/ablations_test/baseline/predictions.json"
data = json.load(open(PRED_FILE))
pred_map = {item["filename"]: item for item in data}

# 3 failures cases exemple: playfields/playground + other
examples = [
    {"filename": "playground_59.jpg",  "caption": "Playground→playfields"},
    {"filename": "playground_367.jpg", "caption": "Playground→Stadium"},
    {"filename": "commercial_37.jpg",  "caption": "Commercial→School"},
]

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
for ax, ex in zip(axes, examples):
    fname = ex["filename"]
    img = mpimg.imread(IMAGE_DIR / fname)
    p = pred_map[fname]
    ax.imshow(img)
    ax.axis("off")
    color = "red" if p["prediction"] != p["ground_truth"] else "green"
    ax.set_title(
        f"True: {p['ground_truth']}\nPred: {p['prediction']}",
        fontsize=10, color=color
    )
    ax.set_xlabel(fname, fontsize=7)

plt.suptitle("Qualitative failure cases (SemiCLIP baseline)", fontsize=12)
plt.tight_layout()
out = OUTPUT / "failure_cases.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Figure saved to {out}")
