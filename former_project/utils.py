import torch
from torch.utils.data import random_split
import yaml
from pathlib import Path
import random
import numpy as np

from collections import defaultdict


def collect_caption_stats(dataset):
    """
    Collect statistics about the number of unique captions per class in a dataset.

    Args:
        dataset: Instance of rsicd_loader for split "B".

    Returns:
        stats_dict: A dictionary mapping class name -> list of unique caption counts per image.
    """
    from collections import defaultdict
    caption_stats = defaultdict(list)

    for idx in range(len(dataset)):
        entry = dataset.images_info[idx]

        # Use class field if available, otherwise fallback to filename prefix
        if "class" in entry:
            class_name = entry["class"]
        else:
            class_name = entry["filename"].split("_")[0].lower()

        captions = [" ".join(s["tokens"]) for s in entry["sentences"]]
        unique_captions = set(captions)

        caption_stats[class_name].append(len(unique_captions))

    return caption_stats


def print_caption_stats(stats_dict):
    print("\n📊 Unique Caption Stats Per Class:")
    for cls, counts in sorted(stats_dict.items()):
        if cls.endswith(".jpg"):
            continue  # skip accidental entries
        avg = sum(counts) / len(counts)
        min_ = min(counts)
        max_ = max(counts)
        print(f"{cls:20s} | Images: {len(counts):3d} | Avg Unique Captions: {avg:.2f} | Min: {min_} | Max: {max_}")



def check_duplicates(tokenized, all_captions):
    print("---- Checking for tokenized duplicates ----")
    for i in range(len(tokenized)):
        for j in range(i + 1, len(tokenized)):
            if torch.equal(tokenized[i], tokenized[j]):
                print(f"Duplicate tokenization: Caption {i} == Caption {j}")
                print("Original texts:")
                print(f"  {all_captions[i]}")
                print(f"  {all_captions[j]}")


def tokenize_texts(text_list, tokenizer):
    return [tokenizer([text], truncate=True)[0] for text in text_list]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Slower, but deterministic
    torch.backends.cudnn.benchmark = True     # Disable auto-tuning

def _search_class(filename, txtclasses_rsicd_DIR):
    """ Given a image filename, search for correct class
    If the class cannot be found, raise Error
    Args:
        filename (string): image filename (with .jpg, .png)
    Return:
        class (string)
    """
    txtclasses_path = Path(txtclasses_rsicd_DIR)
    txtclasses = [f.name for f in txtclasses_path.iterdir() if f.is_file()] 
    
    # Start searching
    for text_file in txtclasses:
        lines = [line.strip() for line in open(Path(txtclasses_rsicd_DIR, text_file), 'r', encoding='utf-8')]
        
        if filename in lines:
            ## Return the class name
            class_name = text_file.split('.')[0]
            return class_name
    try:
        class_name
    except NameError:
        raise ValueError('The image is not in RSICD dataset !!!!')


def collate_mixed_advanced(batch):
    """
    Minimal collate function.
    Just separates supervised and unsupervised entries.

    Returns:
        - images: stacked [B, 3, H, W]
        - texts: [N, 77] or None
        - caption_to_image: [N] or None
        - is_supervised: BoolTensor [B]
    """
    images = []
    texts = []
    caption_to_image = []
    is_supervised = []

    for i, item in enumerate(batch):
        if isinstance(item, tuple):
            image, caption_tensor = item
            images.append(image)
            texts.append(caption_tensor)
            caption_to_image.extend([i] * caption_tensor.shape[0])
            is_supervised.append(True)
        else:
            images.append(item)
            is_supervised.append(False)

    images = torch.stack(images)  # [B, 3, H, W]
    is_supervised = torch.tensor(is_supervised, dtype=torch.bool)  # [B]

    if texts:
        texts = torch.cat(texts, dim=0)  # [N, 77]
        caption_to_image = torch.tensor(caption_to_image, dtype=torch.long)
    else:
        texts = None
        caption_to_image = None

    return {
        "images": images,
        "texts": texts,
        "caption_to_image": caption_to_image,
        "is_supervised": is_supervised,
    }



   #return images, texts, caption_to_image, torch.tensor(is_supervised)


# This function convert model parameter, I don't know why
# But seems to avoid some bugs
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()



def split_dataset(dataset, train_ratio=0.8, seed=42):
    """
    Split a dataset into train and validation sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): Fraction of data to use for training (default 0.8).
        seed (int): Random seed for reproducibility.

    Returns:
        (train_dataset, val_dataset): Two random subsets.
    """
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def get_config(config_path=None):
    if config_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
        args = parser.parse_args()
        config_path = args.config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
 