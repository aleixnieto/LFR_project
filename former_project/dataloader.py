"""
IMPLEMENTING DATALOADER FOR RSICD DATASET
- FOR FULL-SUPERVISED FINE-TUNE: TAKE IMAGE FROM "VAL" THEN SPLIT TO "REAL" TRAIN AND VAL
- FOR SEMI-SUPERVISED FINE-TUNE: TAKE IMAGE FROM "TRAIN" BUT NO CAPTION + "REAL" TRAIN SET FOR TRAINING
- FOR TRAINING MODE: RETURN PAIR OF IMAGE-TEXT
- FOR TESTING MODE: RETURN IMAGE AND IT'S FILENAME (FOR SEARCHING CORRECT CLASS)
"""

import json
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
from PIL import Image 
import numpy as np
import torch
from torchvision import transforms
import os
from utils import _search_class, tokenize_texts, check_duplicates



#========= DATALOADER CLASS ========#
class rsicd_loader(Dataset):
    """ Load image and caption from RSICD data.
    Depending on training mode, either single or multiple captions will be loaded
    Depending on which phase [train or test], different split will be loaded
    Mode = train:  load from Split B and return (image, caption)
    Mode = test:   load from Split C and ONLY return image
    
    Args:
        image_dir   : path to the image directory
        json_dataset: .json file contains list of images and captions
        multi_cap   : option to choose single or multiple captions
        fully_sup   : option for fully-supervised or NOT
        mode       : ['train' or 'test']
    Return:
        if mode = train: image (Tensor), caption
        if mode = test: image, image's filename
    """
    
    def __init__(self, image_dir, json_path, text_class, split, num_captions = None, tokenizer = None):
        self.image_dir = image_dir
        self.json_path = json_path
        self.text_class = text_class
        self.split = split
        self.num_captions = num_captions
        self.tokenizer = tokenizer 

        # === CLIP-style preprocessing ===
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        
        with open(self.json_path, 'r') as file:
            self.data_info = json.load(file)
            self.data_info = self.data_info['images']
        
        if self.split == "sup_and_unsup":
            # Combine both train and val splits. These will have the train or split in them
            self.images_info = [item for item in self.data_info if item.get('split') in ['train', 'val']]
        
        else:
            if self.split == "unsup":
                self.images_info = [item for item in self.data_info if item.get('split') == 'train']

            elif self.split == "sup":
                self.images_info = [item for item in self.data_info if item.get('split') == 'val']

            elif self.split == "test":
                self.images_info = [item for item in self.data_info if item.get('split') == 'test']           
            else:
                raise ValueError(f"Invalid split: {self.split}") 

            # NOTE overfit to one example
            #self.images_info = self.images_info[:2]
            #self.images_info = [first_entry]

        #print(len(self.images_info))
        #print(self.images_info[0])
        #print(self.images_info[0].keys())

    def process_captions(self, img_info):
        """Helper to extract, tokenize, deduplicate, and select captions."""
        all_captions = [" ".join(s["tokens"]) for s in img_info["sentences"]]


        # Tokenize all
        tokenized = tokenize_texts(tuple(all_captions), self.tokenizer)

        # Optional sanity check
        # check_duplicates(tokenized, all_captions)

        # Remove duplicates
        unique_captions = list({tuple(t.tolist()): t for t in tokenized}.values())

        # Select subset (if needed)
        selected = unique_captions[:self.num_captions] if len(unique_captions) <= self.num_captions \
                else random.sample(unique_captions, self.num_captions)

        caption_tensor = torch.stack(selected)
        if self.num_captions == 1:
            caption_tensor = caption_tensor[0].unsqueeze(0)

        return caption_tensor

    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, idx):
        #print(f'Working wiht idx: {idx}')
        
        img_info = self.images_info[idx]
        img_filename = img_info['filename']
        img_path = os.path.join(self.image_dir, img_filename)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.split == "sup_and_unsup":
            split = img_info['split']  # This is either 'train' (A) or 'val' (B)

            if split == 'train':
                # Split A: return only image
                return image
            
            elif split == 'val':
                caption_tensor = self.process_captions(img_info)
                return image, caption_tensor
        
        if self.split == "sup":
            
            caption_tensor = self.process_captions(img_info)
            return image, caption_tensor
        
        elif self.split == "unsup":
            return image

        elif self.split == 'test':
            self.label = _search_class(img_filename, self.text_class)
            return image, self.label


#============ MAIN FOR TEST ========#
if __name__ == '__main__':
    
    image_dir = 'data/RSICD_images'
    
    json_path = 'data/dataset_rsicd.json'
    
    data = rsicd_loader(image_dir, json_path, split="B")
    a = data[0]

    #plt.imshow(a[0])
    
    #for i, (img, text) in enumerate(data):
    #    plt.imshow(img)
    #    plt.title(text)
    #    plt.savefig('test_loader.png')
    #    break
