import os
import sys
import yaml
import argparse
import clip
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import shutil
from transformers import Adafactor, get_scheduler

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataloader import rsicd_loader
from utils import *

from torchvision import transforms
from torchvision.transforms import v2 as T

# TODO, this should be put somewhere else of course
unsup_augment = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

def clip_loss(model, imgs, texts, caption_to_image):
    """
    Contrastive loss supporting variable number of captions per image.

    Args:
        imgs: Tensor [B, 3, H, W]
        texts: Tensor [N, 77]
        caption_to_image: LongTensor [N], index mapping text to image ID
    """
    device = imgs.device
    B = imgs.shape[0]
    N = texts.shape[0]


    img_features = model.encode_image(imgs)  # [B, D]
    txt_features = model.encode_text(texts)  # [N, D]
 
    # Normalize
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp()


    # === Text-to-image ===
    logits_txt2img = logit_scale * (txt_features @ img_features.T)  # [N, B]

    targets_txt2img = torch.zeros_like(logits_txt2img)
    targets_txt2img[torch.arange(N), caption_to_image] = 1.0


    log_probs_txt2img = torch.log_softmax(logits_txt2img, dim=1)


    loss_txt2img = -(targets_txt2img * log_probs_txt2img).sum(dim=1).mean()

    # === Image-to-text ===
    logits_img2txt = logit_scale * (img_features @ txt_features.T)  # [B, N]
    targets_img2txt = torch.zeros_like(logits_img2txt)
    

    # NOTE, soft targets
    for i in range(B):
        pos = (caption_to_image == i).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            targets_img2txt[i, pos] = 1.0 / len(pos)  # soft assignment
    

    log_probs_img2txt = torch.log_softmax(logits_img2txt, dim=1)
    loss_img2txt = -(targets_img2txt * log_probs_img2txt).sum(dim=1).mean()

    total_loss = (loss_txt2img + loss_img2txt) / 2

    # === Logging cosine similarities (only positive pairs) ===
    sims_txt2img = logits_txt2img[torch.arange(N), caption_to_image]
    avg_sim_txt2img = sims_txt2img.mean().item()

    sims_img2txt = logits_img2txt[targets_img2txt.bool()]
    avg_sim_img2txt = sims_img2txt.mean().item()

    return total_loss, avg_sim_txt2img, avg_sim_img2txt



def train_one_epoch(model, dataloader, optimizer, lr_scheduler, device):
    model.train()
    total_loss = 0
    total_sup_loss = 0
    total_unsup_loss = 0
    total_sim_txt2img = 0
    total_sim_img2txt = 0
    num_batches = 0
    
    tbar = tqdm(dataloader, desc="Training")
    for batch in tbar:
        imgs = batch["images"].to(device)
        is_supervised = batch["is_supervised"].to(device)

        texts = batch["texts"]
        caption_to_image = batch["caption_to_image"]

        if texts is not None:
            texts = texts.to(device)
            caption_to_image = caption_to_image.to(device)
        

        sup_imgs = imgs[is_supervised]
        unsup_imgs = imgs[~is_supervised]

        # === Remap caption_to_image indices ===
        if texts is not None:
            old_to_new = {i: new_i for new_i, i in enumerate(torch.where(is_supervised)[0].tolist())}
            caption_mask = torch.tensor(
                [i.item() in old_to_new for i in caption_to_image],
                dtype=torch.bool,
                device=device
            )
            texts = texts[caption_mask]
            caption_to_image = caption_to_image[caption_mask]
            caption_to_image = torch.tensor(
                [old_to_new[i.item()] for i in caption_to_image],
                dtype=torch.long,
                device=device
            )

        optimizer.zero_grad()
        sup_loss = torch.zeros(1, device=device, requires_grad=True)
        unsup_loss = torch.zeros(1, device=device, requires_grad=True)

        # === Supervised loss (CLIP) ===
        if sup_imgs.shape[0] > 0:
            sup_loss, sims_txt2img, sims_img2txt = clip_loss(model, sup_imgs, texts, caption_to_image)
        
        else:
            sims_txt2img = sims_img2txt = 0.0

        # NOTE this should be a function later of course 
        # === Unsupervised SimCLR loss ===
        if unsup_imgs.shape[0] > 0:
            # Generate two augmentations per image (returns [2B, 3, 224, 224])
            views1 = unsup_augment(unsup_imgs)
            views2 = unsup_augment(unsup_imgs)
            views = torch.cat([views1, views2], dim=0)  # [2B, 3, 224, 224]
            
            feats = model.encode_image(views)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            logits = feats @ feats.T  # [2B, 2B]
            logits = logits / 0.07  # temperature

            B = unsup_imgs.size(0)
            labels = torch.cat([
                torch.arange(B, 2 * B, device=device),
                torch.arange(B, device=device)
            ])

            # Mask diagonal (self-similarity)
            logits.masked_fill_(torch.eye(2 * B, device=device).bool(), -1e4)

            unsup_loss = nn.CrossEntropyLoss()(logits, labels)

        num_sup = sup_imgs.shape[0]
        num_unsup = unsup_imgs.shape[0]

        print(f'supvised samples {num_sup} vs unsupervised samples {num_unsup}')

        total = num_sup + num_unsup

        # NOTE weigthing
        beta_sup = num_sup / total
        beta_unsup = num_unsup / total
                
        loss = beta_sup * sup_loss + beta_unsup * unsup_loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        logit_scale_value = model.logit_scale.exp().item()
        current_lr = optimizer.param_groups[0]["lr"]
        
        model.logit_scale.data.clamp_(0, 4.6052)
        

        total_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()
        
        total_sim_txt2img += sims_txt2img
        total_sim_img2txt += sims_img2txt
        num_batches += 1


    avg_loss = total_loss / num_batches
    avg_sup_loss = total_sup_loss / num_batches
    avg_unsup_loss = total_unsup_loss / num_batches
    
    avg_sim_txt2img = total_sim_txt2img / num_batches
    avg_sim_img2txt = total_sim_img2txt / num_batches
    
    logit_scale_value = model.logit_scale.exp().item()
    current_lr = optimizer.param_groups[0]["lr"]

    return {
    "loss": avg_loss,
    "sup_loss": avg_sup_loss,
    "unsup_loss": avg_unsup_loss,
    "sim_img2txt": avg_sim_img2txt,
    "sim_txt2img": avg_sim_txt2img,
    "logit_scale": logit_scale_value,
    "lr": current_lr,
    }



@torch.no_grad()
def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_sim_txt2img = 0
    total_sim_img2txt = 0
    num_batches = 0

    vbar = tqdm(dataloader, desc="Validating")

    for batch in vbar:
        images = batch["images"].to(device)
        is_supervised = batch["is_supervised"].to(device)
        texts = batch["texts"]
        caption_to_image = batch["caption_to_image"]

        if is_supervised.sum() == 0 or texts is None or caption_to_image is None:
            continue

        sup_imgs = images[is_supervised]  # fix: slice inside function
        texts = texts.to(device)
        caption_to_image = caption_to_image.to(device)

        # === Remap caption_to_image indices ===
        old_to_new = {i: new_i for new_i, i in enumerate(torch.where(is_supervised)[0].tolist())}
        caption_mask = torch.tensor(
            [i.item() in old_to_new for i in caption_to_image],
            dtype=torch.bool,
            device=device
        )
        texts = texts[caption_mask]
        caption_to_image = caption_to_image[caption_mask]
        caption_to_image = torch.tensor(
            [old_to_new[i.item()] for i in caption_to_image],
            dtype=torch.long,
            device=device
        )


        loss, sims_txt2img, sims_img2txt = clip_loss(model, sup_imgs, texts, caption_to_image)

        total_loss += loss.item()
        total_sim_txt2img += sims_txt2img
        total_sim_img2txt += sims_img2txt
        num_batches += 1

    if num_batches == 0:
        print("⚠️ No supervised validation batches found!")
        return {
            "loss": float("nan"),
            "sim_img2txt": float("nan"),
            "sim_txt2img": float("nan"),
        }

    return {
        "loss": total_loss / num_batches,
        "sim_img2txt": total_sim_img2txt / num_batches,
        "sim_txt2img": total_sim_txt2img / num_batches,
    }



if __name__ == "__main__":
    
    log_data = defaultdict(list)

    assert len(sys.argv) > 1, "Please provide the path to the config file."
    config_name = sys.argv[1]

    config_rel_path = os.path.relpath(config_name, "config") 
    config_no_ext = os.path.splitext(config_rel_path)[0]      

    config = get_config(config_name)
    experiment_name = config["experiment_name"]
    checkpoint_dir = os.path.join("results", config_no_ext)       

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config in folder
    saved_config_path = os.path.join(checkpoint_dir, "config.yaml")
    shutil.copy(config_name, saved_config_path)

    set_seed(config["seed"])

    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model, _ = clip.load(config["model_path"], device=device)
    tokenizer = clip.tokenize

    dataset = rsicd_loader(
        image_dir = config["image_dir"], 
        json_path = config["json_path"], 
        text_class = config["txtclasses"],
        split = config["split"], 
        num_captions = config["num_captions"],  
        tokenizer=tokenizer)
    

    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.7)
    

    train_dt = DataLoader(train_dataset, 
    config["batch_size"], 
    shuffle=True, 
    num_workers = config["num_workers"], 
    pin_memory = True,
    collate_fn = collate_mixed_advanced)  

    # NOTE for advanced version, I could only read in the B-split since im not valadiing on unsupervised data?
    val_dt = DataLoader(val_dataset, 
    config["batch_size"], 
    shuffle=True, 
    num_workers = config["num_workers"], 
    pin_memory = True,
    collate_fn = collate_mixed_advanced)  



    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr = config["lr"],
    #     betas=(0.9, 0.98),
    #     eps=1e-6,
    #     weight_decay=config["weight_decay"]
    # )

    # NOTE Try new optimizer
    optimizer = Adafactor(
    model.parameters(),
    scale_parameter=True,
    relative_step=False,  # We'll do our own scheduler
    warmup_init=False,
    lr=config["lr"]  # should be 1e-4
    )

    num_training_steps = config["epochs"] * len(train_dt)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # NOTE Investigate statistics of the data
    #stats = collect_caption_stats(dataset)
    #print_caption_stats(stats)


    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        train_stats  = train_one_epoch(model, train_dt, optimizer, lr_scheduler, device)
        val_stats = validate_one_epoch(model, val_dt, device)


        print(f"Epoch {epoch}:")
        print(f"  Total      Loss: {train_stats['loss']:.4f}")
        print(f"  Supervised Loss: {train_stats['sup_loss']:.4f}")
        print(f"  Unsupervised Loss: {train_stats['unsup_loss']:.4f}")
        print(f"  Img→Txt Sim: {train_stats['sim_img2txt']:.4f}")
        print(f"  Txt→Img Sim: {train_stats['sim_txt2img']:.4f}")
        print()

        print("\n")
        
        # Logging
        log_data["epoch"].append(epoch)
        log_data["train_loss"].append(float(train_stats["loss"]))
        log_data["train_sup_loss"].append(float(train_stats["sup_loss"]))
        log_data["train_unsup_loss"].append(float(train_stats["unsup_loss"]))
        log_data["train_Sim_Img2Txt"].append(float(train_stats["sim_img2txt"]))
        log_data["train_Sim_Txt2Img"].append(float(train_stats["sim_txt2img"]))
        log_data["logit_scale"].append(float(train_stats["logit_scale"]))
        log_data["lr_rate"].append(float(train_stats["lr"]))
        
        log_data["val_loss"].append(float(val_stats["loss"]))
        log_data["val_Sim_Img2Txt"].append(float(val_stats["sim_img2txt"]))
        log_data["val_Sim_Txt2Img"].append(float(val_stats["sim_txt2img"]))


        
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_checkpoint_path = os.path.join(checkpoint_dir, "ckp_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_stats['loss'],
                'val_loss': val_stats['loss'],
            }, best_checkpoint_path)
            print(f"✅ Best checkpoint updated at {best_checkpoint_path}")

        
        # Always save last checkpoint
        if (epoch + 1) == config["epochs"]:
            last_checkpoint_path = os.path.join(checkpoint_dir, "ckp_last.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_stats['loss'],
                'val_loss': val_stats['loss'],
            }, last_checkpoint_path)
            print(f"💾 Last checkpoint saved at {last_checkpoint_path}")

    log_path = os.path.join(checkpoint_dir, "metrics_log.yaml")
    with open(log_path, "w") as f:
        yaml.dump(dict(log_data), f)
    print(f"Metrics log updated at {log_path}")

