import os
import sys
import clip
import copy

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torch.utils.data import ConcatDataset

import timm
import yaml
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import shutil
from transformers import Adafactor, get_scheduler

from dataloader import rsicd_loader
from utils import *


class DINOModel(nn.Module):
    def __init__(self, clip_visual_encoder):
        super().__init__()
        # Student and teacher start identical
        self.student = clip_visual_encoder
        self.teacher = copy.deepcopy(clip_visual_encoder)
        # Freeze teacher weights
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x, is_teacher=False):
        if is_teacher:
            return self.teacher(x)
        else:
            return self.student(x)

    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data

def dino_loss(student, teacher, temp_student=0.1, temp_teacher=0.04, center=None):
    #student_feats = F.normalize(student_feats, dim=-1)
    #teacher_feats = F.normalize(teacher_feats, dim=-1)

    #student_logits = student_feats / temp_student
    #with torch.no_grad():
    #    teacher_probs = F.softmax(teacher_feats / temp_teacher, dim=-1)

    #loss = -torch.sum(teacher_probs * F.log_softmax(student_logits, dim=-1), dim=-1).mean()

    # Normalize and scale student outputs
    student_out = student / temp_student
    
    # Apply softmax to teacher outputs with optional centering
    if center is not None:
        teacher = teacher - center
    teacher_out = F.softmax(teacher / temp_teacher, dim=-1).detach()

    # Compute cross-entropy loss
    loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
    
    return loss



# DINO-style augmentation for unsupervised learning
unsup_augment = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def contrastive_loss(logits, targets):
    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss

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

    # NOTE Maybe we add these back later
    # === Logging cosine similarities (only positive pairs) ===
    # sims_txt2img = logits_txt2img[torch.arange(N), caption_to_image]
    # avg_sim_txt2img = sims_txt2img.mean().item()

    # sims_img2txt = logits_img2txt[targets_img2txt.bool()]
    # avg_sim_img2txt = sims_img2txt.mean().item()

    #return total_loss, avg_sim_txt2img, avg_sim_img2txt
    return total_loss



def train_one_epoch(dino_model, clip_model, dataloader, optimizer, lr_scheduler, device, loss_weights):
    clip_model.train()
    dino_model.train()
    total_loss = 0
    total_sup_loss = 0
    total_unsup_loss = 0
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
        '''
        if texts is not None:
            old_indices = torch.where(is_supervised)[0].tolist()
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(old_indices)}
            
            caption_to_image_list = caption_to_image.tolist()
            caption_mask = [i in old_to_new for i in caption_to_image_list]

            texts = texts[caption_mask]
            caption_to_image = torch.tensor([old_to_new[i] for i in caption_to_image if i in old_to_new], device=device)'''
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


        optimizer.zero_grad(set_to_none=True)
        sup_loss = torch.tensor(0.0, device=device, requires_grad=True)
        unsup_loss = torch.tensor(0.0, device=device, requires_grad=True)

        #with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        # === Unsupervised DINO loss ===
        if unsup_imgs.shape[0] > 0:
            views1 = unsup_augment(unsup_imgs)
            views2 = unsup_augment(unsup_imgs)
            student_out1 = dino_model(views1, is_teacher=False)
            student_out2 = dino_model(views2, is_teacher=False)
            with torch.no_grad():
                teacher_out1 = dino_model(views1, is_teacher=True)
                teacher_out2 = dino_model(views2, is_teacher=True)
            center = torch.zeros_like(teacher_out1)
            unsup_loss = (dino_loss(student_out1, teacher_out2, center=center) + dino_loss(student_out2, teacher_out1, center=center)) / 2
        # === Supervised CLIP loss ===
        if sup_imgs.shape[0] > 0 and texts is not None and texts.shape[0] > 0:
            sup_loss = clip_loss(clip_model, sup_imgs, texts, caption_to_image)
        beta_sup = loss_weights.get("sup", 0.5)
        beta_unsup = loss_weights.get("unsup", 0.5)
        loss = beta_sup * sup_loss + beta_unsup * unsup_loss

        loss.backward()        
        optimizer.step()         
        lr_scheduler.step()

        #dino_model.update_teacher()

        # Logging metrics
        total_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()

        num_batches += 1
        current_lr = optimizer.param_groups[0]["lr"]
        logit_scale_value = (
            clip_model.logit_scale.exp().item()
            if hasattr(clip_model, "logit_scale")
            else 1.0
        )

        avg_loss = total_loss / num_batches     
        avg_sup_loss = total_sup_loss / num_batches     
        avg_unsup_loss = total_unsup_loss / num_batches

    
    return {
        "loss": avg_loss,
        "sup_loss": avg_sup_loss,
        "unsup_loss": avg_unsup_loss,
        "logit_scale": logit_scale_value,
        "lr": current_lr,
    }

@torch.no_grad()
def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(dataloader, desc="Validating"):
        imgs   = batch["images"].to(device)
        texts  = batch["texts"].to(device)
        cmap   = batch["caption_to_image"].to(device)
        
        loss = clip_loss(model, imgs, texts, cmap)
        total_loss += loss.item()
        n_batches  += 1

    return {"loss": total_loss / n_batches}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINO with CLIP backbone")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    log_data = defaultdict(list)

    config_name = args.config
    config_rel_path = os.path.relpath(config_name, "config")
    config_no_ext = os.path.splitext(config_rel_path)[0]

    config = get_config(config_name)
    experiment_name = config["experiment_name"]
    checkpoint_dir = os.path.join("results", config_no_ext)
    os.makedirs(checkpoint_dir, exist_ok=True)

    shutil.copy(config_name, os.path.join(checkpoint_dir, "config.yaml"))

    set_seed(config["seed"])
    loss_weights = config["loss_weights"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to {device}")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()
    clip_model.eval()
    
    dino_model = DINOModel(clip_model.visual).to(device).float()
    
    tokenizer = clip.tokenize

    sup_ds   = rsicd_loader(
        image_dir   = config["image_dir"],
        json_path   = config["json_path"],
        text_class  = config["txtclasses"],
        split       = "sup",            # <-- only the 'val' (captioned) images
        num_captions= config["num_captions"],
        tokenizer   = tokenizer,
    )

    unsup_ds = rsicd_loader(
        image_dir   = config["image_dir"],
        json_path   = config["json_path"],
        text_class  = config["txtclasses"],
        split       = "unsup",          # <-- only the 'train' (uncaptioned) images
        num_captions= None,
        tokenizer   = None,
    )

    sup_train_ds, sup_val_ds = split_dataset(sup_ds, train_ratio=0.7)
    train_ds = ConcatDataset([sup_train_ds, unsup_ds])

    train_dt = DataLoader(train_ds, config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True, collate_fn=collate_mixed_advanced)
    val_dt = DataLoader(sup_val_ds, config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True, collate_fn=collate_mixed_advanced)

    optimizer = Adafactor(
        list(clip_model.parameters()) + list(dino_model.parameters()),
        scale_parameter=True,
        relative_step=False,
        warmup_init=False,
        lr=config["lr"]
        )

    num_training_steps = config["epochs"] * len(train_dt)
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    #scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        train_stats  = train_one_epoch(dino_model, clip_model, train_dt, optimizer, lr_scheduler, device, loss_weights)
        val_stats = validate_one_epoch(clip_model, val_dt, device)

        print(f"Epoch {epoch}:")
        print(f"Total Loss: {train_stats['loss']:.4f}")
        print(f"Supervised Loss: {train_stats['sup_loss']:.4f}")
        print(f"Unsupervised Loss: {train_stats['unsup_loss']:.4f}")
        print("\n")
        
        # Logging
        log_data["epoch"].append(epoch)
        log_data["train_loss"].append(float(train_stats["loss"]))
        log_data["train_sup_loss"].append(float(train_stats["sup_loss"]))
        log_data["train_unsup_loss"].append(float(train_stats["unsup_loss"]))
        log_data["logit_scale"].append(float(train_stats["logit_scale"]))
        log_data["lr_rate"].append(float(train_stats["lr"]))
        log_data["val_loss"].append(float(val_stats["loss"]))

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_checkpoint_path = os.path.join(checkpoint_dir, "ckp_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': clip_model.state_dict(),
                'dino_state_dict': dino_model.state_dict(),
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
                'model_state_dict': clip_model.state_dict(),
                'dino_state_dict': dino_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_stats['loss'],
                'val_loss': val_stats['loss'],
            }, last_checkpoint_path)
            print(f"💾 Last checkpoint saved at {last_checkpoint_path}")

    log_path = os.path.join(checkpoint_dir, "metrics_log.yaml")
    with open(log_path, "w") as f:
        yaml.dump(dict(log_data), f)
    print(f"📊 Metrics log updated at {log_path}")
