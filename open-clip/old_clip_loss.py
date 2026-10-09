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
    
    # NOTE Hard targets
    # for i in range(B):
    #     caption_indices = (caption_to_image == i).nonzero(as_tuple=True)[0]
    #     targets_img2txt[i, caption_indices] = 1.0

    

    log_probs_img2txt = torch.log_softmax(logits_img2txt, dim=1)
    loss_img2txt = -(targets_img2txt * log_probs_img2txt).sum(dim=1).mean()

    total_loss = (loss_txt2img + loss_img2txt) / 2

    # === Logging cosine similarities (only positive pairs) ===
    sims_txt2img = logits_txt2img[torch.arange(N), caption_to_image]
    avg_sim_txt2img = sims_txt2img.mean().item()

    sims_img2txt = logits_img2txt[targets_img2txt.bool()]
    avg_sim_img2txt = sims_img2txt.mean().item()

    return total_loss, avg_sim_txt2img, avg_sim_img2txt