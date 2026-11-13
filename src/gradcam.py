from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import build_model


def load_member(checkpoint: Path, device: torch.device) -> nn.Module:
    model = build_model(pretrained=False) # Checkpoint overwrites every weight anyway, no point downloading ImageNet weights
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)) # weights_only=True: safe loading mode, no pickled code
    return model.to(device, memory_format=torch.channels_last).eval() # .eval() makes BatchNorm use its running statistics. Very important.


# Same steps as torchvision's DenseNet.forward, but returns the last feature map too so we can differentiate against it
def forward_with_features(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    feats = F.relu(model.features(x)) # (B, 1024, 7, 7)
    logits = model.classifier(F.adaptive_avg_pool2d(feats, 1).flatten(1))
    return feats, logits


# Grad-CAM before its final ReLU: sum over channels of alpha_k * A_k, alpha_k = spatial mean of d(logit)/dA_k. Shape (B, 7, 7)
def gradcam_raw(model: nn.Module, x: torch.Tensor, class_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    feats, logits = forward_with_features(model, x)
    score = logits[:, class_idx]
    (grads,) = torch.autograd.grad(score.sum(), feats) # score.sum() -> each img's logit only depends on its own input (eval mode), one backward pass gives per-img gradients for the whole batch
    alpha = grads.mean(dim=(2, 3), keepdim=True)
    return (alpha * feats).sum(dim=1).detach(), score.detach()
