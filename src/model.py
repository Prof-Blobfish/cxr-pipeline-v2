import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.data import NIH_DISEASES


# bf16 backbone for speed, but bf16 head squashes logits (only 576 distinct logits/class across 10,542 val imgs vs 10,534 logits fp32)
class FP32Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            return super().forward(x.float())


def build_model(num_classes: int = len(NIH_DISEASES), pretrained: bool = True) -> nn.Module: # Head size comes from disease list / pretrained: False is useful for ablation later (experiment that removes one ingredient to measure what contributes)
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None # Names the exact weights version for reproducibility
    model = densenet121(weights=weights)

    in_features = model.classifier.in_features # Reads the 1024 off existing layer

    # Head is replaced, not appended, since ImageNet head's 1000 classes mean nothing here
    model.classifier = FP32Linear(in_features, num_classes) # Subclasses nn.Linear, so init_head_bias and state_dict keys work the same

    return model


def init_head_bias(model: nn.Module, bias: torch.Tensor) -> None:
    with torch.no_grad(): # One-time setup step; changing a param in place while autograd is recording raises an error
        model.classifier.bias.copy_(bias) # Trailing underscore "copy_(bias)" means it changes the tensor in place. Param obj stays the same one the optimizer will track
