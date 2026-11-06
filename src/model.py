import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.data import NIH_DISEASES


def build_model(num_classes: int = len(NIH_DISEASES), pretrained: bool = True) -> nn.Module: # Head size comes from disease list / pretrained: False is useful for ablation later (experiment that removes one ingredient to measure what contributes)
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None # Names the exact weights version for reproducibility
    model = densenet121(weights=weights)

    in_features = model.classifier.in_features # Reads the 1024 off existing layer
    model.classifier = nn.Linear(in_features, num_classes) # Head is replaced, not appended, since ImageNet head's 1000 classes mean nothing here

    return model
