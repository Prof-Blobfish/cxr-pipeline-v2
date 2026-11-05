import torch
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool, image_size: int = 224) -> v2.Compose: # v2 -> current torchvision transforms API
    steps = [
        v2.ToImage(), # Turns the PIL image into a tensor (still uint8)
        v2.Resize((image_size, image_size), antialias=True), # Before augmenting, since it's way cheaper to augment a shrunken image
    ]
    if train:
        steps += [
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)), # Realistic patient positioning variance (rotation, shift, zoom)
            v2.ColorJitter(brightness=0.1, contrast=0.1), # Saturation and hue don't matter for a grayscale image
        ]
    steps += [
        v2.ToDtype(torch.float32, scale=True), # Converts 0-255 ints to 0.0-1.0 floats.
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD), # Applies the ImageNet statistics pretrained weights were learned with
    ]
    return v2.Compose(steps)
