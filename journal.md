# 11/04/2026
- Hit the build_image_paths assert (112,123 files) → cause + fix

# 11/05/2026
- Built src/model.py (DenseNet-121, Linear(1024→14) head)
- Sanity checks: initial loss 0.6633, overfit to 0.0034
- Fixed code to isolate and keep test data hidden
- Learned: classes/__init__/dunders/self/type hints, SGD vs AdamW

## Theory/Notes:
- Softmax vs. Sigmoid:
    - Softmax: all classes have probabilities that add up to 1; makes ranking relative and gives us a clear winner, making others clear losers
    - Sigmoid: each class has its own independent probability and may add up beyond 1; better for multilabel classes like cxr diseases

- Transfer learning
    - A CNN learns features in a hierarchy. Early layers detect edges, gradients, textures; middle combines them into shapes/patterns; late layers combine those into objects. Early and middle are universal (edge is edge no matter what object it is)
    - Pretraining = basics learned from 1.2M ImageNet images -> we only adapt to X-rays → faster, better with limited data, less overfitting.
    - Domain shift: X-rays are grayscale projections of overlapping 3D anatomy, unlike natural photos.
        - We must fine-tune every layer, not just the head. Freezing backbone would assume ImageNet's high-level features ("dog face," "car wheel") are already right for "pleural effusion," and they're not.

- DenseNet-121
    - Standard baseline: best known papers for this dataset use this model and later work compares against it
    - Dense connections: each layer receives the feature maps of all earlier layers in its block, which reuses features efficiently and gives gradients short paths back through the network so it trains well.
    - Small enough to iterate fast (~1.5 min per epoch)
    - Newer architectures for later experiments (ConvNeXt, ViTs, CXR-specific pretraining)

- Global average pooling (GAP) averages each 7x7 feature map down to one number.
    - Makes output less sensitive to where a finding appears. Effusion is effusion no matter where it is in the image.
    - Works for Grad-CAM: head is linear on top of averaged spatial maps, class weights can be projected back onto the 7x7 grid and can show where the model looked.

- Multi-label head: replace Linear(1024 -> 1000) (ImageNet classes) with Linear(1024 -> 14) (one output per disease)
    - 14 independent outputs, each meaning "is disease k present?" Since multiple can be true, each gets its own sigmoid. Softmax would force them to compete.
    - Model outputs logits, not probabilities. BCEWithLogitsLoss combines sigmoid + BCE using log-sum-exp trick (numerically stable).
    - New head is randomly initialized, so at the start its logits are close to 0 -> sigmoid ≈ 0.5 -> "50% chance of every disease." (Predictable starting loss)

- Sanity checks each model should pass:
    - 1: initial loss ≈ ln2. With logits near 0, every pred is ~0.5. BCE at p = 0.5 is -ln(0.5) = 0.693 no matter the label. If init loss is like 5.0 or 0.01, something's wired wrong. Mine was 0.744, so close enough
    - 2: overfit a single batch. Train on one fixed batch over and over. Working model + loss + optimizer should memorize it, with loss going to ~0. If it can't the bug is in the wiring, and it would be hidden in a full training run. Mine went 0.6633 -> 0.0034.
        - Overfitting the batch proves the model can learn. It says nothing about generalization.

- Mixed precision (bf16): the 9x speed difference:
    - bf16 = 16-bit with fp32's range, less precision; autocast does convs/matmuls in bf16, loss in fp32. ~9× faster (106 → ~900 img/s), roughly matches the DataLoader, so no cache needed.