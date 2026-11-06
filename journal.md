# 11/05/2026
## Theory/Notes:
### Milestone 4
- Softmax vs. Sigmoid:
    - Softmax: all classes have probabilities that add up to 1; makes ranking relative and gives us a clear winner, making others clear losers
    - Sigmoid: each class has its own independent probability and may add up beyond 1; better for multilabel classes like cxr diseases

- Transfer learning
    - A CNN learns features in a hierarchy. Early layers detect edges, gradients, textures; middle combines them into shapes/patterns; late layers combine those into objects. Early and middle are universal (edge is edge no matter what object it is)
    - Pretraining = basics learned from 1.2M ImageNet images -> we only adapt to X-rays -> faster, better with limited data, less overfitting.
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
    - 1: initial loss ≈ ln2. With logits near 0, every pred is ~0.5. BCE at p = 0.5 is -ln(0.5) = 0.693 no matter the label. If init loss is like 5.0 or 0.01, something's wired wrong. Mine was 0.6633, so close enough
    - 2: overfit a single batch. Train on one fixed batch over and over. Working model + loss + optimizer should memorize it, with loss going to ~0. If it can't the bug is in the wiring, and it would be hidden in a full training run. Mine went 0.6633 -> 0.0034.
        - Overfitting the batch proves the model can learn. It says nothing about generalization.

- Mixed precision (bf16): the 9x speed difference:
    - bf16 = 16-bit with fp32's range, less precision; autocast does convs/matmuls in bf16, loss in fp32. ~9× faster (106 -> ~900 img/s), roughly matches the DataLoader, so no cache needed.

### Milestone 5 (Loss and Class Imbalance)
- Loss
    - Model outputs a [B, 14] grid of logits. BCEWithLogitsLoss treats every cell as a yes/no problem, computes binary cross-entropy for each, and avgs over all B x 14 cells:
        $$\ell = -\big[y \log \sigma(z) + (1-y)\log(1-\sigma(z))\big]$$
        - z = the logit (model's raw output for that image and disease)
        - $\sigma(z)$ = sigmoid(z) = the predicted prob that the disease is present
        - y = the true label: 1 (present) or 0 (absent)
        - $\ell$ = the loss for that one cell (lower = better)
        - Negative sign because probabilities are between 0 and 1, so their logs are negative
    - It's basically 14 classifiers sharing one backbone, since features learned for Effusion can help with Atelectasis.

- Class Imbalance and Learning
    - For Hernia, 9 out of 10 batches contain zero positives. Those batches tell the model to change so that z (the logit) gets smaller for that logit (lower probability of that disease), while almost none push z up. The negatives drown out the positives and the shared weights mostly learn to make Hernia's z low for basically everything.

- `pos_weights` Mathematically
    - Multiplies the positive term by w:
        $$\ell_w = -\big[w,y \log \sigma(z) + (1-y)\log(1-\sigma(z))\big]$$
    - Standard choice w = N_neg / N_pos: positives and negatives contribute equal total weight: 119 Hernia pos x 637.5 ≈ 75,863 neg x 1 (acts as if each pos was duplicated w times)
    - Hidden cost: calibration. With full ratio weighting, a calibrated 1% probability becomes 50%. The weighted loss shifts every logit by +log w. Can be undone by subtracting log w from logits after (called prior correction).
    - If trained with `pos_weights`, raw probs can't be used in calibration analysis without the correction, which needs to be reported.

- Variance Cost: Hernia's w = 637
    - Since Hernia shows up about 1/10 batches, when it does appear, its loss term is weighted 637x. The gradient is near-zero for nine batches, then spikes on the tenth. High-var gradients make optimization noisy and can destabilize training.
    - Common compromise is sqrt weighting (w = sqrt(ratio)): Hernia gets 25 instead of 637, Infiltration gets 2.3 instead of 5.3. Still a meaninful boost for rare classes, with much milder spikes and calibration distortion.

- Prior Bias Init
    - Head starts with bias approx. 0, so every disease starts at about 50%. With unweighted loss, first few hundred steps are just learning "diseases are rare" by pushing every bias down. Early huge gradients flow back into the pretrained backbone and can damage good features before the model learns anything useful.
    - Fix is to start the bias at the answer it would learn anyway (Lin et al., 2017, the focal-loss paper, popularized this):
        $$b_k = \log\frac{\pi_k}{1-\pi_k} + \log w_k$$
    - This way, training starts at the "know-nothing" baseline and every gradient goes into learning real features

- Init-loss check now depends on the configuration
    - Prior init: expecting starting loss changes per config.
    - Whether we use (unweights or sqrt(ratio) or full ratio) + prior init is a val decision later

