# 11/05/2026
## Theory/Notes:
### Milestone 4 (Model)
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

# 11/06/2026
## Theory/Notes:
### Milestone 6 (Training Loop + First Val Experiment)
- Per Training Step
    - Forward pass -> loss -> backward() (computes gradients) -> optimizer.step() (updates weights) -> scheduler.step() (updates LRs).
    - Repeat this about 1,187 times per epoch (75,982 training images / 64 batch size, dropping the last partial batch).

- AdamW
    - Keeps a running mean and variance of each param's gradient. Each param gets its own step size since the random new head and pretrained backbone need very different update sizes.
    - "W" means weight decay is decoupled: weights shrink by lr * wd each step, no matter the gradient.
    - Plain Adam: L2 regularization rescaled by adaptive denominator -> inconsistent
    - `lr = 1e-4` is the standard for fine-tuning a pretrained CNN (`1e-3` wrechs ImageNet features)
- Warmup + Cosine
    - Warmup (500 steps, LR ramps from 1% to 100%): head starts w/ random weights, so early gradients flowing back to the pretrained backbone are large and noisy. Starting small keeps them from destroying the ImageNet features before the head learns anything useful.
    - Cosine decay to ~0: big steps early to explore, tiny steps late to settle into a minimum. Late low-LR epochs usually give the best val numbers.
    - The scheduler steps per batch, not per epoch, so the curve is smooth.

- Overfitting (Why 5 Epochs)
    - 10 epochs: .809 .825 .834 .833 .833 .833 .827 .825 .821 .820 -> Best at epoch 3
    - 5 epochs: .810 .827 .837 .840 .839 -> Best at epoch 4
    - Train loss kept falling while val loss bottome dout at epoch 4 then rose -> textbook overfitting (memorizes training data too well)

- Selecting Best Epoch on Val = Slight Optimistic Bias
    - We save the checkpoint with the highest val AUROC (works since val makes decisions)

- AUROC is a Ranking Statistic
    - AUROC = the prob that a random positive image gets a higher score than a random negative one (Mann-Whitney U statistic). Consequences:
        - Monotone transforms don't change it. Logit, sigmoid, or logit + constant all give the same AUROC. Prior correction (- log w) and calibration never move AUROC
        - Ignores prevalence - GOOD for for comparing classes | BAD for clinical usefuness

- Macro Over 13 Classes (Excluding Hernia)
    - Macro = unweighted mean of the per-class AUROCs, so rare diseases count as much as common ones. Hernia (22 val pos) leads to wild AUROC swings (one image moves it by ~0.5/22 in the worst case). Selecting it means selecting on noise, but it's still reported.


- bf16 Head Problem (found while prototyping M6):
    - bf16 has about 3 significant digits, so the head's logits got rounded -> only ~576 distinct values per class for 10,542 val images (lots of fake ties)
    - AUROC barely changed (0.83359 vs. 0.83360), since it only cares about ranking and ties get half credit
    - But logits were off by up to 0.045, which would corrupt calibration, thresholds, ensembling, and bootstrap CIs later
    - Fix: fp32Linear head (autocast disabled for just the 1024-14 layer); backbone stays bf16. Cost ~0, outputs go back to ~10,534 distinct values
    - Lesson: a metric that ignores a problem (AUROC) doesn't mean the problem isn't tehre; check raw outputs

- Seed Noise -> Pre-Declare Decision Rule
    - Two runs differing only by seed can differ by roughly 0.003-0.006 macro AUROC (the 10- vs. 5-epoch gap above is that size). Before running, write down: "sqrt is adopted only if it beats none by more than the seed-to-seed spread; otherwise none wins (ties -> none, because it keeps probabilities calibrated)." Running seed 1 for both settings gives you that spread directly.

# 11/09/2026
## Theory/Notes:
### Milestone 6 (Training Loop + First Val Experiment) [Continued]
- AUROC:
    - Logits correlate to how likely a model thinks a class is present (higher -> more likely).
    - AUROC is calculated individually for each class (14 diseases -> 14 AUROCs). Macro13 is the average of 13 of these scores (Hernia excluded).
    - AUROC:
        - Pick one pos img and one neg img and see if the pos img scored a higher logit.
        - AUROC is the probability it did out of all possible pos-neg pairs.
            - Scores: [1.0 / 0.5 / <0.5] = [Perfect ranking / coin flip / backwards ranking]

- Mann-Whitney U: Classic stats test that counts pos-neg pairs using ranks (way faster than counting pairs themselves). 
    - Example: Pos (A (2.1), C (0.3)), Neg (B (-1.5), D (0.8), E (-3.0))
    - Steps:
        1. Sort all logits from lowest to highest and number then 1, 2, 3...
            - (Logit, Rank) = [(-3.0 (E), 1), (-1.5 (B), 2), (0.3 (C), 3), (0.8 (D), 4), (2.1 (A), 5)]
        2. Add up the positives' ranks: 3 + 5 = 8
        3. Subtract smallest sum the positives could possibly have (two from the very bottom, ranks 1 + 2 = 3). The formula is n_pos(n_pos + 1) / 2 = 2 * 3/2 = 3.
            - 8 - 3 = 5 = U (exactly the # of correctly ordered pairs)
        4. Divide by total # of pairs: 5 / (2 * 3) = 0.833 (AUPRC)
    - Basically: Order and rank logits, sum positives' ranks, subtract minimum, divide by n_pos * n_neg