"""Train one DenseNet-121 run on NIH train, select the best epoch on val.

Pipeline: frozen split (splits/nih_split.csv) -> load_dev_labels (train/val only)
-> NIHDataset + transforms -> DenseNet-121 with prior-initialized FP32 head
-> BCE (optional pos_weight) -> per-epoch val AUROC -> best checkpoint + val logits
-> one row appended to research/runs.csv.

This script never touches the test set. Test evaluation happens only in
scripts/run_test.py, once, after the analysis plan is frozen.
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.data import NIH_DISEASES, build_image_paths, load_dev_labels
from src.dataset import NIHDataset
from src.losses import build_loss, compute_pos_weight, compute_prevalence, prior_logits
from src.metrics import macro_auroc, per_class_auroc
from src.model import build_model, init_head_bias
from src.paths import CHECKPOINT_DIR, DEV_OUTPUT_DIR, PROJECT_ROOT, RESEARCH_DIR, SPLIT_CSV
from src.transforms import build_transforms

DEVICE = "cuda"

# Saved next to val logits for milestone 7-8: Patient ID -> patient-level bootstrap CIs,
# View Position -> AP/PA confound checks, Follow-up # -> follow-up/drain analysis
META_COLS = ["Image Index", "Patient ID", "View Position", "Follow-up #"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DenseNet-121 on NIH train, select on val.")
    p.add_argument("--pos-weight", choices=["none", "sqrt", "ratio"], default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=500) # 500-step linear warmup -> protects pretrained features from big early updates while head is still somewhat random
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--smoke", action="store_true", help="Tiny run to test the plumbing; never logged to research/") # 30 train / 10 val steps, 1 epoch
    return p.parse_args()


# Only src/ counts as "dirty"; edits to notebooks or journal.md don't affect training
def git_state() -> tuple[str, bool]:
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--", "src"))


def make_loaders(nih_dir: Path, batch_size: int, num_workers: int, seed: int):
    df = load_dev_labels(nih_dir, SPLIT_CSV) # train/val only; raises if a test row ever sneaks in
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    paths = build_image_paths(nih_dir)

    train_ds = NIHDataset(train_df, paths, build_transforms(train=True))
    val_ds = NIHDataset(val_df, paths, build_transforms(train=False))

    # persistent_workers: keep workers alive between epochs (Windows spawn startup is slow)
    # pin_memory: page-locked host memory, needed for non_blocking=True copies to overlap
    common = dict(pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, # drop_last=True keeps batch size constant
        num_workers=num_workers, generator=torch.Generator().manual_seed(seed), **common, # Seeded generator makes shuffle order reproducible
    )

    # Val: batch * 2 is safe since there are no gradients; workers // 4 because 16+16 workers exhausted Windows commit charge (~1GB each) and hung a run
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers // 4, **common)
    return train_df, val_df, train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scheduler, max_steps=None) -> float:
    model.train()
    total = torch.zeros((), device=DEVICE)
    n = 0
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last) # channels_last for speed
        y = y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): # bf16 also for speed
            logits = model(x)
        loss = criterion(logits.float(), y) # Loss computed on logits.float() so BCE runs in fp32

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step() # Scheduler steps per batch, not per epoch: warmup and cosine are defined in steps

        total += loss.detach() * len(x) # total lives on the GPU; loss.detach() so there's no .item() sync every step
        n += len(x)
        if step % 200 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.2e}", flush=True) # .item() runs only every 200 steps for the print
    return (total / n).item()


@torch.no_grad()
def evaluate(model, loader, criterion, max_steps=None):
    model.eval() # eval(): BatchNorm uses running stats instead of batch stats (DenseNet-121 has no dropout by default)
    total = torch.zeros((), device=DEVICE)
    n = 0
    all_logits, all_labels = [], []
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        total += criterion(logits, y) * len(x)
        n += len(x)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    return (total / n).item(), torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


# Raw logits, not probs. Weighted (sqrt/ratio) runs need prior correction (- log pos_weight) before calibration/threshold analysis.
def save_val_logits(val_df: pd.DataFrame, logits: np.ndarray, labels: np.ndarray, path: Path) -> None:
    rows = val_df.iloc[: len(logits)].reset_index(drop=True) # iloc[:len(logits)] handles smoke runs, which only evaluate the first 10 val batches.
    if not np.array_equal(labels, rows[NIH_DISEASES].to_numpy(dtype=np.float32)): # Guard that checks loader's labels against val_df before writing
        raise RuntimeError("Loader order doesn't match val_df; logits would be saved against the wrong images")
    logit_cols = pd.DataFrame(logits, columns=[f"logit_{d}" for d in NIH_DISEASES])
    pd.concat([rows[META_COLS], logit_cols], axis=1).to_csv(path, index=False, float_format="%.6f")


# Schema guard; refuses to mix column layouts in runs.csv
def append_run(row: dict, path: Path) -> None:
    new = pd.DataFrame([row])
    if path.exists():
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != new.columns.tolist():
            raise RuntimeError(f"{path.name} columns changed; start a new runs file instead of mixing schemas")
    new.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    nih_dir = Path(os.getenv("DATA_ROOT")) / "NIH_Chest_X-Rays"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # benchmark=True picks the fastest conv algorithms but is non-deterministic, so a seed fixes data order + head init, not bit-identical results. Seed spread includes this noise.
    torch.backends.cudnn.benchmark = True

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_id = f"{'smoke' if args.smoke else stamp}_{args.pos_weight}_s{args.seed}"
    out_dir = DEV_OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{run_id}.pt"
    runs_path = out_dir / "runs.csv" if args.smoke else RESEARCH_DIR / "runs.csv"
    max_train, max_val = (30, 10) if args.smoke else (None, None) # 30 train / 10 val if smoke test
    epochs = 1 if args.smoke else args.epochs

    train_df, val_df, train_loader, val_loader = make_loaders(
        nih_dir, args.batch_size, args.num_workers, args.seed,
    )

    prevalence = compute_prevalence(train_df)
    pos_weight = compute_pos_weight(train_df, mode=args.pos_weight)
    model = build_model()
    init_head_bias(model, prior_logits(prevalence, pos_weight)) # Prior-init head bias (step-0 starts at know-nothing baseline) -> don't waste early steps learning prevalences
    model = model.to(DEVICE, memory_format=torch.channels_last)
    criterion = build_loss(pos_weight).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_steps), # LR: 1e-6 -> 1e-4 over 500 warmup steps
            CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps), # Then cosine to 0 at the final step (1187 steps/epoch)
        ],
        # Smoke runs build schedule from the full epoch length, so 30 smoke steps never leave warmup.
        milestones=[args.warmup_steps],
    )

    commit, src_dirty = git_state() # Record which commit produced it + whether src/ had uncommitted edits. Commit before real runs
    config = {**vars(args), "run_id": run_id, "git_commit": commit, "src_dirty": src_dirty}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"run {run_id}  commit {commit}{' (src dirty)' if src_dirty else ''}  {len(train_loader)} steps/epoch")

    history = []
    best = {"macro": -1.0}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, max_train)

        # With pos_weight != none this val loss is weighted, so it's NOT comparable across none vs sqrt runs. Compare runs on AUROC only.
        val_loss, logits, labels = evaluate(model, val_loader, criterion, max_val)
        aucs = per_class_auroc(logits, labels)
        macro = macro_auroc(aucs)
        seconds = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_auroc13": macro, "seconds": round(seconds)})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(f"epoch {epoch}  train {train_loss:.4f}  val {val_loss:.4f}  macro13 {macro:.4f}  {seconds:.0f}s", flush=True)

        if macro > best["macro"]: # Best checkpoint chosen on val macro13
            best = {"macro": macro, "epoch": epoch, "aucs": aucs} # If macro were NaN (class w/ no positives), NaN > x is always False and best would never be set -> KeyError at best["epoch"]. Can't happen on full val.
            torch.save(model.state_dict(), ckpt_path)
            save_val_logits(val_df, logits, labels, out_dir / "val_logits.csv") # Saves val logits

    row = { # Hernia AUROC is reported here but never used for selection (22 val positives)
        "run_id": run_id, "git_commit": commit, "src_dirty": src_dirty,
        "pos_weight": args.pos_weight, "seed": args.seed, "epochs": epochs,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps, "best_epoch": best["epoch"],
        "val_macro_auroc13": round(best["macro"], 4),
        **{f"auc_{d}": round(v, 4) for d, v in best["aucs"].items()},
    }
    append_run(row, runs_path)
    print(f"best epoch {best['epoch']}  macro13 {best['macro']:.4f}  -> {runs_path}")


# Windows starts DataLoader workers with spawn, which reimports this file in every worker.
# Without this guard, every worker would start its own training run
if __name__ == "__main__":
    main()
