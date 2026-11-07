"""Fine-tune a timm backbone on Stanford Cars crops (train split), select on val.

    # main model (~45-75 min on an RTX 4070)
    python -m src.deep.train_classifier --backbone convnext_base.fb_in22k_ft_in1k --img 448
    # efficient baseline
    python -m src.deep.train_classifier --backbone resnet50.a1_in1k --img 448 --batch 24 --accum 1 --lr 3e-4
    # ablations (examples)
    python -m src.deep.train_classifier --name convnext_224 --img 224
    python -m src.deep.train_classifier --name convnext_fullimg --crop none
    python -m src.deep.train_classifier --name convnext_nomake --make-weight 0

Best checkpoint (by val top-1) -> models/classifier_<name>.pt
Per-epoch log -> runs/classify/<name>/log.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import CarsDataset, load_index
from src.deep.model import CarClassifier, build_transforms, save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mix_batch(x, y, ymake, alpha_mixup: float, alpha_cutmix: float, prob: float):
    """CutMix or MixUp (50/50). Returns mixed images, the second targets, and lam."""
    if prob <= 0 or random.random() > prob:
        return x, y, ymake, 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if random.random() < 0.5:
        lam = float(np.random.beta(alpha_cutmix, alpha_cutmix))
        h, w = x.shape[2:]
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        lam = float(np.random.beta(alpha_mixup, alpha_mixup))
        x = lam * x + (1 - lam) * x[perm]
    return x, y[perm], ymake[perm], lam


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> dict:
    model.eval()
    correct1 = correct5 = n = 0
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
            logits = model(x)
        top5 = logits.float().topk(5, 1).indices
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(1).sum().item()
        n += len(y)
    return {"val_top1": correct1 / n, "val_top5": correct5 / n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="convnext_base.fb_in22k_ft_in1k")
    ap.add_argument("--name", default=None)
    ap.add_argument("--img", type=int, default=448)
    ap.add_argument("--crop", default="gt", choices=["gt", "none"], help="train on bbox crops or full images")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps (effective batch = batch x accum)")
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone LR; heads use 10x")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2, help="warmup epochs")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--make-weight", type=float, default=0.3, help="aux make-head loss weight (0 disables)")
    ap.add_argument("--mix-prob", type=float, default=0.5, help="probability of CutMix/MixUp per batch")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing to save VRAM")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    name = args.name or f"{args.backbone.split('.')[0]}_{args.img}"
    df = load_index()
    num_makes = int(df["make_id"].max()) + 1
    train_ds = CarsDataset(df[df.split == "train"], build_transforms(args.img, True), crop=args.crop)
    val_ds = CarsDataset(df[df.split == "val"], build_transforms(args.img, False), crop=args.crop)
    loader_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, args.batch * 2, shuffle=False, **loader_kw)

    model = CarClassifier(args.backbone, num_makes=num_makes if args.make_weight > 0 else 0)
    if args.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    model = model.to(device).to(memory_format=torch.channels_last)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": head_params, "lr": args.lr * 10},
    ], weight_decay=args.wd)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = len(train_dl) // args.accum
    total_steps, warmup_steps = steps_per_epoch * args.epochs, steps_per_epoch * args.warmup
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16 and device == "cuda")

    run_dir = config.PROJECT_ROOT / "runs" / "classify" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    ckpt_path = config.MODELS_DIR / f"classifier_{name}.pt"
    hparams = {**vars(args), "name": name, "num_makes": num_makes}
    print(f"[{name}] {len(train_ds)} train / {len(val_ds)} val, device={device}, amp={amp_dtype}")

    best, step = -1.0, 0
    with open(log_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_top1", "val_top5", "lr", "epoch_sec"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, running, nb = time.time(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for i, (x, y, ymake, _) in enumerate(train_dl):
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y, ymake = y.to(device), ymake.to(device)
                x, y2, ymake2, lam = mix_batch(x, y, ymake, 0.8, 1.0, args.mix_prob)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                    logits, make_logits, _ = model(x, return_all=True)
                    ce = lambda out, t: F.cross_entropy(out.float(), t, label_smoothing=args.label_smoothing)
                    loss = lam * ce(logits, y) + (1 - lam) * ce(logits, y2)
                    if make_logits is not None:
                        loss = loss + args.make_weight * (lam * ce(make_logits, ymake) + (1 - lam) * ce(make_logits, ymake2))
                scaler.scale(loss / args.accum).backward()
                running += loss.item()
                nb += 1
                if (i + 1) % args.accum == 0:
                    scale = (step + 1) / max(1, warmup_steps) if step < warmup_steps else \
                        0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
                    for g, lr in zip(opt.param_groups, base_lrs):
                        g["lr"] = lr * scale
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1

            metrics = evaluate(model, val_dl, device, amp_dtype)
            sec = time.time() - t0
            log.writerow([epoch, f"{running / nb:.4f}", f"{metrics['val_top1']:.4f}",
                          f"{metrics['val_top5']:.4f}", f"{opt.param_groups[0]['lr']:.2e}", f"{sec:.0f}"])
            f.flush()
            flag = ""
            if metrics["val_top1"] > best:
                best = metrics["val_top1"]
                save_checkpoint(ckpt_path, model, hparams, {**metrics, "epoch": epoch})
                flag = "  * saved"
            print(f"epoch {epoch:>2}/{args.epochs}  loss {running / nb:.3f}  "
                  f"val top1 {metrics['val_top1']:.4f}  top5 {metrics['val_top5']:.4f}  {sec:.0f}s{flag}")

    print(f"Best val top-1 {best:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
