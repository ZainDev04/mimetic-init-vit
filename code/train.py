# Train a ViT on CIFAR-10, with the default init (base) or mimetic init.
# Settings follow Section 5.1 of the paper: AdamW, lr 3e-3, weight decay 0.01,
# batch 512, 100 epochs, RandAugment + Cutout. Unstated parts are our choices
# (see Training settings in the README): 5 warmup epochs then cosine decay, RandAugment(2, 9), Cutout 16x16.
#
# Example (Colab):
#   python train.py --init mimetic --out /content/drive/MyDrive/ct468/init_192
# A checkpoint is saved every epoch. Running the same command again resumes from it.

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from model import ViT
from mimetic_init import mimetic_init_

MEAN = (0.4914, 0.4822, 0.4465)   # CIFAR-10 per-channel mean and std of the training set
STD = (0.2470, 0.2435, 0.2616)


class Cutout:
    # Set one random size x size square of the image to zero (DeVries and Taylor, 2017).
    # The center can be near the edge, so part of the square may fall outside the image.
    def __init__(self, size=16):
        self.size = size

    def __call__(self, img):                     # img: tensor (3, H, W)
        _, H, W = img.shape
        cy = torch.randint(H, (1,)).item()
        cx = torch.randint(W, (1,)).item()
        y0, y1 = max(cy - self.size // 2, 0), min(cy + self.size // 2, H)
        x0, x1 = max(cx - self.size // 2, 0), min(cx + self.size // 2, W)
        img = img.clone()
        img[:, y0:y1, x0:x1] = 0
        return img


def get_loaders(batch_size, data_dir="./data", workers=2):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),             # shift the image by up to 4 pixels
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),   # 2 random ops (rotate, contrast, ...) at strength 9
        T.ToTensor(),                            # PIL image -> float tensor in [0, 1], shape (3, 32, 32)
        T.Normalize(MEAN, STD),
        Cutout(16),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    # drop_last keeps every training step the same size; the test set uses every image.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                               num_workers=workers, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=True)
    return train_loader, test_loader


def lr_at(step, total_steps, warmup_steps, base_lr):
    # Linear warmup from 0 to base_lr, then cosine decay from base_lr to 0.
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)   # goes 0 -> 1
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, scaler, step, total_steps, warmup_steps, base_lr, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Set the learning rate by hand for this step.
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, total_steps, warmup_steps, base_lr)
        # Mixed precision: the forward pass runs in float16 on the GPU, which is faster.
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = model(x)
            loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        # The scaler multiplies the loss before backward so small float16 gradients do not become 0,
        # then divides them back before the optimizer step.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, 100 * correct / seen, step


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, seen = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    return 100 * correct / seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", choices=["base", "mimetic"], required=True)
    p.add_argument("--out", required=True, help="folder for checkpoint and log, e.g. on Google Drive")
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5, help="warmup epochs")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pos-scale", type=float, default=1.0, help="multiply position embeddings (gamma)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    log_path = os.path.join(args.out, "log.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.batch)
    model = ViT(dim=args.dim, depth=args.depth, heads=args.heads, pos_scale=args.pos_scale)
    if args.init == "mimetic":
        mimetic_init_(model)          # base keeps the PyTorch default init
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch

    # Resume if a checkpoint from an earlier (disconnected) session exists.
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        torch.set_rng_state(ckpt["rng"].cpu())   # map_location moved it to the GPU; it must be on the CPU
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")
    else:
        with open(log_path, "w") as f:
            f.write("epoch,lr,train_loss,train_acc,test_acc,seconds\n")

    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc, step = train_one_epoch(model, train_loader, optimizer, scaler, step,
                                                      total_steps, warmup_steps, args.lr, device)
        test_acc = evaluate(model, test_loader, device)
        seconds = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        gpu_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(f"epoch {epoch + 1}/{args.epochs}  lr {lr:.2e}  loss {train_loss:.3f}  "
              f"train {train_acc:.2f}  test {test_acc:.2f}  {seconds:.0f}s  peak GPU {gpu_gb:.1f} GB")
        with open(log_path, "a") as f:
            f.write(f"{epoch + 1},{lr:.6f},{train_loss:.4f},{train_acc:.2f},{test_acc:.2f},{seconds:.1f}\n")
        # Write to a temporary file first, so a disconnect during saving cannot corrupt the checkpoint.
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "rng": torch.get_rng_state(), "epoch": epoch,
                    "args": vars(args)}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    print(f"done, results in {log_path}")


if __name__ == "__main__":
    main()
