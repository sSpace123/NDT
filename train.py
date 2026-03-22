# -*- coding: utf-8 -*-
"""
train.py  —  训练 (v4.1 — 多种子集成 + TTA)
  - 支持 --num_seeds 训练多个模型
  - 保存每个 seed 的最佳模型
"""
import os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import SAVE_DIR, EPOCHS, LR, WEIGHT_DECAY, SEED, denormalize_coord
from dataset import get_dataloaders
from model import NDTLocalizer
from loss import NDTLoss


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    L, C, R, B, correct, total = 0, 0, 0, 0, 0, 0
    for x, di, ri, cn in loader:
        x, di, ri, cn = x.to(device), di.to(device), ri.to(device), cn.to(device)
        out = model(x, di)
        ld = criterion(out, ri, cn)
        loss = ld["total"]
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(); continue
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        bs = x.size(0)
        L += loss.item()*bs; C += ld["cls"].item()*bs
        R += ld["reg"].item()*bs; B += ld["bnd"].item()*bs
        correct += (out["loc_logits"].argmax(-1) == ri).sum().item()
        total += bs
    n = max(total, 1)
    return {"loss":L/n, "cls":C/n, "reg":R/n, "bnd":B/n}, correct/n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    L, C, R, B, correct, total = 0, 0, 0, 0, 0, 0
    errs = []
    for x, di, ri, cn in loader:
        x, di, ri, cn = x.to(device), di.to(device), ri.to(device), cn.to(device)
        out = model(x, di)
        ld = criterion(out, ri, cn)
        loss = ld["total"]
        bs = x.size(0)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            L += loss.item()*bs; C += ld["cls"].item()*bs
            R += ld["reg"].item()*bs; B += ld["bnd"].item()*bs
        correct += (out["loc_logits"].argmax(-1) == ri).sum().item()
        total += bs
        p = denormalize_coord(out["pred_loc"].cpu().numpy())
        g = denormalize_coord(cn.cpu().numpy())
        errs.extend(np.linalg.norm(p-g, axis=-1).tolist())
    n = max(total, 1)
    return {"loss":L/n, "cls":C/n, "reg":R/n, "bnd":B/n,
            "acc":correct/n, "mae":np.mean(errs) if errs else 0}


def train_single(seed, epochs, lr, device):
    """训练单个 seed 的模型"""
    set_seed(seed)
    print(f"\n{'='*70}")
    print(f"  Seed {seed}: 训练 {epochs}ep, lr={lr}")
    print(f"{'='*70}")

    train_loader, val_loader, _ = get_dataloaders()
    model = NDTLocalizer().to(device)
    criterion = NDTLoss().to(device)
    print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    warmup = max(20, epochs // 10)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=1e-6)

    best_val, best_ep = float("inf"), 0
    ckpt_path = os.path.join(SAVE_DIR, f"best_model_seed{seed}.pt")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        if ep <= warmup:
            cur_lr = 1e-6 + (lr - 1e-6) * ep / warmup
            for pg in optimizer.param_groups: pg["lr"] = cur_lr
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vm = validate(model, val_loader, criterion, device)
        if ep > warmup: scheduler.step()
        dt = time.time() - t0

        if 0 < vm["loss"] < best_val:
            best_val, best_ep = vm["loss"], ep
            torch.save({"epoch":ep, "seed":seed,
                        "model_state_dict":model.state_dict(),
                        "val_loss":best_val, "val_mae_mm":vm["mae"]},
                       ckpt_path)

        if ep % 50 == 0 or ep == 1 or ep == epochs:
            print(f"  S{seed} E{ep:4d}/{epochs} | L={tl['loss']:.3f} | V={vm['loss']:.3f} "
                  f"MAE={vm['mae']:.1f}mm | Acc={ta:.0%}/{vm['acc']:.0%} | {dt:.1f}s")

    print(f"  Seed {seed}: Best E{best_ep}, Val={best_val:.4f}")
    # 同时保存为 best_model.pt (最后一个 seed)
    return ckpt_path, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_seeds", type=int, default=5, help="集成模型数量")
    parser.add_argument("--base_seed", type=int, default=SEED)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                           else args.device if args.device != "auto" else "cpu")
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 集成训练: {args.num_seeds} 个 seed")

    seeds = [args.base_seed + i * 7 for i in range(args.num_seeds)]
    results = []

    for seed in seeds:
        path, val = train_single(seed, args.epochs, args.lr, device)
        results.append({"seed": seed, "path": path, "val": val})

    # 保存集成信息
    import shutil
    best = min(results, key=lambda r: r["val"])
    shutil.copy(best["path"], os.path.join(SAVE_DIR, "best_model.pt"))
    with open(os.path.join(SAVE_DIR, "ensemble_info.json"), "w") as f:
        json.dump({"seeds": seeds, "models": [r["path"] for r in results],
                   "best_seed": best["seed"]}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  集成训练完成! {args.num_seeds} 个模型")
    for r in results:
        print(f"    Seed={r['seed']}, Val={r['val']:.4f}, Path={r['path']}")
    print(f"  最佳单模型: Seed={best['seed']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
