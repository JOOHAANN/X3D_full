#!/usr/bin/env python3
"""High-throughput single-GPU X3D-S training for the ETRI tensor cache."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tsn.config import get_cfg_defaults
from tsn.model.recognizers.official_x3d_recognizer import OfficialX3DRecognizer


class TensorDataset(Dataset):
    def __init__(self, directory, split, exclude_classes=()):
        self.data = np.load(directory / f"{split}_float16.npy", mmap_mode="r")
        labels = np.load(directory / f"{split}_labels.npy", mmap_mode="r")
        valid_path = directory / f"{split}_valid.npy"
        valid = np.load(valid_path, mmap_mode="r") if valid_path.exists() else np.ones(len(labels), bool)
        keep = valid & ~np.isin(labels, np.asarray(exclude_classes, dtype=labels.dtype))
        self.rows = np.flatnonzero(keep)
        self.labels = labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        return torch.from_numpy(np.array(self.data[row], copy=True)), int(self.labels[row])


class DevicePrefetchLoader:
    """Overlap the next pinned-memory transfer with GPU computation."""
    def __init__(self, loader, device):
        self.loader, self.device = loader, device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        stream = torch.cuda.Stream(device=self.device)
        source = iter(self.loader)
        def load():
            batch = next(source, None)
            if batch is None:
                return None
            with torch.cuda.stream(stream):
                return tuple(x.to(self.device, non_blocking=True) for x in batch)
        upcoming = load()
        while upcoming is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(stream)
            batch = upcoming
            for tensor in batch:
                tensor.record_stream(current)
            upcoming = load()
            yield batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=0, help="0 runs an OOM-safe autotune")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-device-prefetch", action="store_true",
                        help="Disable one-batch CUDA transfer overlap for comparison or low VRAM")
    parser.add_argument("--save-step", type=int, default=500)
    parser.add_argument("--eval-step", type=int, default=500)
    parser.add_argument("--exclude-classes", type=int, nargs="*", default=[])
    parser.add_argument("--warmup-iterations", type=int, default=400)
    parser.add_argument("--lr-steps", type=int, nargs=2, default=[6000, 10000])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_pretrained(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
    current = model.state_dict()
    loadable = {key: value for key, value in state.items()
                if key in current and getattr(value, "shape", None) == current[key].shape}
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    print(f"pretrained: loaded={len(loadable)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if len(loadable) < 100:
        raise RuntimeError("Too few pretrained tensors matched the X3D-S model")


def augment_batch(images, training):
    images = images.float()
    if not training:
        return F.interpolate(images, size=(13, 182, 182), mode="trilinear", align_corners=False)
    batch = images.shape[0]
    device = images.device
    crop_scale = torch.empty(batch, device=device).uniform_(0.70, 1.0)
    crop_ratio = torch.empty(batch, device=device).uniform_(0.85, 1.15)
    sx = (crop_scale * crop_ratio.sqrt()).clamp(max=1.0)
    sy = (crop_scale / crop_ratio.sqrt()).clamp(max=1.0)
    tx = (torch.rand(batch, device=device) * 2 - 1) * (1 - sx)
    ty = (torch.rand(batch, device=device) * 2 - 1) * (1 - sy)
    flip = torch.where(torch.rand(batch, device=device) < 0.5, -1.0, 1.0)
    angle = torch.empty(batch, device=device).uniform_(-math.pi / 18, math.pi / 18)
    cosine, sine = angle.cos(), angle.sin()
    theta = torch.zeros(batch, 3, 4, device=device)
    theta[:, 0, 0] = sx * cosine * flip
    theta[:, 0, 1] = -sy * sine
    theta[:, 1, 0] = sx * sine * flip
    theta[:, 1, 1] = sy * cosine
    theta[:, 2, 2] = 1.0
    theta[:, 0, 3] = tx
    theta[:, 1, 3] = ty
    grid = F.affine_grid(theta, (batch, 3, 13, 182, 182), align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    mean = images.mean(dim=(2, 3, 4), keepdim=True)
    contrast = torch.empty(batch, 1, 1, 1, 1, device=device).uniform_(0.9, 1.1)
    brightness = torch.empty(batch, 1, 1, 1, 1, device=device).uniform_(-0.1, 0.1)
    images = (images - mean) * contrast + mean + brightness
    erase = torch.rand(batch, 1, 1, device=device) < 0.25
    erase_h = torch.randint(18, 55, (batch, 1, 1), device=device)
    erase_w = torch.randint(18, 55, (batch, 1, 1), device=device)
    erase_y = torch.randint(0, 128, (batch, 1, 1), device=device)
    erase_x = torch.randint(0, 128, (batch, 1, 1), device=device)
    yy = torch.arange(182, device=device).view(1, 182, 1)
    xx = torch.arange(182, device=device).view(1, 1, 182)
    erase_mask = erase & (yy >= erase_y) & (yy < erase_y + erase_h) & \
        (xx >= erase_x) & (xx < erase_x + erase_w)
    images.masked_fill_(erase_mask[:, None, None], 0.0)
    images.add_(torch.randn_like(images) * 0.01)
    return images


def checkpoint_payload(model, optimizer, scheduler, scaler, iteration, best_accuracy, batch_size):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": iteration,
        "best_accuracy": best_accuracy,
        "batch_size": batch_size,
        "s5_contract": [13, 192, 6, 6],
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct1 = correct5 = count = 0
    for images, labels in loader:
        images = augment_batch(images.to(device, non_blocking=True), False)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)["probs"]
        top5 = logits.topk(5, dim=1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels[:, None]).any(dim=1).sum().item()
        count += labels.numel()
    model.train()
    return correct1 / count, correct5 / count


def autotune_batch(model, device):
    for batch_size in (384, 352, 320, 288, 256, 224, 192, 160, 128):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            images = torch.randn(batch_size, 3, 13, 182, 182, device=device, dtype=torch.float16)
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(images)["probs"], labels)
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            del images, labels, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if peak < 88.0:
                print(f"autotune selected batch={batch_size}, probe peak={peak:.1f} GiB", flush=True)
                return batch_size
            print(f"autotune rejected batch={batch_size}, probe peak={peak:.1f} GiB", flush=True)
        except torch.OutOfMemoryError:
            print(f"autotune OOM at batch={batch_size}", flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise RuntimeError("Batch-size autotune failed even at 128")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(args.config))
    model = OfficialX3DRecognizer(cfg).to(device)
    load_pretrained(model, args.pretrained)

    captured = {}
    def capture_shape(_module, _inputs, output):
        captured["shape"] = tuple(output.shape)
    hook = model.s5.register_forward_hook(capture_shape)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        model(torch.zeros(1, 3, 13, 182, 182, device=device))
    hook.remove()
    if captured["shape"][1:] != (192, 13, 6, 6):
        raise RuntimeError(f"VPOCLIP s5 contract mismatch: {captured['shape']}")
    print(f"VPOCLIP s5 contract verified: {captured['shape']}", flush=True)

    batch_size = args.batch_size or autotune_batch(model, device)
    train_set = TensorDataset(args.data_dir, "train", args.exclude_classes)
    val_set = TensorDataset(args.data_dir, "val", args.exclude_classes)
    print(f"dataset: train={len(train_set)} val={len(val_set)} "
          f"excluded_classes={sorted(args.exclude_classes)}", flush=True)
    loader_options = dict(num_workers=args.workers, pin_memory=True,
                          persistent_workers=args.workers > 0)
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                              **loader_options)
    val_loader = DataLoader(val_set, batch_size=min(batch_size * 2, 512), shuffle=False,
                            **loader_options)
    if not args.no_device_prefetch:
        train_loader = DevicePrefetchLoader(train_loader, device)
        val_loader = DevicePrefetchLoader(val_loader, device)
        print('CUDA transfer prefetch enabled (one batch ahead)', flush=True)
    learning_rate = 1e-3 * batch_size / 64
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                                weight_decay=1e-4 * batch_size / 64)
    warmup = args.warmup_iterations
    first_decay, second_decay = args.lr_steps
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        if step < first_decay:
            return 1.0
        if step < second_decay:
            return 0.1
        return 0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    start_iteration = 0
    best_accuracy = 0.0
    last_path = args.output_dir / "last_checkpoint.txt"
    if args.resume and last_path.exists():
        resume_path = Path(last_path.read_text().strip())
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_iteration = int(state["iteration"])
        best_accuracy = float(state.get("best_accuracy", 0.0))
        print(f"resumed {resume_path} at iteration {start_iteration}", flush=True)

    iterator = iter(train_loader)
    model.train()
    started = time.time()
    interval_started = started
    running_loss = running_correct = running_count = 0
    for iteration in range(start_iteration + 1, args.max_iter + 1):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels = next(iterator)
        images = augment_batch(images.to(device, non_blocking=True), True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)["probs"]
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item() * labels.numel()
        running_correct += (logits.argmax(1) == labels).sum().item()
        running_count += labels.numel()

        if iteration % 10 == 0:
            elapsed = time.time() - interval_started
            throughput = running_count / elapsed
            eta = (args.max_iter - iteration) * elapsed / 10
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"iter={iteration:06d} lr={optimizer.param_groups[0]['lr']:.6g} "
                  f"loss={running_loss/running_count:.4f} acc={running_correct/running_count:.4f} "
                  f"samples/s={throughput:.1f} peak_mem={peak:.1f}GiB eta_h={eta/3600:.2f}", flush=True)
            running_loss = running_correct = running_count = 0
            interval_started = time.time()

        should_save = iteration % args.save_step == 0 or iteration == args.max_iter
        if should_save:
            path = args.output_dir / f"model_{iteration:06d}.pth"
            torch.save(checkpoint_payload(model, optimizer, scheduler, scaler, iteration,
                                          best_accuracy, batch_size), path)
            last_path.write_text(str(path.resolve()), encoding="utf-8")
            print(f"saved {path}", flush=True)
        if iteration % args.eval_step == 0 or iteration == args.max_iter:
            top1, top5 = evaluate(model, val_loader, device)
            print(f"validation iter={iteration:06d} top1={top1:.4f} top5={top5:.4f}", flush=True)
            if top1 > best_accuracy:
                best_accuracy = top1
                torch.save(checkpoint_payload(model, optimizer, scheduler, scaler, iteration,
                                              best_accuracy, batch_size), args.output_dir / "model_best.pth")

    final_path = args.output_dir / "model_final.pth"
    torch.save(checkpoint_payload(model, optimizer, scheduler, scaler, args.max_iter,
                                  best_accuracy, batch_size), final_path)
    summary = {"iterations": args.max_iter, "batch_size": batch_size,
               "best_val_top1": best_accuracy, "hours": (time.time() - started) / 3600,
               "s5_contract": [13, 192, 6, 6]}
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
