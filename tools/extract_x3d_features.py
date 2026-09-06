# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsn.config import get_cfg_defaults  # noqa: E402
from tsn.data.datasets.clipgcn_tensor_dataset import CLIPGCNTensorDataset  # noqa: E402
from tsn.model.recognizers.build import build_recognizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract intermediate X3D feature maps from CLIPGCN tensor data."
    )
    parser.add_argument(
        "--config",
        default="configs/x3d-s_clipgcn_tensor_cross_subject_70_10_20.yaml",
        help="Path to the X3D config file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/x3d-s_clipgcn_tensor_cs_70_10_20/model_006500.pth",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/clipgcn_tensor_cs_70_10_20",
        help="Directory containing train_float16.npy and train_labels.npy.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to extract.")
    parser.add_argument(
        "--output",
        default="outputs/x3d-s_clipgcn_tensor_cs_70_10_20/train_s5_features.npy",
        help="Output .npy path.",
    )
    parser.add_argument(
        "--layer",
        default="s5",
        help="Module name to hook, for example: s5, head.conv_5, head.relu.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for the output feature npy.",
    )
    parser.add_argument(
        "--layout",
        choices=("btchw", "bcthw"),
        default="btchw",
        help="Output layout. btchw is [B, T, C, H, W].",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=None,
        help="Optionally resize feature maps to spatial-size x spatial-size.",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Optionally reduce the channel dimension to this value.",
    )
    parser.add_argument(
        "--channel-reduction",
        choices=("adaptive_avg", "first"),
        default="adaptive_avg",
        help="How to reduce channels when --num-channels is set.",
    )
    parser.add_argument(
        "--tensor-resize-size",
        type=int,
        default=None,
        help="Input tensor resize size before X3D. Defaults to the config split setting.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the full split.",
    )
    parser.add_argument(
        "--save-sidecars",
        action="store_true",
        help="Also save labels, valid source indices, and metadata next to the feature file.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_module(model, dotted_name):
    module = model
    for part in dotted_name.split("."):
        if not hasattr(module, part):
            raise ValueError(f"Model has no module '{dotted_name}' at '{part}'")
        module = getattr(module, part)
    return module


def reduce_channels(features, num_channels, mode):
    if num_channels is None or features.shape[1] == num_channels:
        return features
    if features.shape[1] < num_channels:
        raise ValueError(
            f"Cannot reduce channel dimension from {features.shape[1]} to {num_channels}"
        )
    if mode == "first":
        return features[:, :num_channels]

    bsz, channels, frames, height, width = features.shape
    features = features.permute(0, 2, 3, 4, 1).reshape(-1, 1, channels)
    features = F.adaptive_avg_pool1d(features, num_channels)
    return features.reshape(bsz, frames, height, width, num_channels).permute(
        0, 4, 1, 2, 3
    )


def adapt_features(features, args):
    if features.ndim != 5:
        raise ValueError(f"Expected 5D feature tensor [B,C,T,H,W], got {tuple(features.shape)}")
    features = reduce_channels(features, args.num_channels, args.channel_reduction)
    if args.spatial_size is not None and tuple(features.shape[-2:]) != (
        args.spatial_size,
        args.spatial_size,
    ):
        bsz, channels, frames, height, width = features.shape
        features = features.permute(0, 2, 1, 3, 4).reshape(
            bsz * frames, channels, height, width
        )
        features = F.interpolate(
            features,
            size=(args.spatial_size, args.spatial_size),
            mode="bilinear",
            align_corners=False,
        )
        features = features.reshape(
            bsz, frames, channels, args.spatial_size, args.spatial_size
        ).permute(0, 2, 1, 3, 4)
    if args.layout == "btchw":
        features = features.permute(0, 2, 1, 3, 4).contiguous()
    return features.contiguous()


def build_cfg(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(resolve_path(args.config)))
    cfg.defrost()
    cfg.NUM_GPUS = 0
    cfg.MODEL.PRETRAINED = str(resolve_path(args.checkpoint))
    cfg.freeze()
    return cfg


def get_tensor_resize_size(cfg, args):
    if args.tensor_resize_size is not None:
        return args.tensor_resize_size
    if args.split == "train":
        return int(cfg.TRANSFORM.TRAIN.TENSOR_RESIZE_SIZE)
    return int(cfg.TRANSFORM.TEST.TENSOR_RESIZE_SIZE)


def main():
    args = parse_args()
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    device = torch.device(args.device)
    model = build_recognizer(cfg, device=device)
    model.eval()

    captured = {}

    def hook(_module, _inputs, output):
        captured["features"] = output.detach()

    handle = get_module(model, args.layer).register_forward_hook(hook)

    dataset = CLIPGCNTensorDataset(
        str(resolve_path(args.data_dir)),
        is_train=False,
        split=args.split,
        mean=cfg.TRANSFORM.MEAN,
        std=cfg.TRANSFORM.STD,
        tensor_resize_size=get_tensor_resize_size(cfg, args),
    )
    sample_count = len(dataset)
    if args.max_samples is not None:
        sample_count = min(sample_count, args.max_samples)
        dataset = torch.utils.data.Subset(dataset, range(sample_count))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_dtype = np.float16 if args.dtype == "float16" else np.float32
    features_out = None
    labels = []

    with torch.no_grad():
        write_start = 0
        for clips, targets in tqdm(loader, desc=f"Extracting {args.layer}"):
            captured.clear()
            clips = clips.to(device=device, non_blocking=True)
            _ = model(clips)
            if "features" not in captured:
                raise RuntimeError(f"Hook for layer '{args.layer}' did not capture any features")

            features = adapt_features(captured["features"], args).cpu().numpy().astype(
                output_dtype, copy=False
            )
            if features_out is None:
                final_shape = (sample_count,) + tuple(features.shape[1:])
                features_out = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=output_dtype,
                    shape=final_shape,
                )
                print(f"Writing {output_path} with shape {final_shape} and dtype {output_dtype}")

            write_end = write_start + features.shape[0]
            features_out[write_start:write_end] = features
            labels.append(targets.numpy())
            write_start = write_end

    handle.remove()
    if features_out is not None:
        features_out.flush()

    if args.save_sidecars:
        labels = np.concatenate(labels, axis=0)
        np.save(output_path.with_suffix(".labels.npy"), labels)
        valid_indices = (
            dataset.dataset.valid_indices
            if isinstance(dataset, torch.utils.data.Subset)
            else dataset.valid_indices
        )
        np.save(
            output_path.with_suffix(".valid_indices.npy"),
            np.asarray(valid_indices[:sample_count]),
        )
        metadata = {
            "config": str(resolve_path(args.config)),
            "checkpoint": str(resolve_path(args.checkpoint)),
            "data_dir": str(resolve_path(args.data_dir)),
            "split": args.split,
            "layer": args.layer,
            "layout": args.layout,
            "shape": list(features_out.shape) if features_out is not None else [],
            "dtype": str(output_dtype),
            "spatial_size": args.spatial_size,
            "num_channels": args.num_channels,
            "channel_reduction": args.channel_reduction,
            "tensor_resize_size": get_tensor_resize_size(cfg, args),
        }
        output_path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    print("Done.")


if __name__ == "__main__":
    main()
