#!/usr/bin/env python
"""Build class-filtered CS tensor and object-map datasets from the 55-class cache."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/clipgcn_tensor_cs_70_10_20")
    parser.add_argument("--split-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--object-output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


def copy_rows(source_path, output_path, rows, chunk_size):
    source = np.load(source_path, mmap_mode="r")
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=source.dtype,
        shape=(len(rows), *source.shape[1:]),
    )
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        output[start:stop] = source[rows[start:stop]]
    del output


def filter_object_dict(source_path, rows, source_count):
    source = np.load(source_path, allow_pickle=True).item()
    return {
        key: value[rows] if isinstance(value, np.ndarray) and value.ndim and len(value) == source_count else value
        for key, value in source.items()
    }


def object_rs_maps(obj, grid_size=6, max_weight=10.0):
    presence = np.asarray(obj["presence"], dtype=np.float32)
    center_xy = np.asarray(obj["center_xyz"], dtype=np.float32)[..., :2]
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis[::-1], axis, indexing="ij")
    output = np.zeros((*presence.shape, grid_size, grid_size), dtype=np.float32)
    for start in range(0, len(presence), 512):
        stop = min(start + 512, len(presence))
        x_obj = 2.0 * center_xy[start:stop, :, 0] - 1.0
        y_obj = 1.0 - 2.0 * center_xy[start:stop, :, 1]
        dx = grid_x[None, None] - x_obj[:, :, None, None]
        dy = grid_y[None, None] - y_obj[:, :, None, None]
        distance = np.clip(1.0 / (np.sqrt(dx * dx + dy * dy) + 1e-6), 0.0, max_weight)
        output[start:stop] = presence[start:stop, :, None, None] * distance
    return output


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    object_output_dir = Path(args.object_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    object_output_dir.mkdir(parents=True, exist_ok=True)

    split_metadata_path = Path(args.split_metadata)
    split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))
    seen = np.asarray(split_metadata["seen_classes"], dtype=np.int64)
    unseen = np.asarray(split_metadata["unseen_classes"], dtype=np.int64)
    selections = {
        "train": ("train", seen),
        "val": ("val", seen),
        "test": ("test", unseen),
        "test_seen": ("test", seen),
    }

    subsets = {}
    for output_split, (source_split, classes) in selections.items():
        labels = np.load(source_dir / f"{source_split}_labels.npy", mmap_mode="r")
        rows = np.flatnonzero((labels >= 0) & np.isin(labels, classes)).astype(np.int64)
        selected_labels = np.asarray(labels[rows], dtype=np.int64)
        copy_rows(
            source_dir / f"{source_split}_float16.npy",
            output_dir / f"{output_split}_float16.npy",
            rows,
            args.chunk_size,
        )
        np.save(output_dir / f"{output_split}_labels.npy", selected_labels)
        np.save(output_dir / f"{output_split}_source_indices.npy", rows)
        with open(output_dir / f"{output_split}_manifest.txt", "w", encoding="utf-8") as handle:
            for row, label in zip(rows, selected_labels):
                handle.write(f"{source_split}:{int(row)} {int(label)}\n")

        raw_objects = filter_object_dict(
            source_dir / f"{source_split}_frame7_yolov5m_objects.npy", rows, len(labels)
        )
        maps = object_rs_maps(raw_objects)
        object_path = object_output_dir / f"{output_split}_frame7_yolov5m_objects.npy"
        np.save(object_path, maps)
        object_metadata = {
            "output": str(object_path.resolve()),
            "source": str((source_dir / f"{source_split}_frame7_yolov5m_objects.npy").resolve()),
            "source_split": source_split,
            "source_indices": str((output_dir / f"{output_split}_source_indices.npy").resolve()),
            "shape": list(maps.shape),
            "dtype": str(maps.dtype),
            "source_format": "per_class_single_center",
            "grid_size": 6,
            "max_distance_weight": 10.0,
        }
        object_path.with_suffix(".metadata.json").write_text(
            json.dumps(object_metadata, indent=2), encoding="utf-8"
        )
        subsets[output_split] = {
            "source_split": source_split,
            "num_samples": int(len(rows)),
            "class_ids": sorted(np.unique(selected_labels).astype(int).tolist()),
        }
        print(f"{output_split}: {len(rows)} samples, classes={len(np.unique(selected_labels))}", flush=True)

    source_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        **source_metadata,
        "name": output_dir.name,
        "description": "Class-filtered X3D tensors using the unchanged 35/5/10 cross-subject protocol.",
        "source_dir": str(source_dir.resolve()),
        "split_metadata": str(split_metadata_path.resolve()),
        "seen_classes": seen.astype(int).tolist(),
        "unseen_classes": unseen.astype(int).tolist(),
        "label_ids_are_original": True,
        "subsets": subsets,
    }
    metadata["shape"] = {
        split: [entry["num_samples"], 3, 13, 160, 160] for split, entry in subsets.items()
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
