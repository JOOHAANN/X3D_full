#!/usr/bin/env python3
"""Derive the canonical 45/25/10/20 subject split from the completed 70/10/20 cache."""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


SUBJECT_RE = re.compile(r"P\d{3}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--vpo-subjects", type=int, default=45)
    parser.add_argument("--dqn-subjects", type=int, default=25)
    return parser.parse_args()


def subject_of(name):
    match = SUBJECT_RE.search(Path(str(name)).name)
    if not match:
        raise ValueError(f"No subject id in {name}")
    return match.group(0)


def link_unchanged_split(source, output, split):
    for suffix in ("float16.npy", "labels.npy", "sample_names.npy", "valid.npy", "done.npy"):
        source_path = source / f"{split}_{suffix}"
        output_path = output / f"{split}_{suffix}"
        if output_path.exists():
            continue
        os.link(source_path, output_path)
    (output / f".{split}.complete").touch()


def main():
    args = parse_args()
    source_meta = json.loads((args.source_dir / "metadata.json").read_text())
    old_train = sorted(source_meta["split_subjects"]["train"])
    if len(old_train) != args.vpo_subjects + args.dqn_subjects:
        raise ValueError("VPO+DQN counts must equal the existing training-subject count")

    rng = np.random.default_rng(args.seed)
    shuffled = list(np.asarray(old_train)[rng.permutation(len(old_train))])
    vpo_subjects = sorted(map(str, shuffled[:args.vpo_subjects]))
    dqn_subjects = sorted(map(str, shuffled[args.vpo_subjects:]))
    val_subjects = sorted(source_meta["split_subjects"]["val"])
    test_subjects = sorted(source_meta["split_subjects"]["test"])
    groups = {
        "vpo_train": vpo_subjects,
        "dqn_train": dqn_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }
    group_sets = {key: set(value) for key, value in groups.items()}
    for key, values in groups.items():
        for other_key, other_values in groups.items():
            if key < other_key and group_sets[key] & group_sets[other_key]:
                raise RuntimeError(f"Subject overlap: {key}/{other_key}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_names = np.load(args.source_dir / "train_sample_names.npy", allow_pickle=True)
    old_labels = np.load(args.source_dir / "train_labels.npy")
    old_valid = np.load(args.source_dir / "train_valid.npy", mmap_mode="r")
    old_data = np.load(args.source_dir / "train_float16.npy", mmap_mode="r")
    subjects = np.asarray([subject_of(name) for name in old_names])
    vpo_rows = np.flatnonzero(np.isin(subjects, vpo_subjects))
    dqn_rows = np.flatnonzero(np.isin(subjects, dqn_subjects))

    train_path = args.output_dir / "train_float16.npy"
    expected_shape = (len(vpo_rows),) + old_data.shape[1:]
    if train_path.exists():
        train_data = np.load(train_path, mmap_mode="r+")
        if train_data.shape != expected_shape:
            raise RuntimeError(f"Existing train cache shape mismatch: {train_data.shape}")
    else:
        train_data = np.lib.format.open_memmap(
            train_path, mode="w+", dtype=old_data.dtype, shape=expected_shape
        )
        for start in range(0, len(vpo_rows), 32):
            stop = min(start + 32, len(vpo_rows))
            train_data[start:stop] = old_data[vpo_rows[start:stop]]
            if stop % 256 == 0 or stop == len(vpo_rows):
                train_data.flush()
                print(f"copied VPO train cache: {stop}/{len(vpo_rows)}", flush=True)

    np.save(args.output_dir / "train_labels.npy", old_labels[vpo_rows])
    np.save(args.output_dir / "train_sample_names.npy", old_names[vpo_rows])
    np.save(args.output_dir / "train_valid.npy", np.asarray(old_valid[vpo_rows]))
    np.save(args.output_dir / "train_done.npy", np.ones(len(vpo_rows), dtype=bool))
    np.save(args.output_dir / "train_source_rows.npy", vpo_rows)
    np.save(args.output_dir / "dqn_labels.npy", old_labels[dqn_rows])
    np.save(args.output_dir / "dqn_sample_names.npy", old_names[dqn_rows])
    np.save(args.output_dir / "dqn_source_rows.npy", dqn_rows)
    (args.output_dir / ".train.complete").touch()
    link_unchanged_split(args.source_dir, args.output_dir, "val")
    link_unchanged_split(args.source_dir, args.output_dir, "test")

    counts = {
        "vpo_train": int(len(vpo_rows)),
        "dqn_train": int(len(dqn_rows)),
        "val": int(len(np.load(args.source_dir / "val_labels.npy", mmap_mode="r"))),
        "test": int(len(np.load(args.source_dir / "test_labels.npy", mmap_mode="r"))),
    }
    manifest = {
        "description": "Canonical ETRI subject split shared by X3D, skeleton, YOLO, and DQN",
        "seed": args.seed,
        "parent_split": str((args.source_dir / "metadata.json").resolve()),
        "subject_groups": groups,
        "sample_counts": counts,
        "zsl_unseen_labels_0_based": [9, 10, 11, 17, 49],
        "invariants": {
            "all_subject_groups_disjoint": True,
            "vpo_plus_dqn_equals_original_train": sorted(vpo_subjects + dqn_subjects) == old_train,
            "val_and_test_unchanged": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    output_meta = dict(source_meta)
    output_meta.update({
        "description": "ETRI C001 X3D cache with canonical 45/25/10/20 subject split",
        "subject_split_seed": args.seed,
        "split_subjects": groups,
        "shape": {
            "train": list(expected_shape),
            "val": list(np.load(args.source_dir / "val_float16.npy", mmap_mode="r").shape),
            "test": list(np.load(args.source_dir / "test_float16.npy", mmap_mode="r").shape),
        },
        "sample_counts": counts,
        "canonical_manifest": str(args.manifest.resolve()),
    })
    (args.output_dir / "metadata.json").write_text(json.dumps(output_meta, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
