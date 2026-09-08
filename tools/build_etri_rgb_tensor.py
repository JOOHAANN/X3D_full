#!/usr/bin/env python3
"""Build the X3D tensor cache used by the VPOCLIP video stream.

Each camera video is a distinct sample. Modalities must be extracted from the
same video, and all cameras of a subject must remain in its assigned split.
"""

import argparse
import json
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


NAME_RE = re.compile(r"^(A\d{3})_(P\d{3})_(G\d{3})_(C\d{3})\.mp4$")
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 1, 3)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 1, 3)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="C001")
    parser.add_argument("--subject-manifest", type=Path)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_clip(item):
    row, path, frames, size = item
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return row, None, "frame_count=0"
    wanted = np.linspace(0, total - 1, frames).round().astype(np.int64)
    wanted_map = {int(frame): positions for frame, positions in _group_indices(wanted).items()}
    output = np.empty((frames, size, size, 3), dtype=np.uint8)
    found = np.zeros(frames, dtype=bool)
    frame_index = 0
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in wanted_map:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            for position in wanted_map[frame_index]:
                output[position] = rgb
                found[position] = True
            last = rgb
        frame_index += 1
        if found.all():
            break
    cap.release()
    if not found.any():
        return row, None, "decode_failed"
    if last is None:
        last = output[np.flatnonzero(found)[-1]]
    for position in np.flatnonzero(~found):
        output[position] = last
    clip = output.astype(np.float32) / 255.0
    clip = ((clip - MEAN) / STD).transpose(3, 0, 1, 2).astype(np.float16)
    return row, clip, None


def _group_indices(values):
    grouped = {}
    for position, value in enumerate(values):
        grouped.setdefault(int(value), []).append(position)
    return grouped


def discover(root, camera):
    records = []
    for path in sorted(root.rglob("*.mp4")):
        match = NAME_RE.match(path.name)
        if match and (camera == "all" or match.group(4) == camera):
            action, subject, group, selected_camera = match.groups()
            records.append(
                {
                    "path": path,
                    "relative_path": str(path.relative_to(root)),
                    "action": action,
                    "subject": subject,
                    "group": group,
                    "camera": selected_camera,
                    "label": int(action[1:]) - 1,
                }
            )
    if not records:
        raise RuntimeError(f"No {camera} MP4 files found below {root}")
    return records


def make_splits(records, seed):
    subjects = sorted({record["subject"] for record in records})
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(subjects)[rng.permutation(len(subjects))])
    train_end = round(0.70 * len(shuffled))
    val_end = train_end + round(0.10 * len(shuffled))
    subject_splits = {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }
    split_records = {
        split: [record for record in records if record["subject"] in selected]
        for split, selected in subject_splits.items()
    }
    return subject_splits, split_records


def build_split(name, records, output_dir, frames, size, workers, overwrite):
    output_path = output_dir / f"{name}_float16.npy"
    labels_path = output_dir / f"{name}_labels.npy"
    names_path = output_dir / f"{name}_sample_names.npy"
    done_path = output_dir / f"{name}_done.npy"
    valid_path = output_dir / f"{name}_valid.npy"
    complete_path = output_dir / f".{name}.complete"

    if complete_path.exists() and not overwrite:
        print(f"{name}: complete, skipping")
        return
    if overwrite:
        for path in (output_path, labels_path, names_path, done_path, valid_path, complete_path):
            path.unlink(missing_ok=True)

    count = len(records)
    if output_path.exists():
        output = np.load(output_path, mmap_mode="r+")
        if output.shape != (count, 3, frames, size, size):
            raise RuntimeError(f"Existing cache has wrong shape: {output.shape}")
    else:
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float16, shape=(count, 3, frames, size, size)
        )
    if done_path.exists():
        done = np.load(done_path, mmap_mode="r+")
        valid = np.load(valid_path, mmap_mode="r+")
    else:
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.bool_, shape=(count,))
        valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.bool_, shape=(count,))
        done[:] = False
        valid[:] = False

    np.save(labels_path, np.asarray([record["label"] for record in records], dtype=np.int64))
    np.save(names_path, np.asarray([record["relative_path"] for record in records]))
    pending_rows = [row for row in range(count) if not done[row]]
    failures = []
    max_pending = max(workers * 2, 1)
    with ThreadPoolExecutor(max_workers=workers) as executor, tqdm(
        total=count, initial=count - len(pending_rows), desc=f"building {name}"
    ) as progress:
        futures = {}
        iterator = iter(pending_rows)
        for _ in range(min(max_pending, len(pending_rows))):
            row = next(iterator)
            futures[executor.submit(decode_clip, (row, records[row]["path"], frames, size))] = row
        completed_since_flush = 0
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                row = futures.pop(future)
                try:
                    _, clip, error = future.result()
                except Exception as exc:  # keep the long build resumable
                    clip, error = None, repr(exc)
                if clip is not None:
                    output[row] = clip
                    valid[row] = True
                else:
                    output[row] = 0
                    valid[row] = False
                    failures.append((records[row]["relative_path"], error))
                done[row] = True
                completed_since_flush += 1
                progress.update(1)
                try:
                    next_row = next(iterator)
                except StopIteration:
                    next_row = None
                if next_row is not None:
                    futures[executor.submit(
                        decode_clip, (next_row, records[next_row]["path"], frames, size)
                    )] = next_row
                if completed_since_flush >= 128:
                    output.flush(); done.flush(); valid.flush()
                    completed_since_flush = 0
    output.flush(); done.flush(); valid.flush()
    (output_dir / f"{name}_failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    complete_path.touch()
    print(f"{name}: {int(valid.sum())}/{count} valid")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = discover(args.rgb_root.resolve(), args.camera)
    subject_splits, split_records = make_splits(records, args.seed)
    if args.subject_manifest:
        canonical = json.loads(args.subject_manifest.read_text())
        groups = canonical["subject_groups"]
        assert [len(groups[k]) for k in ("vpo_train", "dqn_train", "val", "test")] == [45, 25, 10, 20]
        assert len(set.union(*(set(v) for v in groups.values()))) == 100
        subject_splits = {"train": groups["vpo_train"], "dqn": groups["dqn_train"],
                          "val": groups["val"], "test": groups["test"]}
        split_records = {s: [r for r in records if r["subject"] in people]
                         for s, people in subject_splits.items()}
        assert sum(map(len, split_records.values())) == len(records)
        for s, items in split_records.items():
            np.save(args.output_dir / f"{s}_sample_names.npy", np.asarray([r["relative_path"] for r in items]))
            np.save(args.output_dir / f"{s}_labels.npy", np.asarray([r["label"] for r in items]))
    metadata = {
        "description": "ETRI RGB per-camera X3D cache for the VPOCLIP trimodal stream",
        "source_root": str(args.rgb_root.resolve()),
        "camera": args.camera,
        "classes": [f"A{index:03d}" for index in range(1, 56)],
        "class_to_idx": {f"A{index:03d}": index - 1 for index in range(1, 56)},
        "split_rule": "canonical 45/25/10/20" if args.subject_manifest else "70/10/20",
        "canonical_manifest": str(args.subject_manifest),
        "split_seed": args.seed,
        "split_subjects": subject_splits,
        "shape": {
            split: [len(items), 3, args.frames, args.size, args.size]
            for split, items in split_records.items()
        },
        "dtype": "float16",
        "normalization": {"mean": MEAN.reshape(-1).tolist(), "std": STD.reshape(-1).tolist()},
        "vpoclip_s5_contract": [13, 192, 6, 6],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    for split in ("train", "val", "test"):
        build_split(split, split_records[split], args.output_dir, args.frames, args.size,
                    args.workers, args.overwrite)


if __name__ == "__main__":
    main()
