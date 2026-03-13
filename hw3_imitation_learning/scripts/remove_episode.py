#!/usr/bin/env python3
"""Remove a single episode from a teleop zarr file.

Usage:
    python scripts/remove_episode.py datasets/raw/single_cube/teleop/2026-03-13_13-23-57/so100_transfer_cube_teleop.zarr --episode 1

Episodes are 0-indexed. Use --list to see episode boundaries.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import zarr


def get_episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return list(zip(starts.tolist(), episode_ends.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove one episode from a teleop zarr file."
    )
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to the .zarr store (e.g. datasets/raw/single_cube/teleop/.../so100_transfer_cube_teleop.zarr)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=False,
        help="0-indexed episode number to remove. Use --list to see episodes.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List episodes and exit (do not remove).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without modifying the file.",
    )
    args = parser.parse_args()

    zarr_path = args.zarr_path.resolve()
    if not zarr_path.exists():
        print(f"ERROR: {zarr_path} not found")
        return

    root = zarr.open_group(str(zarr_path), mode="r")
    ep_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    data_grp = root["data"]
    n_steps = int(ep_ends[-1])

    ranges = get_episode_ranges(ep_ends)
    print(f"Found {len(ranges)} episodes, {n_steps} total steps")
    for i, (start, end) in enumerate(ranges):
        print(f"  Episode {i}: steps {start}–{end-1} ({end - start} steps)")

    if args.list:
        return

    if args.episode is None:
        print("\nERROR: Specify --episode N to remove, or use --list to inspect.")
        return

    ep_idx = args.episode
    if ep_idx < 0 or ep_idx >= len(ranges):
        print(f"ERROR: Episode {ep_idx} out of range (0–{len(ranges)-1})")
        return

    start, end = ranges[ep_idx]
    print(f"\nRemoving episode {ep_idx} (steps {start}–{end-1}, {end - start} steps)")

    if args.dry_run:
        print("Dry run – no changes made.")
        return

    # Build new data: concatenate [0:start] and [end:]
    new_ep_ends_list: list[int] = []
    running = 0
    for i, (s, e) in enumerate(ranges):
        if i == ep_idx:
            continue
        running += e - s
        new_ep_ends_list.append(running)

    new_ep_ends = np.array(new_ep_ends_list, dtype=np.int64)

    compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)
    compressors = (compressor,)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "edited.zarr"
        out_root = zarr.open_group(str(tmp_path), mode="w", zarr_format=3)
        out_data = out_root.require_group("data")
        out_meta = out_root.require_group("meta")

        for key in data_grp:
            arr = np.asarray(data_grp[key][:n_steps])
            before = arr[:start]
            after = arr[end:]
            new_arr = np.concatenate([before, after], axis=0)
            out_data.create_array(key, data=new_arr, compressors=compressors)

        out_meta.create_array(
            "episode_ends", data=new_ep_ends, compressors=compressors
        )

        for k, v in root.attrs.items():
            out_root.attrs[k] = v

        shutil.rmtree(zarr_path)
        shutil.move(str(tmp_path), str(zarr_path))

    print(f"Done. Removed episode {ep_idx}. {len(ranges) - 1} episodes remain.")


if __name__ == "__main__":
    main()
