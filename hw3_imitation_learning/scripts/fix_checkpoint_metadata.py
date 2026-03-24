"""Patch checkpoint metadata (hidden_dim, n_layers) inferred from state_dict.

Use when loading fails due to size mismatch: old checkpoints may lack these
fields and eval falls back to wrong defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def infer_from_state_dict(state_dict: dict) -> tuple[int, int]:
    """Infer hidden_dim and n_layers from ObstaclePolicy/MultiTaskPolicy state_dict."""
    # net.0.weight has shape [hidden_dim, state_dim]
    w0 = state_dict.get("net.0.weight")
    if w0 is None:
        raise ValueError("Checkpoint has no net.0.weight - not an ObstaclePolicy/MultiTaskPolicy?")
    hidden_dim = int(w0.shape[0])

    # Count only Linear layers (2D weight); LayerNorm has 1D weight
    linear_indices = []
    for k in state_dict:
        if k.startswith("net.") and ".weight" in k:
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                w = state_dict[k]
                if hasattr(w, "dim") and w.dim() == 2:
                    linear_indices.append(int(parts[1]))
    if not linear_indices:
        raise ValueError("No Linear layers found in state_dict")
    # n_layers = number of hidden blocks (total Linear layers - 1 for output)
    n_layers = len(linear_indices) - 1

    return hidden_dim, n_layers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch checkpoint with hidden_dim and n_layers inferred from state_dict."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint.")
    parser.add_argument("--inplace", action="store_true", help="Overwrite checkpoint in place.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: <ckpt>_fixed.pt).")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint has no model_state_dict.")

    hidden_dim, n_layers = infer_from_state_dict(state_dict)
    print(f"Inferred: hidden_dim={hidden_dim}, n_layers={n_layers}")

    ckpt["hidden_dim"] = hidden_dim
    ckpt["n_layers"] = n_layers

    if args.inplace:
        out_path = args.checkpoint
    elif args.output:
        out_path = args.output
    else:
        out_path = args.checkpoint.with_stem(args.checkpoint.stem + "_fixed")

    torch.save(ckpt, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
