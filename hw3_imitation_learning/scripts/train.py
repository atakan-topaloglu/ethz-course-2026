"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    build_train_source_mix_sampler,
    chunk_dagger_recency_multipliers,
    chunk_source_from_episodes,
    episode_source_labels_from_processed_zarr,
    episode_source_labels_from_zarr_paths,
    load_and_merge_zarrs,
    load_zarr,
    log_source_mix_sanity,
)
from hw3.model import BasePolicy, build_policy


from torch.utils.data import DataLoader, random_split
import torch.optim as optim

hyperparameters_ex1 = {
    "epochs": 300,
    "batch_size": 64,
    "lr": 1e-4,
    "val_split": 0.1,
    "hidden_dim": 512,
    "n_layers": 5,
    "chunk_size": 16,
}
hyperparameters_ex2 = {
    "epochs": 300,
    "batch_size": 64,
    "lr": 1e-4,
    "val_split": 0.1,
    "hidden_dim": 512,
    "n_layers": 4,
    "chunk_size": 16,
}

hyperparameters_ex3 = {
    "epochs": 400,
    "batch_size": 256,
    "lr": 3e-4,
    "val_split": 0.1,
    "hidden_dim": 512,
    "n_layers": 6,
    "chunk_size": 16,
}


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_grad_norm_sq = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        optimizer.zero_grad()
        loss = model.compute_loss(states, action_chunks)
        
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf") 
        )
        total_grad_norm_sq += grad_norm.item() ** 2

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_grad_norm = (total_grad_norm_sq / max(n_batches, 1)) ** 0.5
    return avg_loss, avg_grad_norm


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--exercise",
        choices=["1", "2", "3"],
        default="1",
        help="Exercise number(1, 2 or 3, default: 1)",
    )
    parser.add_argument(
        "--extra-zarr",
        type=Path,
        nargs="*",
        default=None,
        help="Additional zarr paths to merge with --zarr (optional).",
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Action chunk horizon H (default: from exercise hyperparameters). Override per run if needed.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="MLP hidden dimension (default: 256). Use smaller (e.g. 128) for simpler ee_xyz.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Number of MLP layers (default: 3). Use fewer (e.g. 2) for simpler ee_xyz.",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--source-mix",
        choices=["uniform", "weighted", "batch_split"],
        default="uniform",
        help="Train-time mixing of teleop vs DAgger chunk samples. "
        "'uniform' = default DataLoader shuffle. "
        "'weighted' = WeightedRandomSampler toward --dagger-fraction. "
        "'batch_split' = each batch has round(batch*dagger_fraction) DAgger rows. "
        "Requires DAgger episodes (path contains 'dagger' or processed zarr source_zarrs).",
    )
    parser.add_argument(
        "--dagger-fraction",
        type=float,
        default=0.5,
        help="Target fraction of DAgger chunks when using weighted or batch_split (0–1).",
    )
    parser.add_argument(
        "--dagger-latest-boost",
        type=float,
        default=1.0,
        help="Within the DAgger pool, upweight newer episodes (dataset / merge order): "
        "oldest DAgger ep multiplier 1.0, newest *boost*. Values <=1 disable. "
        "DAgger chunk weights are mean-normalized so teleop vs DAgger rate still follows "
        "--dagger-fraction. Example: 3.0 with 13 DAgger episodes → linear ramp 1→3.",
    )
    parser.add_argument(
        "--no-log-mix-verify",
        action="store_true",
        help="Disable one-time [source-mix] sanity line at startup (numpy / duplicate batch sampler only).",
    )
    args = parser.parse_args()

    match args.exercise:
        case "1":
            hp = hyperparameters_ex1
        case "2":
            hp = hyperparameters_ex2
        case "3":
            hp = hyperparameters_ex3
        case _:
            raise ValueError(f"Unknown exercise: {args.exercise}")

    EPOCHS = hp["epochs"]
    BATCH_SIZE = hp["batch_size"]
    LR = hp["lr"]
    VAL_SPLIT = hp["val_split"]
    HDIM = hp["hidden_dim"]
    NLAY = hp["n_layers"]
    CHUNK = hp["chunk_size"]

    hdim = args.hidden_dim if args.hidden_dim is not None else HDIM
    nlay = args.n_layers if args.n_layers is not None else NLAY
    chunk_size = args.chunk_size if args.chunk_size is not None else CHUNK

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
        n_ep = int(np.asarray(ep_ends).size)
        ep_source = episode_source_labels_from_processed_zarr(args.zarr, n_ep)
        if ep_source is None:
            ep_source = np.zeros(n_ep, dtype=np.int64)
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
        ep_source = episode_source_labels_from_zarr_paths(zarr_paths)
    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=chunk_size,
        normalizer=normalizer,
    )
    chunk_source_full = chunk_source_from_episodes(
        ep_ends, dataset.indices, ep_source
    )
    latest_boost = max(1.0, float(args.dagger_latest_boost))
    chunk_recency_full = chunk_dagger_recency_multipliers(
        ep_ends, dataset.indices, ep_source, latest_boost
    )
    n_dag_eps = int(ep_source.sum())
    n_tel_eps = int(ep_source.size - n_dag_eps)
    print(f"Dataset: {len(dataset)} samples, chunk_size={chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")
    print(
        f"  episodes: {ep_source.size} total ({n_tel_eps} teleop, {n_dag_eps} DAgger); "
        f"chunk labels: {(chunk_source_full == 0).sum()} teleop / {(chunk_source_full == 1).sum()} DAgger"
    )

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_idx = np.asarray(train_ds.indices, dtype=np.int64)
    train_chunk_source = chunk_source_full[train_idx]
    train_recency = chunk_recency_full[train_idx].astype(np.float64, copy=False)
    train_recency_for_mix = (
        train_recency if float(args.dagger_latest_boost) > 1.0 else None
    )
    mix_gen = torch.Generator().manual_seed(args.seed)
    mix_sampler = build_train_source_mix_sampler(
        train_chunk_source,
        mode=args.source_mix,
        dagger_fraction=float(args.dagger_fraction),
        batch_size=BATCH_SIZE,
        generator=mix_gen,
        dagger_recency_mult_train=train_recency_for_mix,
    )
    if args.source_mix != "uniform" and mix_sampler is None:
        print(
            f"  source-mix={args.source_mix!r} unavailable (no mixed train chunks); "
            "using uniform shuffle."
        )

    if mix_sampler is None:
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
    elif args.source_mix == "weighted":
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=mix_sampler,
            shuffle=False,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_sampler=mix_sampler, num_workers=0
        )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    if not args.no_log_mix_verify:
        effective_mix = "uniform" if mix_sampler is None else args.source_mix
        log_source_mix_sanity(
            train_chunk_source,
            mode=effective_mix,
            dagger_fraction=float(args.dagger_fraction),
            batch_size=BATCH_SIZE,
            verify_seed=args.seed + 1_000_003,
            dagger_recency_mult_train=train_recency_for_mix,
            dagger_latest_boost=float(args.dagger_latest_boost),
        )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=chunk_size,
        hidden_dim=hdim,
        n_layers=nlay,
        state_mean=torch.as_tensor(normalizer.state_mean),
        state_std=torch.as_tensor(normalizer.state_std)
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, grad_norm = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": chunk_size,
                    "policy_type": args.policy,
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "hidden_dim": hdim,
                    "n_layers": nlay,
                    "d_model": hdim,
                    "depth": nlay,
                    "val_loss": val_loss,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | grad_norm {grad_norm:.4f}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()