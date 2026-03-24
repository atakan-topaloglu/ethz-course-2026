"""Dataset utilities for SO-100 teleop imitation learning."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


def _parse_key_spec(spec: str) -> tuple[str, slice]:
    """Parse a key spec like ``"state_cube[:3]"`` into (key, col_slice).

    Supports slicing notations: ``key``, ``key[:N]``, ``key[M:]``, ``key[M:N]``.
    Returns the array name and a column slice to apply on axis=1.
    """
    if "[" not in spec:
        return spec, slice(None)
    name, rest = spec.split("[", 1)
    rest = rest.rstrip("]")
    parts = rest.split(":")
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        return name, slice(start, stop)
    raise ValueError(
        f"Invalid key spec: {spec!r}  (expected 'key', 'key[:N]', 'key[M:]', or 'key[M:N]')"
    )


def load_zarr(
    zarr_path: Path,
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load states, actions, and episode_ends from a processed .zarr.

    Args:
        zarr_path: Path to the processed .zarr store.
        state_keys: List of data array key specs to concatenate as the state.
            Each entry can include an optional column slice, e.g.
            ``["state_ee_xyz", "state_cube[:3]"]``.
            If ``None``, falls back to the ``state_key`` attribute in the zarr metadata.
        action_keys: List of data array key specs to concatenate as the action.
            Supports column slicing, e.g. ``["action_ee_xyz", "action_gripper"]``.
            If ``None``, falls back to the ``action_key`` attribute in the zarr metadata.

    Returns:
        states, actions, episode_ends
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    data = root["data"]

    # ── states: concatenate one or more arrays ────────────────────────
    if state_keys is None:
        sk = root.attrs.get("state_key", "state")
        state_keys = [sk]

    state_parts: list[np.ndarray] = []
    for spec in state_keys:
        name, col_slice = _parse_key_spec(spec)
        arr = np.asarray(data[name][:], dtype=np.float32)
        state_parts.append(arr[:, col_slice] if col_slice != slice(None) else arr)
    states = (
        np.concatenate(state_parts, axis=1) if len(state_parts) > 1 else state_parts[0]
    )

    # ── actions: concatenate one or more arrays ───────────────────────
    if action_keys is None:
        ak = root.attrs.get("action_key", "action")
        action_keys = [ak]

    action_parts: list[np.ndarray] = []
    for spec in action_keys:
        act_name, act_slice = _parse_key_spec(spec)
        arr = np.asarray(data[act_name][:], dtype=np.float32)
        action_parts.append(arr[:, act_slice] if act_slice != slice(None) else arr)
    actions = (
        np.concatenate(action_parts, axis=1)
        if len(action_parts) > 1
        else action_parts[0]
    )

    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

    return states, actions, episode_ends


def load_and_merge_zarrs(
    zarr_paths: list[Path],
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate data from multiple processed .zarr stores.

    Each zarr store is loaded independently via :func:`load_zarr` and the
    results are concatenated.  Episode-end indices are shifted so they remain
    globally correct after concatenation.

    Returns the same ``(states, actions, episode_ends)`` tuple
    as :func:`load_zarr`.
    """
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_ep_ends: list[np.ndarray] = []
    offset = 0

    for zp in zarr_paths:
        states, actions, ep_ends = load_zarr(
            zp, state_keys=state_keys, action_keys=action_keys,
        )
        all_states.append(states)
        all_actions.append(actions)
        all_ep_ends.append(ep_ends + offset)
        offset += states.shape[0]

    merged_states = np.concatenate(all_states, axis=0)
    merged_actions = np.concatenate(all_actions, axis=0)
    merged_ep_ends = np.concatenate(all_ep_ends, axis=0)

    return merged_states, merged_actions, merged_ep_ends


def path_looks_like_dagger(path: Path | str) -> bool:
    """Heuristic: DAgger-collected zarrs are expected to live under a *dagger* path."""
    return "dagger" in str(path).lower()


def episode_source_labels_from_zarr_paths(zarr_paths: list[Path]) -> np.ndarray:
    """One label per episode in merge order: 0 = teleop, 1 = DAgger.

    Episode order matches :func:`load_and_merge_zarrs` (concatenation order of *zarr_paths*).
    """
    labels: list[int] = []
    for zp in zarr_paths:
        root = zarr.open_group(str(zp), mode="r")
        n_ep = int(np.asarray(root["meta"]["episode_ends"]).size)
        tag = 1 if path_looks_like_dagger(zp) else 0
        labels.extend([tag] * n_ep)
    return np.asarray(labels, dtype=np.int64)


def episode_source_labels_from_processed_zarr(
    processed_path: Path, n_episodes: int
) -> np.ndarray | None:
    """Recover per-episode teleop vs DAgger labels using ``source_zarrs`` metadata.

    Written by ``scripts/compute_actions.py`` when merging multiple raw stores. If paths
    are missing or episode counts disagree with *n_episodes*, returns ``None``.
    """
    root = zarr.open_group(str(processed_path), mode="r")
    src = root.attrs.get("source_zarrs")
    if not src:
        return None
    labels: list[int] = []
    proc_dir = processed_path.resolve().parent
    for p in src:
        raw = Path(p)
        candidates = [raw, proc_dir / raw.name, raw.expanduser()]
        found: Path | None = None
        for cand in candidates:
            if cand.exists():
                found = cand
                break
        if found is None:
            return None
        try:
            r = zarr.open_group(str(found), mode="r")
            n_ep = int(np.asarray(r["meta"]["episode_ends"]).size)
        except (KeyError, ValueError, OSError):
            return None
        tag = 1 if path_looks_like_dagger(p) else 0
        labels.extend([tag] * n_ep)
    out = np.asarray(labels, dtype=np.int64)
    if int(out.size) != int(n_episodes):
        return None
    return out


def chunk_source_from_episodes(
    episode_ends: np.ndarray,
    chunk_start_indices: np.ndarray,
    episode_source: np.ndarray,
) -> np.ndarray:
    """Label each chunk row (valid start index *t*) with the source of its episode.

    *episode_source* has shape ``(n_episodes,)`` with 0 = teleop, 1 = DAgger.
    """
    if episode_source.size != episode_ends.size:
        raise ValueError(
            f"episode_source length {episode_source.size} != episode_ends length {episode_ends.size}"
        )
    ep_idx = np.searchsorted(episode_ends, chunk_start_indices, side="right")
    return episode_source[ep_idx].astype(np.int64, copy=False)


def dagger_episode_recency_multipliers(
    episode_source: np.ndarray, latest_boost: float
) -> np.ndarray:
    """Per-episode multipliers: teleop = 1.0; DAgger linear from 1 (oldest) to *latest_boost* (newest).

    Episode order is the index order in *episode_source* (merge / zarr order). Only DAgger
    episodes are ranked among themselves (0 … K−1).
    """
    n = int(episode_source.size)
    mult = np.ones(n, dtype=np.float64)
    boost = float(latest_boost)
    if boost <= 1.0 or n == 0:
        return mult
    k = int(episode_source.sum())
    if k <= 1:
        return mult
    j = 0
    for i in range(n):
        if int(episode_source[i]) == 1:
            mult[i] = 1.0 + (j / (k - 1.0)) * (boost - 1.0)
            j += 1
    return mult


def chunk_dagger_recency_multipliers(
    episode_ends: np.ndarray,
    chunk_start_indices: np.ndarray,
    episode_source: np.ndarray,
    latest_boost: float,
) -> np.ndarray:
    """Per chunk-start row: multiplier from :func:`dagger_episode_recency_multipliers` for its episode."""
    ep_m = dagger_episode_recency_multipliers(episode_source, latest_boost)
    ep_i = np.searchsorted(episode_ends, chunk_start_indices, side="right")
    return ep_m[ep_i]


def _normalize_dagger_recency_train(
    chunk_source_train: np.ndarray, mult_per_train_sample: np.ndarray
) -> np.ndarray:
    """Keep teleop at 1.0; scale DAgger rows so their mean is 1 (preserves teleop/DAgger balance)."""
    out = np.ones(len(chunk_source_train), dtype=np.float64)
    mask = chunk_source_train == 1
    if not np.any(mask):
        return out
    d = np.asarray(mult_per_train_sample, dtype=np.float64)[mask]
    m = float(d.mean())
    if m <= 0.0:
        return out
    out[mask] = d / m
    return out


def compute_weighted_mix_sampling_weights(
    chunk_source_train: np.ndarray,
    dagger_fraction: float,
    dagger_recency_mult_train: np.ndarray | None = None,
) -> np.ndarray:
    """Unnormalized weights for :class:`~torch.utils.data.WeightedRandomSampler` (teleop vs DAgger mix)."""
    n_t = int(np.sum(chunk_source_train == 0))
    n_d = int(np.sum(chunk_source_train == 1))
    df = min(max(float(dagger_fraction), 1e-6), 1.0 - 1e-6)
    w_t = (1.0 - df) / max(n_t, 1)
    w_d = df / max(n_d, 1)
    w = np.where(chunk_source_train == 1, w_d, w_t).astype(np.float64)
    if dagger_recency_mult_train is not None:
        rec = _normalize_dagger_recency_train(
            chunk_source_train, dagger_recency_mult_train
        )
        w *= rec
    return w


def build_train_source_mix_sampler(
    chunk_source_train: np.ndarray,
    *,
    mode: str,
    dagger_fraction: float,
    batch_size: int,
    generator: torch.Generator,
    dagger_recency_mult_train: np.ndarray | None = None,
) -> Sampler[int] | BatchSampler | None:
    """Build a sampler or batch sampler for teleop vs DAgger mixing (train indices only).

    *chunk_source_train* is 0/1 per training sample (index into the train subset).

    Returns ``None`` for uniform shuffle (caller uses ``shuffle=True``).

    Modes:
        ``uniform`` — no structured mixing (return ``None``).
        ``weighted`` — :class:`~torch.utils.data.WeightedRandomSampler` targeting
            *dagger_fraction* probability mass on DAgger chunks (with replacement).
        ``batch_split`` — each batch contains ``round(batch_size * dagger_fraction)``
            DAgger rows and the rest teleop; pools are shuffled each epoch and cycled.
        *dagger_recency_mult_train* — optional per-train-sample multipliers (e.g. from
            :func:`chunk_dagger_recency_multipliers`); DAgger rows are mean-normalized so
            overall teleop/DAgger rates stay matched to *dagger_fraction*, while newer
            DAgger episodes get more mass within the DAgger pool.
    """
    if mode == "uniform":
        return None

    teleop = np.flatnonzero(chunk_source_train == 0).astype(np.int64, copy=False)
    dagger = np.flatnonzero(chunk_source_train == 1).astype(np.int64, copy=False)
    n_t, n_d = int(teleop.size), int(dagger.size)

    if n_d == 0 or n_t == 0:
        return None

    if not (0.0 <= dagger_fraction <= 1.0):
        raise ValueError("dagger_fraction must be in [0, 1]")

    if mode == "weighted":
        w = compute_weighted_mix_sampling_weights(
            chunk_source_train, dagger_fraction, dagger_recency_mult_train
        )
        return WeightedRandomSampler(
            weights=torch.as_tensor(w, dtype=torch.double),
            num_samples=len(chunk_source_train),
            replacement=True,
            generator=generator,
        )

    if mode == "batch_split":
        d_choice: torch.Tensor | None = None
        if dagger_recency_mult_train is not None:
            rec = _normalize_dagger_recency_train(
                chunk_source_train, dagger_recency_mult_train
            )
            d_prob = rec[dagger].astype(np.float64)
            d_prob /= d_prob.sum()
            d_choice = torch.as_tensor(d_prob, dtype=torch.double)
        return SourceStratifiedBatchSampler(
            teleop_idx=teleop,
            dagger_idx=dagger,
            batch_size=batch_size,
            dagger_fraction=dagger_fraction,
            generator=generator,
            dagger_choice_probs=d_choice,
        )

    raise ValueError(f"Unknown source mix mode: {mode!r}")


class SourceStratifiedBatchSampler(Sampler[list[int]]):
    """Yields batches with a fixed DAgger vs teleop count (indices into a train subset)."""

    def __init__(
        self,
        teleop_idx: np.ndarray,
        dagger_idx: np.ndarray,
        batch_size: int,
        dagger_fraction: float,
        generator: torch.Generator,
        dagger_choice_probs: torch.Tensor | None = None,
    ) -> None:
        self.teleop_idx = teleop_idx.astype(np.int64, copy=False)
        self.dagger_idx = dagger_idx.astype(np.int64, copy=False)
        self.batch_size = batch_size
        self.dagger_fraction = dagger_fraction
        self.generator = generator
        self.dagger_choice_probs = dagger_choice_probs
        self.n_train = int(self.teleop_idx.size + self.dagger_idx.size)

    def __len__(self) -> int:
        return (self.n_train + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        g = self.generator
        t_perm = self.teleop_idx[torch.randperm(len(self.teleop_idx), generator=g).numpy()]
        n_d = int(round(self.batch_size * self.dagger_fraction))
        n_d = max(0, min(n_d, self.batch_size))
        n_t = self.batch_size - n_d

        def cycle_from(perm: np.ndarray) -> Iterator[int]:
            if perm.size == 0:
                return
            i = 0
            while True:
                yield int(perm[i % len(perm)])
                i += 1

        it_t = cycle_from(t_perm)
        n_batches = len(self)
        dp = self.dagger_choice_probs

        if dp is not None and n_d > 0:
            for _ in range(n_batches):
                batch: list[int] = []
                pick = torch.multinomial(dp, n_d, replacement=True, generator=g)
                for li in pick.tolist():
                    batch.append(int(self.dagger_idx[int(li)]))
                for _ in range(n_t):
                    batch.append(next(it_t))
                yield batch
            return

        d_perm = self.dagger_idx[torch.randperm(len(self.dagger_idx), generator=g).numpy()]
        it_d = cycle_from(d_perm)
        for _ in range(n_batches):
            batch = []
            for _ in range(n_d):
                batch.append(next(it_d))
            for _ in range(n_t):
                batch.append(next(it_t))
            yield batch


def log_source_mix_sanity(
    train_chunk_source: np.ndarray,
    *,
    mode: str,
    dagger_fraction: float,
    batch_size: int,
    verify_seed: int,
    dagger_recency_mult_train: np.ndarray | None = None,
    dagger_latest_boost: float = 1.0,
) -> None:
    """Log a one-time check of teleop vs DAgger mixing (no per-step overhead).

    Uses NumPy with *verify_seed* only — does not advance the training sampler's RNG.
    """
    n_t = int(np.sum(train_chunk_source == 0))
    n_d = int(np.sum(train_chunk_source == 1))
    tag = "[source-mix]"

    if mode == "uniform":
        print(
            f"{tag} training will use uniform shuffle | "
            f"train chunks: {n_t} teleop, {n_d} DAgger"
        )
        if n_d > 0:
            natural = n_d / max(n_t + n_d, 1)
            print(
                f"{tag} with uniform shuffle, ~{100.0 * natural:.1f}% of random batches are DAgger "
                f"(same ratio as in the table above). To train on DAgger more often, use for example:\n"
                f"      --source-mix weighted --dagger-fraction 0.35\n"
                f"    or  --source-mix batch_split --dagger-fraction 0.35\n"
                f"  (--dagger-fraction is the *target* share of DAgger in training draws or per batch; "
                f"try 0.25–0.50.)"
            )
        return

    if n_d == 0 or n_t == 0:
        print(
            f"{tag} mode={mode!r} not active on train split "
            f"(only one source: {n_t} teleop, {n_d} DAgger)"
        )
        return

    if mode == "weighted":
        df = min(max(float(dagger_fraction), 1e-6), 1.0 - 1e-6)
        w = compute_weighted_mix_sampling_weights(
            train_chunk_source, dagger_fraction, dagger_recency_mult_train
        )
        p = w / w.sum()
        n_draw = min(4096, max(512, (n_t + n_d) * 2))
        rng = np.random.default_rng(int(verify_seed))
        pick = rng.choice(train_chunk_source.size, size=n_draw, replace=True, p=p)
        empirical = float(train_chunk_source[pick].mean())
        rec_note = ""
        if float(dagger_latest_boost) > 1.0:
            rec_note = f" | DAgger episodes: oldest→newest linear weight 1…{float(dagger_latest_boost):.2g} (mean-normalized)"
        print(
            f"{tag} mode=weighted | target P(DAgger)≈{df:.4f} | "
            f"empirical {n_draw} draws: {empirical:.4f} | "
            f"train pool: {n_t} teleop / {n_d} DAgger chunks{rec_note}"
        )
        return

    if mode == "batch_split":
        n_d_batch = max(0, min(int(round(batch_size * float(dagger_fraction))), batch_size))
        n_t_batch = batch_size - n_d_batch
        vgen = torch.Generator().manual_seed(int(verify_seed))
        sampler = build_train_source_mix_sampler(
            train_chunk_source,
            mode="batch_split",
            dagger_fraction=float(dagger_fraction),
            batch_size=batch_size,
            generator=vgen,
            dagger_recency_mult_train=dagger_recency_mult_train,
        )
        assert sampler is not None
        n_check = min(16, len(sampler))
        mismatches = 0
        for bi, batch in enumerate(iter(sampler)):
            if bi >= n_check:
                break
            b = np.asarray(batch, dtype=np.int64)
            d_cnt = int(train_chunk_source[b].sum())
            if d_cnt != n_d_batch:
                mismatches += 1
        ok = "OK" if mismatches == 0 else f"{mismatches}/{n_check} batches off"
        rec_note = ""
        if float(dagger_latest_boost) > 1.0:
            rec_note = f" | DAgger rows sampled with newest-episode bias (boost={float(dagger_latest_boost):.2g})"
        print(
            f"{tag} mode=batch_split | per batch: {n_d_batch} DAgger, {n_t_batch} teleop | "
            f"first {n_check} batches: {ok} | pool: {n_t} teleop / {n_d} DAgger chunks{rec_note}"
        )
        return

    print(f"{tag} mode={mode!r} (no extra sanity check)")


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    """Return flat indices where a full action chunk of length ``chunk_size`` fits.

    For each episode [start, end) we keep indices start … (end - chunk_size).
    """
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):
        last_start = end - chunk_size
        if last_start < start:
            continue
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


class SO100ChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs with a sliding window of size H.

    Each sample consists of:
        state:        (state_dim,)             - state at timestep t
        action_chunk: (chunk_size, action_dim) - actions [t, t+1, …, t+H-1]
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.indices = build_valid_indices(episode_ends, chunk_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        state = self.states[t]
        action_chunk = self.actions[t : t + self.chunk_size]

        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        state_t = torch.from_numpy(state).float()
        action_t = torch.from_numpy(action_chunk).float()

        return state_t, action_t
