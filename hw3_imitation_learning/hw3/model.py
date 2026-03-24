"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, chunk_size * action_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        B = state.shape[0]
        flat_actions = self.net(state)
        return flat_actions.view(B, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_actions = self(state)
        return nn.functional.mse_loss(pred_actions, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(state)


class FeatureStem(nn.Module):
    """Linear map from input features to the trunk hidden width."""

    def __init__(self, in_features: int, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ResidualWidthBlock(nn.Module):
    """Width-preserving MLP block used as a residual delta."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualFeatureStack(nn.Module):
    """Stack of width blocks with optional residual connections."""

    def __init__(
        self,
        width: int,
        n_blocks: int,
        dropout: float,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        self.blocks = nn.ModuleList(
            ResidualWidthBlock(width, dropout) for _ in range(n_blocks)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            delta = block(h)
            h = h + delta if self.use_skip else delta
        return h


class ActionChunkHead(nn.Module):
    """Maps trunk features to an action chunk (B, chunk_size, action_dim)."""

    def __init__(self, width: int, chunk_size: int, action_dim: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.linear = nn.Linear(width, chunk_size * action_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        flat = self.linear(h)
        return flat.view(-1, self.chunk_size, self.action_dim)


class ResidualChunkBackbone(nn.Module):
    """Stem → residual width stack → chunk head (multitask policy trunk)."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        chunk_size: int,
        width: int,
        n_blocks: int,
        dropout: float = 0.05,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.stem = FeatureStem(in_features, width)
        self.trunk = ResidualFeatureStack(
            width, n_blocks, dropout, use_skip=use_skip
        )
        self.head = ActionChunkHead(width, chunk_size, action_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.stem(features)
        h = self.trunk(h)
        return self.head(h)


class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 4,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        if state_mean is None:
            mean_buf = torch.zeros(state_dim, dtype=torch.float32)
        else:
            mean_buf = torch.as_tensor(state_mean, dtype=torch.float32)
        if state_std is None:
            std_buf = torch.ones(state_dim, dtype=torch.float32)
        else:
            std_buf = torch.as_tensor(state_std, dtype=torch.float32)
        self.register_buffer("state_mean", mean_buf)
        self.register_buffer("state_std", std_buf)
        self.backbone = ResidualChunkBackbone(
            in_features=7, # grip (1) + vectocube (3) + vectogoal (3)
            action_dim=action_dim,
            chunk_size=chunk_size,
            width=hidden_dim,
            n_blocks=n_layers,
            dropout=0.1,
            use_skip=True,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        batch = state.shape[0]
        grip = state[:, 3:4]

        physical = state * self.state_std + self.state_mean
        goal_scores = physical[:, 13:16]
        goal_sel = torch.argmax(goal_scores, dim=1)
        goal_one_hot = nn.functional.one_hot(goal_sel, num_classes=3).float()
        ee = physical[:, :3]
        goal_xyz = physical[:, 16:19]
        stacked = physical[:, 4:13].view(batch, 3, 3)
        weights = goal_one_hot.unsqueeze(-1)
        focused_cube = (stacked * weights).sum(dim=1)
        vec_to_cube = focused_cube - ee
        vec_to_goal = goal_xyz - ee
        features = torch.cat([grip, vec_to_cube, vec_to_goal], dim=1)
        return self.backbone(features)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 32,
    d_model: int | None = None,
    depth: int | None = None,
    hidden_dim: int | None = None,
    n_layers: int | None = None,
    state_mean: torch.Tensor | None = None,
    state_std: torch.Tensor | None = None,
) -> BasePolicy:
    hdim = hidden_dim if hidden_dim is not None else (d_model or 256)
    nlay = n_layers if n_layers is not None else (depth or 3)

    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            hidden_dim=hdim,
            n_layers=nlay,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            hidden_dim=hdim,
            n_layers=nlay,
            state_mean=state_mean,
            state_std=state_std,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
