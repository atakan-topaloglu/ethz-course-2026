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
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

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
        for _ in range(n_layers):
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
        return self(state)


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        input_dim = state_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

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
        return self(state)

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        B = state.shape[0]
        flat_actions = self.net(state)
        return flat_actions.view(B, self.chunk_size, self.action_dim)


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
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
