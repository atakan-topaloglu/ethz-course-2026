"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily


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
    """Predicts action chunks using a Mixture Density Network (MDN) and NLL.

    Outputs parameters for a Gaussian Mixture Model (weights, means, and variances) 
    to handle multi-modal action distributions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.02,
        num_mixtures: int = 2,  # Number of Gaussian components (K)
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.num_mixtures = num_mixtures
        self.D = chunk_size * action_dim  # Total dimensions per component
        
        layers = []
        input_dim = state_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim
            
        # Output dim = K weights + K*D means + K*D log_sigmas
        out_dim = self.num_mixtures * (1 + 2 * self.D)
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mixture weights, means, and sigmas."""
        B = state.shape[0]
        out = self.net(state)
        
        # 1. Mixture probabilities (logits)
        pi_logits = out[:, :self.num_mixtures]
        
        # 2. Means (mu)
        start_mu = self.num_mixtures
        end_mu = self.num_mixtures + (self.num_mixtures * self.D)
        mu = out[:, start_mu:end_mu].view(B, self.num_mixtures, self.chunk_size, self.action_dim)
        
        # 3. Standard Deviations (sigma)
        log_sigma = out[:, end_mu:]
        # Clamp log_sigma for numerical stability (prevents NaNs from 0 variance or explosions)
        sigma = torch.exp(torch.clamp(log_sigma, min=-7.0, max=2.0))
        sigma = sigma.view(B, self.num_mixtures, self.chunk_size, self.action_dim)
        
        return pi_logits, mu, sigma

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:            
        pi_logits, mu, sigma = self.forward(state)
        
        # Construct the GMM distribution
        mix = Categorical(logits=pi_logits)
        # Event shape is 2D: (chunk_size, action_dim), so we use Independent(*, 2)
        comp = Independent(Normal(mu, sigma), 2)
        gmm = MixtureSameFamily(mix, comp)
        
        # Compute Negative Log-Likelihood (NLL)
        log_probs = gmm.log_prob(action_chunk)
        return -log_probs.mean()

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministically select the mean of the most probable Gaussian."""
        pi_logits, mu, _ = self.forward(state)
        
        # Find the index of the most likely component for each item in the batch
        best_idx = torch.argmax(pi_logits, dim=-1)  # Shape: (B,)
        
        # Gather the means of the most likely components
        B = state.shape[0]
        best_mu = mu[torch.arange(B), best_idx]  # Shape: (B, chunk_size, action_dim)
        
        return best_mu


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
        dropout: float = 0.02,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        input_dim = state_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            # layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
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
