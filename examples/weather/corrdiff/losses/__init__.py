"""Local loss variants for corrdiff (embedding-aware wrappers)."""

from .emb_branch_losses import EmbRegressionLoss, EmbResidualLoss

__all__ = ["EmbRegressionLoss", "EmbResidualLoss"]
