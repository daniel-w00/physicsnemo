"""Pipeline-level error-field accumulator (spatial maps + temporal cycles).

Complements the vendored core accumulators (which reduce over all pixels and
timesteps) with two error breakdowns the comparison plots need:

* per-pixel time-mean bias / RMSE maps — where on the grid a model is wrong;
* domain-mean error bucketed by calendar month and hour of day — when it is wrong.

One ``update`` per (channel, timestep) computes the ensemble-mean error field once
and feeds both breakdowns, so it adds a single (H, W) pass on top of the core
accumulators in the streaming loop.
"""

from __future__ import annotations

import numpy as np
import torch


class FieldErrorAccumulator:
    """Accumulates ensemble-mean error per pixel and per month/hour bucket."""

    def __init__(self, img_shape: tuple[int, int]):
        H, W = img_shape
        self.err_sum = np.zeros((H, W), dtype=np.float64)
        self.se_sum = np.zeros((H, W), dtype=np.float64)
        self.n = 0

        self.month_se = np.zeros(12, dtype=np.float64)
        self.month_err = np.zeros(12, dtype=np.float64)
        self.month_n = np.zeros(12, dtype=np.int64)
        self.hour_se = np.zeros(24, dtype=np.float64)
        self.hour_err = np.zeros(24, dtype=np.float64)
        self.hour_n = np.zeros(24, dtype=np.int64)

    @torch.no_grad()
    def update(self, pred_ens: torch.Tensor, target: torch.Tensor, month: int, hour: int):
        """Accumulate one timestep.

        Args:
            pred_ens: (N_ens, 1, H, W) predictions in physical units.
            target:   (1, H, W) ground truth.
            month:    Calendar month 1–12.
            hour:     Hour of day 0–23.
        """
        err = (pred_ens.mean(dim=0)[0] - target[0]).numpy().astype(np.float64)
        se = err ** 2
        self.err_sum += err
        self.se_sum += se
        self.n += 1

        mi, hi = month - 1, hour
        self.month_err[mi] += err.mean()
        self.month_se[mi] += se.mean()
        self.month_n[mi] += 1
        self.hour_err[hi] += err.mean()
        self.hour_se[hi] += se.mean()
        self.hour_n[hi] += 1

    # -- spatial maps -----------------------------------------------------------

    def bias_map(self) -> np.ndarray:
        """(H, W) time-mean signed error."""
        return self.err_sum / max(self.n, 1)

    def rmse_map(self) -> np.ndarray:
        """(H, W) time-mean RMSE."""
        return np.sqrt(self.se_sum / max(self.n, 1))

    # -- temporal cycles --------------------------------------------------------

    def cycle(self, which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bucketed cycle: ``(positions, rmse, bias, counts)`` for ``month`` or ``hour``.

        Buckets with no samples have NaN rmse/bias so plots leave gaps.
        """
        if which == "month":
            se, err, n = self.month_se, self.month_err, self.month_n
            pos = np.arange(1, 13)
        elif which == "hour":
            se, err, n = self.hour_se, self.hour_err, self.hour_n
            pos = np.arange(24)
        else:
            raise ValueError(f"which must be 'month' or 'hour', got {which!r}")
        valid = n > 0
        rmse = np.full(len(pos), np.nan)
        bias = np.full(len(pos), np.nan)
        rmse[valid] = np.sqrt(se[valid] / n[valid])
        bias[valid] = err[valid] / n[valid]
        return pos, rmse, bias, n
