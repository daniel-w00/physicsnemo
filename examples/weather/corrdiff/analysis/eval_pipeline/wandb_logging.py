"""Thin, optional wandb logging helpers.

A :class:`WandbLogger` is constructed only when ``--wandb`` is passed; everywhere else
callers hold ``None`` and guard with ``if wb is not None``. Disk PNG/CSV outputs are
produced regardless of whether wandb is enabled.
"""

from __future__ import annotations

import numbers


class WandbLogger:
    def __init__(self, project: str, group: str | None = None, name: str | None = None,
                 job_type: str = "evaluation", config: dict | None = None):
        import wandb  # imported lazily so the pipeline runs without wandb installed
        self.wandb = wandb
        self.run = wandb.init(
            project=project, group=group, name=name, job_type=job_type,
            config=config or {},
        )

    def log_metrics(self, flat: dict):
        """Log only scalar metrics (skip list-valued diagnostics like rank_histogram)."""
        scalars = {k: v for k, v in flat.items() if isinstance(v, numbers.Number)}
        if scalars:
            self.run.log(scalars)

    def log_figure(self, name: str, fig):
        self.run.log({name: self.wandb.Image(fig)})

    def log_table(self, df, name: str = "metrics"):
        self.run.log({name: self.wandb.Table(dataframe=df)})

    def finish(self):
        self.run.finish()
