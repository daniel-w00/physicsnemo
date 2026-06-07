"""Core metric and plot engine for the evaluation pipeline.

The reference metric/plot implementations from the CorrDiff project (see the SPDX
headers in :mod:`~.metrics` and :mod:`~.plots`). These modules are vendored unchanged
and treated as the source of truth — the rest of the pipeline only orchestrates them.
Do not edit the algorithms here; adapt behaviour in the surrounding pipeline modules.
"""