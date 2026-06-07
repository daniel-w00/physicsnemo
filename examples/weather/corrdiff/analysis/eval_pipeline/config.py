"""Optional YAML config loading for reproducible evaluation runs.

CLI flags always win: :func:`merge` fills a value from the config file only when the
corresponding CLI argument was left at its default (``None``). Plain ``yaml.safe_load``
— deliberately not Hydra.
"""

from __future__ import annotations

import os


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(data).__name__}")
    return data


def merge(cli: dict, cfg: dict) -> dict:
    """Return cli with any None values filled from cfg (CLI overrides config)."""
    out = dict(cfg)
    for k, v in cli.items():
        if v is not None:
            out[k] = v
    return out


def resolve_event_times(event_times, times_yaml: str | None) -> list[str]:
    """Combine inline event timestamps with an optional times-YAML file.

    ``times_yaml`` may point at an existing curated file such as
    ``conf/test_times_2021.yaml`` (a mapping ``{times: [iso, ...]}``).
    """
    out = list(event_times or [])
    if times_yaml:
        if not os.path.exists(times_yaml):
            raise FileNotFoundError(f"times file not found: {times_yaml}")
        data = load_yaml(times_yaml)
        out.extend(data.get("times", []))
    return out
