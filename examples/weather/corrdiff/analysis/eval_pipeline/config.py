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


def _default_name(pred: str) -> str:
    return os.path.splitext(os.path.basename(pred))[0]


def resolve_models(cfg: dict) -> list[dict]:
    """Build the ordered list of models to compare from CLI flags or config.

    Resolution order (first match wins):

    1. Repeatable CLI flags ``--pred/--name/--kind`` (parsed into ``preds/names/kinds``).
    2. A config ``models:`` list of ``{pred, name, kind}`` mappings.
    3. Legacy suffixed keys ``pred_a/name_a/kind_a``, ``pred_b/...`` (up to ``h``).

    Each returned dict has ``pred``, ``name`` (defaulted from the filename), and ``kind``.
    """
    preds = cfg.get("preds")
    if preds:
        names = cfg.get("names") or []
        kinds = cfg.get("kinds") or []
        return [
            {
                "pred": p,
                "name": names[i] if i < len(names) else _default_name(p),
                "kind": kinds[i] if i < len(kinds) else "auto",
            }
            for i, p in enumerate(preds)
        ]

    if cfg.get("models"):
        out = []
        for m in cfg["models"]:
            pred = m["pred"]
            out.append({
                "pred": pred,
                "name": m.get("name") or _default_name(pred),
                "kind": m.get("kind") or "auto",
            })
        return out

    out = []
    for sfx in "abcdefgh":
        pred = cfg.get(f"pred_{sfx}")
        if pred:
            out.append({
                "pred": pred,
                "name": cfg.get(f"name_{sfx}") or _default_name(pred),
                "kind": cfg.get(f"kind_{sfx}") or "auto",
            })
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
