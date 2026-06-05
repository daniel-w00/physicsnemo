"""
Sample two disjoint lists of validation and test timestamps from the CWA zarr dataset.

Picks 2 * N distinct timestamps (default N=256) from year 2021 that are marked valid
in the zarr store (`cwb_valid` AND all `era5_valid` channels), splits them randomly into
val and test, and writes them as a Hydra-friendly YAML file with ISO 8601 timestamps.

With --with-events, the curated TYPHOON_TIMES list below is force-included in test_times.
Those timestamps are removed from the random pool, so the random portion of test shrinks
accordingly and val stays disjoint. Edit TYPHOON_TIMES in this file to adjust the events.

Usage (inside apptainer on the HPC):
    apptainer exec ~/apptainer/corrdiff_10_02.sif \
        python3 make_eval_timestamps.py \
            --output-val val_times_2021.yaml \
            --output-test test_times_2021.yaml

With curated typhoon events forced into the test set:
    python3 make_eval_timestamps.py \
        --output-val val_times_2021.yaml \
        --output-test test_times_2021.yaml \
        --with-events
"""
import argparse
import sys
from pathlib import Path

import cftime
import numpy as np
import yaml
import zarr


DEFAULT_DATA_PATH = (
    "/data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset/cwa_dataset.zarr"
)

# Curated 2021 Taiwan typhoon timestamps, force-included in test_times when --with-events.
# Tweak this list freely — invalid/missing timestamps are warned and skipped at runtime.
TYPHOON_TIMES = [
    # In-Fa — affected Taiwan ~July 21-25
    "2021-07-22T12:00:00",
    "2021-07-23T00:00:00",
    "2021-07-23T18:00:00",
    # Lupit — heavy rain ~August 4-6
    "2021-08-05T00:00:00",
    "2021-08-05T12:00:00",
    "2021-08-06T00:00:00",
    # Chanthu — closest approach ~September 12
    "2021-09-11T12:00:00",
    "2021-09-12T00:00:00",
    "2021-09-12T12:00:00",
    "2021-09-13T00:00:00",
]


def load_valid_times_2021(data_path: str) -> np.ndarray:
    """Open the zarr store and return an array of valid timestamps in 2021."""
    group = zarr.open_consolidated(data_path, mode="r")

    cwb_valid = np.asarray(group["cwb_valid"][:]) != 0
    era5_valid = np.asarray(group["era5_valid"][:]) != 0
    era5_all = np.all(era5_valid, axis=-1)
    valid_mask = cwb_valid & era5_all

    times = cftime.num2date(group["time"][:], units=group["time"].attrs["units"])
    times = np.asarray(times)

    valid_times = times[valid_mask]
    year_mask = np.array([t.year == 2021 for t in valid_times])
    return valid_times[year_mask]


def to_iso(t) -> str:
    """Format a cftime/datetime object as ISO 8601 (no timezone)."""
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}T{t.hour:02d}:{t.minute:02d}:{t.second:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to CWA zarr dataset. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--output-val",
        type=str,
        required=True,
        help="Output YAML file for validation timestamps (flat list of ISO strings).",
    )
    parser.add_argument(
        "--output-test",
        type=str,
        required=True,
        help="Output YAML file for test timestamps (flat list of ISO strings, "
             "with typhoons appended at the end if --with-events).",
    )
    parser.add_argument(
        "--n-per-split",
        type=int,
        default=256,
        help="Number of timestamps per split (val and test each get N). Default: 256",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--with-events",
        action="store_true",
        help="Force-include the curated TYPHOON_TIMES into test_times "
             "(reduces the random portion of test accordingly).",
    )
    args = parser.parse_args()

    print(f"Opening zarr dataset: {args.data_path}")
    valid_2021 = load_valid_times_2021(args.data_path)
    n_available = len(valid_2021)
    n_needed = 2 * args.n_per_split
    print(f"Found {n_available} valid timestamps in 2021.")

    if n_available < n_needed:
        print(
            f"Error: only {n_available} valid 2021 timestamps available, "
            f"need {n_needed} for two disjoint splits of {args.n_per_split}.",
            file=sys.stderr,
        )
        sys.exit(1)

    event_times = []
    if args.with_events:
        available_iso = {to_iso(t) for t in valid_2021}
        missing = [e for e in TYPHOON_TIMES if e not in available_iso]
        if missing:
            print(
                f"Warning: {len(missing)} TYPHOON_TIMES not in valid 2021 set, skipping: {missing}",
                file=sys.stderr,
            )
        event_times = [e for e in TYPHOON_TIMES if e in available_iso]
        if len(event_times) > args.n_per_split:
            print(
                f"Error: {len(event_times)} curated events exceed --n-per-split {args.n_per_split}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Including {len(event_times)} curated typhoon timestamp(s) in test set")

    event_set = set(event_times)
    pool_mask = np.array([to_iso(t) not in event_set for t in valid_2021])
    pool = valid_2021[pool_mask]
    n_random = n_needed - len(event_times)
    if len(pool) < n_random:
        print(
            f"Error: only {len(pool)} non-event timestamps available, need {n_random}.",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    picked_idx = rng.choice(len(pool), size=n_random, replace=False)
    picked = pool[picked_idx]

    val_times = sorted(to_iso(t) for t in picked[: args.n_per_split])
    test_random = picked[args.n_per_split :]
    test_times = sorted(to_iso(t) for t in test_random) + sorted(event_times)

    overlap = set(val_times) & set(test_times)
    if overlap:
        print(f"Error: val/test overlap detected: {overlap}", file=sys.stderr)
        sys.exit(1)

    base_header = (
        f"# Generated by make_eval_timestamps.py\n"
        f"# Source: {args.data_path}\n"
        f"# Year: 2021, seed: {args.seed}, n_per_split: {args.n_per_split}\n"
        f"# Available valid 2021 timestamps: {n_available}\n"
    )
    val_header = base_header + f"# Split: validation ({len(val_times)} entries)\n\n"
    test_header = (
        base_header
        + f"# Split: test ({len(test_times)} entries, "
        + f"{len(event_times)} forced typhoon events appended at the end)\n\n"
    )

    def _write_times(path_str: str, times: list, header: str):
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(header)
            yaml.safe_dump({"times": times}, f, default_flow_style=False, sort_keys=False)
        return path

    val_path = _write_times(args.output_val, val_times, val_header)
    test_path = _write_times(args.output_test, test_times, test_header)
    print(f"Wrote {len(val_times)} val timestamps to {val_path}")
    print(f"Wrote {len(test_times)} test timestamps to {test_path}")


if __name__ == "__main__":
    main()
