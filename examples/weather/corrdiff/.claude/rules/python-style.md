# Python Code Style Rules

## Script Structure

### Always Use Main Guard
Every executable Python script MUST use the main guard:

```python
if __name__ == "__main__":
    main()
```

This ensures code is only executed when run directly, not when imported.

### Script Template
```python
"""
Module/script description.

Longer explanation if needed.
"""
import argparse
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import xarray as xr
import torch

# Local imports (if any)
from utils import helper_function


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="What this script does"
    )
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Script logic here
    print(f"Processing {args.input}...")

    print("✓ Done")


if __name__ == "__main__":
    main()
```

## Naming Conventions

### Coordinate Variables
Be explicit with geographic and array dimension names:

**Good**:
```python
lat, lon = coordinates
height, width, channels = array.shape
H, W, C = img.shape
```

**Bad**:
```python
x, y = coordinates  # Ambiguous for geographic data
h, w, c = img.shape  # Too terse
```

### Descriptive Names
- Use `lat`, `lon`, `elev` for geographic coordinates and elevation
- Use `H`, `W`, `C` for array dimensions (height, width, channels)
- Use `ds` for xarray Datasets, `da` for DataArrays
- Use `img` for image arrays, `mask` for boolean arrays

## Documentation

### Docstrings
Use docstrings for functions and classes:



Use symbols for status:
- `✓` for success
- `⚠️` for warnings
- `❌` for errors
- `🚀` for starting operations
