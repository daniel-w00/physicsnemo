"""Offline evaluation pipeline for CorrDiff generation outputs.

Reads generated NetCDF files (``truth`` / ``prediction`` groups) and scores them with
the core metric/plot engine in :mod:`analysis.eval_pipeline.core`. Run as a module from
the corrdiff repo root::

    python -m analysis.eval_pipeline.run single  --pred FILE.nc --name NAME
    python -m analysis.eval_pipeline.run compare --pred-a A.nc --pred-b B.nc ...
"""
