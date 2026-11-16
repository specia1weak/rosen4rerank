import os
from pathlib import Path


class DatasetBase:
    """Base class for adapters backed by an external source-data root.

    The root may contain original files or lossless, experiment-agnostic canonical
    forms such as typed Parquet. It is read-only during experiments. Schemas,
    filtered splits, caches, and other experiment-derived assets belong under the
    repository ``workspace`` instead.
    """

    SYSTEM_DATA_DIR = Path(
        os.path.expandvars(
            os.environ.get(
                "ROSEN_DATASOURCE_ROOT",
                os.environ.get("ROSEN_DATA_DIR", "/Users/lxogri/Developer/pyprojects/time-budget/datasource"),
            )
        )
    ).expanduser()

    def __init__(self):
        pass
