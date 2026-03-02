import os
from pathlib import Path

# Get the root of the project (2 levels up from current file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Common data paths
DATA_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings"
DATA_IMAGES = PROJECT_ROOT / "data" / "images"
DATA_DATASETS = PROJECT_ROOT / "data"