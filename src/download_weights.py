"""Download pretrained DL4Lidar model assets into the expected layout."""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models" / "alhambra"
ZENODO_RECORD = "https://zenodo.org/records/21379786"
ZENODO_FILES = {
    "aerosol": {
        "config": f"{ZENODO_RECORD}/files/aerosol_config.yaml?download=1",
        "weights": f"{ZENODO_RECORD}/files/aerosol_weights.pth?download=1",
    },
    "cloud": {
        "config": f"{ZENODO_RECORD}/files/cloud_config.yaml?download=1",
        "weights": f"{ZENODO_RECORD}/files/cloud_weights.pth?download=1",
    },
}


def is_url(value: str) -> bool:
    return urlparse(value).scheme.lower() in {"http", "https"}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
    print(f"Downloading {destination.name} from {url}...")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> int:
    downloads = (
        (MODEL_ROOT / "aerosol" / "aerosol_config.yaml", ZENODO_FILES["aerosol"]["config"]),
        (MODEL_ROOT / "aerosol" / "aerosol_weights.pth", ZENODO_FILES["aerosol"]["weights"]),
        (MODEL_ROOT / "cloud" / "cloud_config.yaml", ZENODO_FILES["cloud"]["config"]),
        (MODEL_ROOT / "cloud" / "cloud_weights.pth", ZENODO_FILES["cloud"]["weights"]),
    )

    try:
        for destination, url in downloads:
            if not is_url(url):
                raise ValueError(f"Not a valid URL: {url}")
            ensure_parent(destination)
            download_file(url, destination)
            print(f"Saved {destination}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
