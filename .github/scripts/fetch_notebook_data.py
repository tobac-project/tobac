"""Fetch the example data used by the notebooks and place it where they expect it.

Files are downloaded into a cache directory (restored/saved by the notebook CI
workflow) and verified against the pinned hashes below. pooch re-checks the
hash of already-cached files and re-downloads any that do not match.

Usage:
    python .github/scripts/fetch_notebook_data.py CACHE_DIR
"""

import shutil
import sys
from pathlib import Path

import pooch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"

# fname: (url, known_hash)
# Changing this registry changes the CI cache key, so caches are invalidated
# automatically when files are added or updated.
REGISTRY = {
    "tobac_example_data-v1.0.1.zip": (
        "https://zenodo.org/records/3195910/files/climate-processes/tobac_example_data-v1.0.1.zip",
        "sha256:609f00173687be856c2d6f5f1ee907b3db36e1c55642fcc7446723d2d84877b7",
    ),
    "Example_input_vorticity_model.nc": (
        "https://zenodo.org/records/6459542/files/Example_input_vorticity_model.nc",
        "sha256:23b917b8c79dc83f69f4c3c438a26420df85fcae640677a0762c83606cc4ddda",
    ),
}


def fetch(cache_dir: Path) -> dict[str, Path]:
    return {
        fname: Path(
            pooch.retrieve(url, known_hash=known_hash, fname=fname, path=cache_dir)
        )
        for fname, (url, known_hash) in REGISTRY.items()
    }


def main() -> None:
    cache_dir = Path(sys.argv[1]).expanduser()
    files = fetch(cache_dir)

    # Notebooks look for data/<file> anywhere under examples/ and only
    # download it themselves when it is missing.
    shutil.unpack_archive(files["tobac_example_data-v1.0.1.zip"], EXAMPLES_DIR)
    data_dir = EXAMPLES_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    shutil.copy2(files["Example_input_vorticity_model.nc"], data_dir)


if __name__ == "__main__":
    main()
