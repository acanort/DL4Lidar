"""Download and prepare the DL4Lidar dataset from Zenodo."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import unquote, urlparse


DEFAULT_DOI = "10.5281/zenodo.20156234"
DEFAULT_DOWNLOAD_URL = (
    "https://zenodo.org/records/20156234/files/"
    "DL4Lidar.rar?download=1&preview=1"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SUFFIXES = {".rar", ".zip"}
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024


class DatasetSetupError(RuntimeError):
    """Raised when the dataset cannot be downloaded or prepared safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the DL4Lidar archive from Zenodo and create data/. "
            "Use --archive to prepare an already-downloaded copy."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--archive",
        type=Path,
        help="Local .rar or .zip archive to use instead of downloading.",
    )
    source.add_argument(
        "--url",
        default=DEFAULT_DOWNLOAD_URL,
        help="Dataset archive URL (defaults to the official Zenodo file).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory in which data/ will be created.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing data/ directory after extraction succeeds.",
    )
    return parser.parse_args()


def download_archive(url: str, destination: Path) -> Path:
    filename = Path(unquote(urlparse(url).path)).name
    if Path(filename).suffix.lower() not in ARCHIVE_SUFFIXES:
        raise DatasetSetupError("The download URL must point to a .rar or .zip file.")

    archive_path = destination / filename
    request = urllib.request.Request(url, headers={"User-Agent": "DL4Lidar/1.0"})
    print(f"Downloading {filename} from Zenodo ({DEFAULT_DOI})...")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with archive_path.open("wb") as output:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = 100 * downloaded / total
                        print(
                            f"\r  {downloaded / 1024**2:,.0f} / "
                            f"{total / 1024**2:,.0f} MiB ({percent:.1f}%)",
                            end="",
                            flush=True,
                        )
        if total:
            print()
    except urllib.error.HTTPError as error:
        archive_path.unlink(missing_ok=True)
        if error.code == 404:
            raise DatasetSetupError(
                "The Zenodo file is not public yet. Use --archive with a local copy."
            ) from error
        raise DatasetSetupError(
            f"Zenodo returned HTTP {error.code} while downloading the dataset."
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        archive_path.unlink(missing_ok=True)
        raise DatasetSetupError(f"Dataset download failed: {error}") from error
    return archive_path


def normalized_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and ":" in path.parts[0])
    ):
        raise DatasetSetupError(f"Unsafe archive member path: {name!r}")
    return path


def validate_member_names(names: Iterable[str]) -> None:
    for name in names:
        normalized_member_path(name)


def extract_zip(archive: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as handle:
            validate_member_names(info.filename for info in handle.infolist())
            handle.extractall(destination)
    except (zipfile.BadZipFile, OSError) as error:
        raise DatasetSetupError(f"Could not extract ZIP archive: {error}") from error


def rar_tool() -> tuple[str, Path]:
    candidates: list[tuple[str, Path | None]] = [
        ("unrar", Path(found) if (found := shutil.which("unrar")) else None),
        ("7z", Path(found) if (found := shutil.which("7z")) else None),
        ("7z", Path(found) if (found := shutil.which("7zz")) else None),
    ]
    if os.name == "nt":
        candidates.extend(
            [
                ("unrar", Path(r"C:\Program Files\WinRAR\UnRAR.exe")),
                ("winrar", Path(r"C:\Program Files\WinRAR\WinRAR.exe")),
                ("7z", Path(r"C:\Program Files\7-Zip\7z.exe")),
            ]
        )
    for kind, path in candidates:
        if path and path.is_file():
            return kind, path
    raise DatasetSetupError(
        "RAR extraction requires WinRAR/UnRAR or 7-Zip. Install one of them and "
        "run this command again."
    )


def rar_member_names(kind: str, executable: Path, archive: Path) -> list[str]:
    if kind in {"unrar", "winrar"}:
        command = [str(executable), "lb", str(archive)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise DatasetSetupError(
                f"Could not list RAR archive contents: {result.stderr.strip()}"
            )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    command = [str(executable), "l", "-slt", str(archive)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise DatasetSetupError(
            f"Could not list RAR archive contents: {result.stderr.strip()}"
        )
    names = [
        line.partition(" = ")[2].strip()
        for line in result.stdout.splitlines()
        if line.startswith("Path = ")
    ]
    archive_names = {str(archive), archive.name}
    return [name for name in names if name not in archive_names]


def extract_rar(archive: Path, destination: Path) -> None:
    kind, executable = rar_tool()
    validate_member_names(rar_member_names(kind, executable, archive))
    if kind == "unrar":
        command = [str(executable), "x", "-o+", "-idq", str(archive), str(destination)]
    elif kind == "winrar":
        command = [str(executable), "x", "-o+", "-inul", str(archive), str(destination)]
    else:
        command = [str(executable), "x", "-y", "-aoa", f"-o{destination}", str(archive)]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise DatasetSetupError(
            f"RAR extraction failed with exit code {result.returncode}."
        )


def extract_archive(archive: Path, destination: Path) -> None:
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        extract_zip(archive, destination)
    elif suffix == ".rar":
        extract_rar(archive, destination)
    else:
        raise DatasetSetupError(
            f"Unsupported archive type {archive.suffix!r}; use .rar or .zip."
        )


def locate_data_directory(extraction_root: Path) -> Path:
    direct = extraction_root / "data"
    if direct.is_dir():
        return direct
    nested = [path for path in extraction_root.glob("*/data") if path.is_dir()]
    if len(nested) == 1:
        return nested[0]
    if (extraction_root / "alhambra").is_dir() and (
        extraction_root / "mulhacen"
    ).is_dir():
        return extraction_root
    raise DatasetSetupError(
        "The archive does not contain the expected data/alhambra and "
        "data/mulhacen structure."
    )


def validate_dataset(data_dir: Path) -> tuple[int, int, int]:
    for instrument in ("alhambra", "mulhacen"):
        instrument_dir = data_dir / instrument
        if not (instrument_dir / "annotations").is_dir():
            raise DatasetSetupError(f"Missing {instrument}/annotations in the archive.")
        if not (instrument_dir / "images").is_dir():
            raise DatasetSetupError(f"Missing {instrument}/images in the archive.")
    files = [path for path in data_dir.rglob("*") if path.is_file()]
    json_files = [path for path in files if path.suffix.lower() == ".json"]
    image_files = [
        path
        for path in files
        if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    ]
    if not json_files or not image_files:
        raise DatasetSetupError("The extracted dataset has no annotations or images.")
    return len(files), len(json_files), len(image_files)


def install_dataset(source: Path, target: Path, replace: bool) -> None:
    if target.exists():
        if not replace:
            raise DatasetSetupError(
                f"{target} already exists. Remove it or rerun with --replace."
            )
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for source_path in source.rglob("*"):
        destination = target / source_path.relative_to(source)
        if source_path.is_dir():
            destination.mkdir(exist_ok=True)
        elif source_path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)


def run(args: argparse.Namespace) -> Path:
    project_root = args.project_root.resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    target = project_root / "data"
    if target.exists() and not args.replace:
        raise DatasetSetupError(
            f"{target} already exists. Remove it or rerun with --replace."
        )

    with tempfile.TemporaryDirectory(prefix=".dl4lidar-", dir=project_root) as temp:
        temp_dir = Path(temp)
        if args.archive:
            archive = args.archive.expanduser().resolve()
            if not archive.is_file():
                raise DatasetSetupError(f"Local archive does not exist: {archive}")
        else:
            archive = download_archive(args.url, temp_dir)

        extraction_root = temp_dir / "extracted"
        extraction_root.mkdir()
        print(f"Extracting {archive.name}...")
        extract_archive(archive, extraction_root)
        extracted_data = locate_data_directory(extraction_root)
        file_count, json_count, image_count = validate_dataset(extracted_data)
        install_dataset(extracted_data, target, args.replace)

    print(f"Dataset ready: {target}")
    print(
        f"Validated {file_count:,} files "
        f"({json_count:,} JSON annotations, {image_count:,} images)."
    )
    return target


def main() -> None:
    try:
        run(parse_args())
    except DatasetSetupError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
