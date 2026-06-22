"""Master script for Detectron2 inference on aerosol/cloud imagery.

The script loads one or two trained class-specific models, runs instance
segmentation on matching image streams, and writes machine-readable prediction
files plus optional visual outputs. It is intended as the public inference
entry point; scenario-specific inference scripts can wrap it with preset paths.

Expected model layout:
models/alhambra/{class}/{class}_config.yaml
models/alhambra/{class}/{class}_weights.pth

Example:
    python src/inference.py \
        --classes aerosol,cloud \
        --aerosol-input data/chm15k/images/1064nm/percentiles_1.0_80.5 \
        --cloud-input data/chm15k/images/1064nm/percentiles_1.0_95.5 \
        --output inference/example_run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings(
    "ignore", message=".*distutils Version classes are deprecated.*"
)
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*indexing argument.*")

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import transforms as T  # noqa: E402
from detectron2.data.detection_utils import read_image  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "models" / "alhambra"

CLASS_ORDER = ("aerosol", "cloud")
CLASS_COLORS = {
    "aerosol": (239, 68, 68),
    "cloud": (250, 204, 21),
    "overlap": (168, 85, 247),
}

INSTRUMENT = "alhambra"
CHANNEL = "1064fta"
SCALE = "3e8"

DEFAULT_INPUTS = {
    "aerosol": DATA_ROOT / INSTRUMENT / "images" / CHANNEL / SCALE,
    "cloud":   DATA_ROOT / INSTRUMENT / "images" / CHANNEL / SCALE,
}

DEFAULT_CLASSES = "cloud"
DEFAULT_OUTPUT = PROJECT_ROOT / "inference" / (SCALE + "_" + time.strftime("%Y%m%d_%H%M%S"))
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
IMAGE_ALPHA = 0.45
SCORE_THRESHOLD = 0.7
VISUAL_OUTPUT_MAX_SIZE = (1280, 720)
VISUAL_OUTPUT_JPG_QUALITY = 90
VISUAL_OUTPUT_JPG_OPTIMIZE = False


class InferenceError(RuntimeError):
    """Raised when inference inputs or model files are not usable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Alhambra inference from models/alhambra/{class} files."
    )
    parser.add_argument(
        "--classes",
        default=DEFAULT_CLASSES,
        help="Comma-separated classes to infer: aerosol, cloud, or aerosol,cloud.",
    )
    parser.add_argument(
        "--aerosol-input",
        type=Path,
        default=DEFAULT_INPUTS["aerosol"],
        help="Image file or directory for the aerosol model.",
    )
    parser.add_argument(
        "--cloud-input",
        type=Path,
        default=DEFAULT_INPUTS["cloud"],
        help="Image file or directory for the cloud model.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--aerosol-model-dir",
        type=Path,
        default=None,
        help="Defaults to models/alhambra/aerosol.",
    )
    parser.add_argument(
        "--cloud-model-dir",
        type=Path,
        default=None,
        help="Defaults to models/alhambra/cloud.",
    )
    parser.add_argument(
        "--plot-base-class",
        choices=CLASS_ORDER,
        default=None,
        help="Image stream used as the background for overlays.",
    )
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--alpha", type=float, default=IMAGE_ALPHA)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Images processed per model forward pass. "
            "Increase on GPU if memory allows."
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=("full", "quick", "predictions"),
        default="full",
        help=(
            "full saves all visuals; quick saves masks, overlays, and the "
            "comparison plot; predictions skips images."
        ),
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=VISUAL_OUTPUT_JPG_QUALITY,
        help="JPEG quality for overlay outputs.",
    )
    parser.add_argument(
        "--optimize-jpg",
        action="store_true",
        default=VISUAL_OUTPUT_JPG_OPTIMIZE,
        help="Optimize JPEG files. Smaller files, but slower writes.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device for selected models.",
    )
    return parser.parse_args()


def configure_quiet_logging() -> None:
    for logger_name in ("detectron2", "fvcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def selected_classes(classes_arg: str) -> list[str]:
    values = [value.strip().lower() for value in classes_arg.split(",")]
    classes = []
    for class_name in values:
        if not class_name:
            continue
        if class_name not in CLASS_ORDER:
            raise InferenceError(
                f"Unknown class '{class_name}'. Use aerosol, cloud, or both."
            )
        if class_name not in classes:
            classes.append(class_name)
    if not classes:
        raise InferenceError("--classes did not contain any valid class names.")
    return [class_name for class_name in CLASS_ORDER if class_name in classes]


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def print_table(title: str, rows: list[tuple[str, object]]) -> None:
    print(title)
    for key, value in rows:
        print(f"  {key}: {value}")


def chunked(items: list, chunk_size: int):
    if chunk_size < 1:
        raise InferenceError("--batch-size must be at least 1.")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def list_images(input_path: Path, recursive: bool) -> list[Path]:
    resolved = input_path.resolve()
    if resolved.is_file():
        images = [resolved] if resolved.suffix.lower() in IMAGE_EXTENSIONS else []
    elif resolved.is_dir():
        pattern = "**/*" if recursive else "*"
        images = sorted(
            path
            for path in resolved.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        raise InferenceError(f"Input path does not exist: {input_path}")

    if not images:
        raise InferenceError(f"No supported image files found in: {input_path}")
    return images


def model_dir_path(class_name: str, explicit_model_dir: Optional[Path]) -> Path:
    path = (
        explicit_model_dir
        if explicit_model_dir is not None
        else MODEL_ROOT / class_name
    )
    resolved = path.resolve()
    if not resolved.is_dir():
        raise InferenceError(f"Model directory does not exist: {path}")
    return resolved


def expected_file(model_dir: Path, class_name: str, suffix: str) -> Path:
    path = (model_dir / f"{class_name}_{suffix}").resolve()
    if not path.is_file():
        raise InferenceError(f"Missing {class_name} {suffix}: {path}")
    return path


def add_project_config_if_needed(cfg, config_file: Path) -> None:
    config_text = config_file.read_text(encoding="utf-8", errors="ignore")
    uses_pointrend = (
        "PointRend" in config_text
        or "POINT_HEAD" in config_text
        or "IMPLICIT_POINTREND" in config_text
    )
    if not uses_pointrend:
        return

    try:
        from detectron2.projects.point_rend import add_pointrend_config
    except ImportError as error:
        raise InferenceError(
            "This config uses PointRend, but detectron2.projects.point_rend "
            "could not be imported."
        ) from error

    add_pointrend_config(cfg)
    cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO = float(
        cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO
    )


def build_predictor(
    config_file: Path, weights: Path, score_threshold: float, device: str
):
    cfg = get_cfg()
    add_project_config_if_needed(cfg, config_file)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(config_file))
    cfg.set_new_allowed(False)
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = (
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    if device == "auto" and not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.freeze()
    return cfg, BatchedPredictor(cfg)


class BatchedPredictor:
    """Small DefaultPredictor equivalent that also accepts image batches."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.load_weights()
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )
        self.input_format = self.cfg.INPUT.FORMAT
        if self.input_format not in ("RGB", "BGR"):
            raise InferenceError(f"Unsupported input format: {self.input_format}")

    def __call__(self, original_image: np.ndarray) -> dict:
        return self.predict_batch([original_image])[0]

    def predict_batch(self, images: list[np.ndarray]) -> list[dict]:
        inputs = [self._prepare_image(image) for image in images]
        with torch.no_grad():
            return self.model(inputs)

    def _prepare_image(self, original_image: np.ndarray) -> dict:
        if self.input_format == "RGB":
            image = original_image[:, :, ::-1]
        else:
            image = original_image
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        return {"image": image, "height": height, "width": width}

    def load_weights(self) -> None:
        weights = Path(self.cfg.MODEL.WEIGHTS)
        checkpoint = torch.load(weights, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            return
        if isinstance(checkpoint, dict) and "model" not in checkpoint:
            try:
                self.model.load_state_dict(checkpoint, strict=False)
                return
            except RuntimeError:
                pass
        DetectionCheckpointer(self.model).load(str(weights))


def resized_for_output(image: Image.Image) -> Image.Image:
    resized = image.copy()
    try:
        resampling = Image.Resampling.LANCZOS
    except AttributeError:
        resampling = Image.LANCZOS
    resized.thumbnail(VISUAL_OUTPUT_MAX_SIZE, resampling)
    return resized


def save_jpg(
    image: Image.Image,
    path: Path,
    quality: int,
    optimize: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=quality, optimize=optimize)


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", optimize=True)


def load_model(
    class_name: str,
    explicit_model_dir: Optional[Path],
    score_threshold: float,
    device: str,
) -> dict:
    model_dir = model_dir_path(class_name, explicit_model_dir)
    config_file = expected_file(model_dir, class_name, "config.yaml")
    weights = expected_file(model_dir, class_name, "weights.pth")
    cfg, predictor = build_predictor(
        config_file, weights, score_threshold, device
    )
    if int(cfg.MODEL.ROI_HEADS.NUM_CLASSES) != 1:
        raise InferenceError(f"{class_name} model must have exactly one class.")
    return {
        "model_dir": model_dir,
        "config": config_file,
        "weights": weights,
        "cfg": cfg,
        "predictor": predictor,
    }


def input_path_for_class(args: argparse.Namespace, class_name: str) -> Path:
    return getattr(args, f"{class_name}_input")


def model_dir_arg_for_class(
    args: argparse.Namespace, class_name: str
) -> Optional[Path]:
    return getattr(args, f"{class_name}_model_dir")


def image_key(input_root: Path, image_path: Path) -> str:
    root = input_root.resolve()
    if root.is_file():
        return image_path.name
    return image_path.resolve().relative_to(root).as_posix()


def image_map(input_path: Path, recursive: bool) -> dict[str, Path]:
    images = list_images(input_path, recursive)
    mapped = {}
    for image_path in images:
        key = image_key(input_path, image_path)
        if key in mapped:
            raise InferenceError(f"Duplicate image key '{key}' in {input_path}")
        mapped[key] = image_path
    return mapped


def paired_images(
    args: argparse.Namespace, classes: list[str]
) -> list[tuple[str, dict[str, Path]]]:
    images_by_class = {
        class_name: image_map(input_path_for_class(args, class_name), args.recursive)
        for class_name in classes
    }
    reference_class = classes[0]
    reference_keys = set(images_by_class[reference_class])

    for class_name in classes[1:]:
        class_keys = set(images_by_class[class_name])
        if class_keys == reference_keys:
            continue
        missing = sorted(reference_keys - class_keys)
        extra = sorted(class_keys - reference_keys)
        details = []
        if missing:
            details.append(f"missing in {class_name}: {', '.join(missing[:8])}")
        if extra:
            details.append(f"only in {class_name}: {', '.join(extra[:8])}")
        raise InferenceError(
            "Selected inputs must contain the same image names; "
            + "; ".join(details)
        )

    keys = sorted(reference_keys)
    if args.max_images is not None:
        keys = keys[: max(args.max_images, 0)]
    if not keys:
        raise InferenceError("No images found for selected classes.")
    return [
        (
            key,
            {
                class_name: images_by_class[class_name][key]
                for class_name in classes
            },
        )
        for key in keys
    ]


def plot_base_class(args: argparse.Namespace, classes: list[str]) -> str:
    if args.plot_base_class is not None:
        if args.plot_base_class not in classes:
            raise InferenceError("--plot-base-class must be one of --classes.")
        return args.plot_base_class
    return "aerosol" if "aerosol" in classes else classes[0]


def mask_from_instances(instances, height: int, width: int) -> np.ndarray:
    if len(instances) == 0 or not instances.has("pred_masks"):
        return np.zeros((height, width), dtype=bool)
    return np.any(instances.pred_masks.numpy().astype(bool), axis=0)


def instance_records(instances, class_name: str) -> list[dict]:
    if len(instances) == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []
    masks = (
        instances.pred_masks.numpy().astype(bool)
        if instances.has("pred_masks")
        else []
    )
    records = []
    for index in range(len(instances)):
        records.append(
            {
                "class_name": class_name,
                "instance_id": index,
                "score": float(scores[index]) if len(scores) else None,
                "bbox_xyxy": [float(value) for value in boxes[index]]
                if len(boxes)
                else None,
                "mask_pixels": int(masks[index].sum()) if len(masks) else None,
            }
        )
    return records


def colored_mask(masks: dict[str, np.ndarray], classes: list[str]) -> np.ndarray:
    height, width = masks[classes[0]].shape
    output = np.zeros((height, width, 3), dtype=np.uint8)
    for class_name in classes:
        output[masks[class_name]] = CLASS_COLORS[class_name]
    if "aerosol" in classes and "cloud" in classes:
        output[masks["aerosol"] & masks["cloud"]] = CLASS_COLORS["overlap"]
    return output


def transparent_overlay(
    image_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    classes: list[str],
    alpha: float,
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32)
    color_mask = colored_mask(masks, classes)
    any_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    for class_name in classes:
        any_mask |= masks[class_name]
    overlay[any_mask] = (
        (1.0 - alpha) * overlay[any_mask] + alpha * color_mask[any_mask]
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_visuals(
    base_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    classes: list[str],
    output_dir: Path,
    image_key_value: str,
    alpha: float,
    base_class: str,
    output_mode: str,
    jpg_quality: int,
    optimize_jpg: bool,
) -> dict[str, str]:
    if output_mode == "predictions":
        return {}

    stem = Path(image_key_value).stem
    image_rgb = base_bgr[:, :, ::-1].copy()
    mask_rgb = colored_mask(masks, classes)
    translucent = transparent_overlay(image_rgb, masks, classes, alpha)
    solid = image_rgb.copy()
    any_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    for class_name in classes:
        any_mask |= masks[class_name]
    solid[any_mask] = mask_rgb[any_mask]

    saved = {}
    for class_name in classes:
        path = output_dir / "masks_png" / f"{stem}_{class_name}.png"
        save_png(Image.fromarray(masks[class_name].astype(np.uint8) * 255), path)
        saved[f"{class_name}_mask"] = str(path)

    combined_mask_path = output_dir / "masks_png" / f"{stem}_combined_rgb.png"
    save_png(Image.fromarray(mask_rgb), combined_mask_path)
    saved["combined_mask_rgb"] = str(combined_mask_path)

    translucent_path = output_dir / "overlays_translucent_jpg" / f"{stem}.jpg"
    save_jpg(
        resized_for_output(Image.fromarray(translucent)),
        translucent_path,
        jpg_quality,
        optimize_jpg,
    )
    saved["translucent_overlay"] = str(translucent_path)

    solid_path = output_dir / "overlays_opaque_jpg" / f"{stem}.jpg"
    save_jpg(
        resized_for_output(Image.fromarray(solid)),
        solid_path,
        jpg_quality,
        optimize_jpg,
    )
    saved["opaque_overlay"] = str(solid_path)

    comparison_path = output_dir / "plots" / f"{stem}_input_translucent.png"
    save_comparison_plot(image_rgb, translucent, comparison_path, base_class)
    saved["comparison_plot"] = str(comparison_path)

    if output_mode == "quick":
        return saved

    masks_path = output_dir / "plots" / f"{stem}_masks.png"
    save_masks_plot(image_rgb, mask_rgb, masks, classes, masks_path, base_class)
    saved["masks_plot"] = str(masks_path)
    return saved


def save_comparison_plot(
    image_rgb: np.ndarray,
    translucent: np.ndarray,
    path: Path,
    base_class: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        (f"{base_class} input", image_rgb),
        ("Translucent overlay", translucent),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(resized_for_output(Image.fromarray(image)))
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_masks_plot(
    image_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    classes: list[str],
    path: Path,
    base_class: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [(f"{base_class} input", image_rgb)]
    for class_name in classes:
        panels.append((f"{class_name} mask", masks[class_name].astype(np.uint8) * 255))
    panels.append(("Combined mask", mask_rgb))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 4.8))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(resized_for_output(Image.fromarray(image)), cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_predictions(output_dir: Path, predictions: list[dict]) -> None:
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)

    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "image",
            "source_image",
            "class_name",
            "instance_id",
            "score",
            "x1",
            "y1",
            "x2",
            "y2",
            "mask_pixels",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_row in predictions:
            for record in image_row["instances"]:
                bbox = record["bbox_xyxy"] or [None, None, None, None]
                writer.writerow(
                    {
                        "image": image_row["image"],
                        "source_image": image_row["source_images"][
                            record["class_name"]
                        ],
                        "class_name": record["class_name"],
                        "instance_id": record["instance_id"],
                        "score": record["score"],
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                        "mask_pixels": record["mask_pixels"],
                    }
                )


def run(args: argparse.Namespace) -> Path:
    classes = selected_classes(args.classes)
    base_class = plot_base_class(args, classes)
    if args.batch_size < 1:
        raise InferenceError("--batch-size must be at least 1.")
    if not 1 <= args.jpg_quality <= 100:
        raise InferenceError("--jpg-quality must be between 1 and 100.")
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = paired_images(args, classes)

    models = {
        class_name: load_model(
            class_name,
            model_dir_arg_for_class(args, class_name),
            args.score_threshold,
            args.device,
        )
        for class_name in classes
    }

    print_table(
        "Alhambra Inference",
        [
            ("classes", ", ".join(classes)),
            (
                "inputs",
                ", ".join(
                    f"{name}={relative_path(input_path_for_class(args, name))}"
                    for name in classes
                ),
            ),
            ("images", len(pairs)),
            (
                "models",
                ", ".join(
                    f"{name}={relative_path(models[name]['model_dir'])}"
                    for name in classes
                ),
            ),
            ("score threshold", args.score_threshold),
            ("device", models[classes[0]]["cfg"].MODEL.DEVICE),
            ("batch size", args.batch_size),
            ("output mode", args.output_mode),
            ("plot base", base_class),
            ("output", relative_path(output_dir)),
        ],
    )

    predictions = []
    for batch in chunked(pairs, args.batch_size):
        batch_items = []
        for key, paths_by_class in batch:
            images = {
                class_name: read_image(str(paths_by_class[class_name]), format="BGR")
                for class_name in classes
            }
            height, width = images[base_class].shape[:2]
            for class_name, image in images.items():
                if image.shape[:2] != (height, width):
                    raise InferenceError(
                        f"Paired images must have the same dimensions: {key} "
                        f"{base_class}={width}x{height}, "
                        f"{class_name}={image.shape[1]}x{image.shape[0]}"
                    )
            batch_items.append((key, paths_by_class, images, height, width))

        predicted_by_class = {}
        for class_name in classes:
            outputs = models[class_name]["predictor"].predict_batch(
                [item[2][class_name] for item in batch_items]
            )
            predicted_by_class[class_name] = [
                output["instances"].to("cpu") for output in outputs
            ]

        for index, (key, paths_by_class, images, height, width) in enumerate(
            batch_items
        ):
            masks = {}
            instances = []
            instance_counts = {}
            for class_name in classes:
                predicted = predicted_by_class[class_name][index]
                masks[class_name] = mask_from_instances(predicted, height, width)
                instances.extend(instance_records(predicted, class_name))
                instance_counts[class_name] = len(predicted)

            outputs = save_visuals(
                images[base_class],
                masks,
                classes,
                output_dir,
                key,
                args.alpha,
                base_class,
                args.output_mode,
                args.jpg_quality,
                args.optimize_jpg,
            )
            predicted_pixels = {
                class_name: int(masks[class_name].sum()) for class_name in classes
            }
            if "aerosol" in classes and "cloud" in classes:
                predicted_pixels["overlap"] = int(
                    (masks["aerosol"] & masks["cloud"]).sum()
                )

            predictions.append(
                {
                    "image": key,
                    "source_images": {
                        class_name: str(paths_by_class[class_name])
                        for class_name in classes
                    },
                    "height": int(height),
                    "width": int(width),
                    "num_instances": len(instances),
                    "predicted_pixels_by_class": predicted_pixels,
                    "instances": instances,
                    "outputs": outputs,
                }
            )
            counts = ", ".join(
                f"{class_name}={instance_counts[class_name]} instances"
                for class_name in classes
            )
            print(f"{key}: {counts}")

    write_predictions(output_dir, predictions)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "classes": classes,
                "inputs": {
                    class_name: str(input_path_for_class(args, class_name))
                    for class_name in classes
                },
                "num_images": len(pairs),
                "plot_base_class": base_class,
                "colors": CLASS_COLORS,
                "score_threshold": args.score_threshold,
                "alpha": args.alpha,
                "batch_size": args.batch_size,
                "output_mode": args.output_mode,
                "jpg_quality": args.jpg_quality,
                "optimize_jpg": args.optimize_jpg,
                "device": models[classes[0]]["cfg"].MODEL.DEVICE,
                "models": {
                    class_name: {
                        "model_dir": str(models[class_name]["model_dir"]),
                        "config": str(models[class_name]["config"]),
                        "weights": str(models[class_name]["weights"]),
                    }
                    for class_name in classes
                },
                "output": str(output_dir),
            },
            handle,
            indent=2,
        )
    return output_dir


def main() -> None:
    configure_quiet_logging()
    output_dir = run(parse_args())
    print_table(
        "Outputs",
        [
            ("comparison plots", relative_path(output_dir / "plots")),
            (
                "translucent overlays",
                relative_path(output_dir / "overlays_translucent_jpg"),
            ),
            ("opaque overlays", relative_path(output_dir / "overlays_opaque_jpg")),
            ("masks", relative_path(output_dir / "masks_png")),
            ("predictions", relative_path(output_dir / "predictions.json")),
        ],
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
