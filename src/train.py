"""Base Detectron2 training script.

This is the script to copy and adapt when creating a new training scenario.
It trains one Mask R-CNN instance-segmentation model from three COCO-style
annotation files: train, validation, and test.

Typical workflow:

1. Copy this file to a scenario-specific name, for example
   ``train_cloud_1064fta_3e8.py``.
2. Edit the user configuration block below.
3. If the scenario needs a different split strategy, such as k-fold
   cross-validation, keep the shared helpers and modify only the dataset
   registration/run section.

The scripts ``train_holdout.py`` and ``train_5_fold_cross_validation.py`` are
examples of such adapted scripts.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings(
    "ignore", message=".*distutils Version classes are deprecated.*"
)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*indexing argument.*")

from detectron2 import model_zoo  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import (  # noqa: E402
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.data.datasets import register_coco_instances  # noqa: E402
from detectron2.engine import DefaultTrainer  # noqa: E402
from detectron2.engine.hooks import (  # noqa: E402
    EvalHook,
    HookBase,
    LRScheduler,
    PeriodicCheckpointer,
)
from detectron2.utils.events import JSONWriter, TensorboardXWriter  # noqa: E402

try:
    from rich.console import Console
    from rich.table import Table

    console = Console()
except ImportError:
    console = None


# --- USER CONFIGURATION ---------------------------------------------------------

DATA_ROOT = Path("data")
INSTRUMENT = "alhambra"
ANNOTATION_SET = "cloud"
CHANNEL = "1064fta"
SCALE = "3e8"

IMAGE_ROOT = DATA_ROOT / INSTRUMENT / "images" / CHANNEL / SCALE
ANNOTATION_ROOT = DATA_ROOT / INSTRUMENT / "annotations" / ANNOTATION_SET / "holdout"
TRAIN_JSON = ANNOTATION_ROOT / "train.json"
VALID_JSON = ANNOTATION_ROOT / "valid.json"
TEST_JSON = ANNOTATION_ROOT / "test.json"

ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
MAX_EPOCHS = 300
IMS_PER_BATCH = 8
BASE_LR = 1e-3
MIN_LR = 1e-6
NUM_WORKERS = 2
ROI_BATCH_SIZE_PER_IMAGE = 64
USE_AMP = True

PLATEAU_PATIENCE = 5
PLATEAU_FACTOR = 0.5
PLATEAU_THRESHOLD = 1e-4

EARLY_STOPPING_PATIENCE = 3 * PLATEAU_PATIENCE
EARLY_STOPPING_MIN_DELTA = 0.0

RUN_NAME = f"{INSTRUMENT}_{ANNOTATION_SET}_{CHANNEL}_{SCALE}_{ARCHITECTURE}"
OUTPUT_ROOT = Path("outputs/train")
USE_TIMESTAMPED_OUTPUT = True
RESUME_TRAINING = False


@dataclass
class TrainingConfig:
    """Configuration for one ordinary train/validation/test run."""

    train_json: Path
    valid_json: Path
    test_json: Path
    image_root: Path
    output_dir: Path
    architecture: str = ARCHITECTURE
    max_epochs: int = MAX_EPOCHS
    ims_per_batch: int = IMS_PER_BATCH
    base_lr: float = BASE_LR
    min_lr: float = MIN_LR
    num_workers: int = NUM_WORKERS
    roi_batch_size_per_image: int = ROI_BATCH_SIZE_PER_IMAGE
    use_amp: bool = USE_AMP
    plateau_patience: int = PLATEAU_PATIENCE
    plateau_factor: float = PLATEAU_FACTOR
    plateau_threshold: float = PLATEAU_THRESHOLD
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA
    resume_training: bool = RESUME_TRAINING
    dataset_prefix: str = RUN_NAME


class StopEarly(Exception):
    """Raised by the early-stopping hook to end training cleanly."""


class DatasetError(RuntimeError):
    """Raised when the selected dataset cannot be used for training."""


def configure_quiet_logging() -> None:
    """Reduce Detectron2/FVCore log noise while preserving useful script output."""

    for logger_name in ("detectron2", "fvcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def timestamped_output_dir() -> Path:
    """Return the default output directory for this base script."""

    output_dir = OUTPUT_ROOT / RUN_NAME
    if USE_TIMESTAMPED_OUTPUT:
        output_dir = output_dir / dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return output_dir


def default_config() -> TrainingConfig:
    """Build a config from the user configuration block."""

    return TrainingConfig(
        train_json=TRAIN_JSON,
        valid_json=VALID_JSON,
        test_json=TEST_JSON,
        image_root=IMAGE_ROOT,
        output_dir=timestamped_output_dir(),
    )


def output_layout(output_dir: Path) -> dict[str, Path]:
    return {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "config": output_dir / "config",
        "logs": output_dir / "logs",
        "plots": output_dir / "plots",
    }


def prepare_output_layout(output_dir: Path) -> dict[str, Path]:
    layout = output_layout(output_dir)
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def print_table(title: str, rows: list[tuple[str, object]]) -> None:
    if console:
        table = Table(title=title)
        table.add_column("Item")
        table.add_column("Value")
        for key, value in rows:
            table.add_row(str(key), str(value))
        console.print(table)
        return

    print(f"{title}:")
    for key, value in rows:
        print(f"  {key}: {value}")


def load_coco_json(path: Path) -> dict:
    if not path.is_file():
        raise DatasetError(f"Annotation file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if key not in coco or not isinstance(coco[key], list):
            raise DatasetError(f"{path} must contain a list named '{key}'")
    return coco


def thing_classes(path: Path) -> list[str]:
    coco = load_coco_json(path)
    classes = [category["name"] for category in coco["categories"]]
    if not classes:
        raise DatasetError(f"{path} does not define any categories")
    return classes


def validate_dataset_files(config: TrainingConfig) -> None:
    """Check that annotations exist and referenced images are present."""

    if not config.image_root.is_dir():
        raise DatasetError(f"Image directory does not exist: {config.image_root}")

    for annotation_path in (
        config.train_json,
        config.valid_json,
        config.test_json,
    ):
        coco = load_coco_json(annotation_path)
        missing_images = [
            image["file_name"]
            for image in coco["images"]
            if not (config.image_root / image["file_name"]).is_file()
        ]
        if missing_images:
            preview = ", ".join(missing_images[:8])
            suffix = (
                ""
                if len(missing_images) <= 8
                else f" ... (+{len(missing_images) - 8})"
            )
            raise DatasetError(
                f"{annotation_path} references missing images in "
                f"{config.image_root}: {preview}{suffix}"
            )


def clear_registered_datasets() -> None:
    """Clear Detectron2 dataset catalogs before registering this run."""

    for dataset_name in list(DatasetCatalog.list()):
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)


def register_training_datasets(
    config: TrainingConfig,
) -> tuple[str, str, str, list[str]]:
    """Register train/valid/test COCO datasets and return their names."""

    validate_dataset_files(config)
    clear_registered_datasets()

    train_name = f"{config.dataset_prefix}_train"
    valid_name = f"{config.dataset_prefix}_valid"
    test_name = f"{config.dataset_prefix}_test"
    classes = thing_classes(config.train_json)

    for name, annotation_path in (
        (train_name, config.train_json),
        (valid_name, config.valid_json),
        (test_name, config.test_json),
    ):
        register_coco_instances(
            name,
            {"thing_classes": classes},
            str(annotation_path),
            str(config.image_root),
        )
        MetadataCatalog.get(name).set(thing_classes=classes)

    return train_name, valid_name, test_name, classes


def build_config(
    config: TrainingConfig,
    train_name: str,
    valid_name: str,
    num_classes: int,
):
    """Build and persist the Detectron2 config for this training run."""

    layout = prepare_output_layout(config.output_dir)
    cfg = get_cfg()
    config_file = f"COCO-InstanceSegmentation/{config.architecture}.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

    num_train_images = len(DatasetCatalog.get(train_name))
    iter_per_epoch = max(1, math.ceil(num_train_images / config.ims_per_batch))

    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (valid_name,)
    cfg.DATALOADER.NUM_WORKERS = config.num_workers
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.roi_batch_size_per_image
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.SOLVER.IMS_PER_BATCH = config.ims_per_batch
    cfg.SOLVER.BASE_LR = config.base_lr
    cfg.SOLVER.MAX_ITER = config.max_epochs * iter_per_epoch
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER + 1
    cfg.SOLVER.LOG_PERIOD = iter_per_epoch
    cfg.SOLVER.AMP.ENABLED = config.use_amp
    cfg.TEST.EVAL_PERIOD = 0
    cfg.OUTPUT_DIR = str(config.output_dir)

    with (layout["config"] / "config.yaml").open("w", encoding="utf-8") as handle:
        handle.write(cfg.dump())

    return cfg, iter_per_epoch


def latest_scalar(storage, name: str) -> Optional[float]:
    if name not in storage._history:
        return None
    value = storage.history(name).latest()
    return float(value[0] if isinstance(value, tuple) else value)


class ValidationLossHook(HookBase):
    """Compute validation loss at epoch boundaries."""

    def __init__(self, model, data_loader, iter_per_epoch: int):
        self.model = model
        self.data_loader = data_loader
        self.iter_per_epoch = iter_per_epoch

    def after_step(self) -> None:
        next_iter = self.trainer.iter + 1
        if next_iter % self.iter_per_epoch != 0 and next_iter != self.trainer.max_iter:
            return

        val_loss, components = self._evaluate_loss()
        self.trainer.storage.put_scalar(
            "validation_loss", val_loss, smoothing_hint=False
        )
        for name, value in components.items():
            self.trainer.storage.put_scalar(
                f"validation/{name}", value, smoothing_hint=False
            )

    def _evaluate_loss(self) -> tuple[float, dict[str, float]]:
        was_training = self.model.training
        self.model.train()

        total_losses = []
        component_losses = defaultdict(list)
        with torch.no_grad():
            for inputs in self.data_loader:
                loss_dict = self.model(inputs)
                reduced = {
                    key: float(value.detach().cpu()) for key, value in loss_dict.items()
                }
                for key, value in reduced.items():
                    component_losses[key].append(value)
                total_losses.append(sum(reduced.values()))

        if not was_training:
            self.model.eval()

        components = {
            key: float(np.mean(values)) for key, values in component_losses.items()
        }
        return float(np.mean(total_losses)), components


class ReduceLROnPlateauHook(HookBase):
    """Reduce LR after validation-loss plateaus and restore the best checkpoint."""

    def __init__(self, config: TrainingConfig, iter_per_epoch: int):
        self.config = config
        self.iter_per_epoch = iter_per_epoch
        self.scheduler = None
        self.best_checkpoint = None

    def before_train(self) -> None:
        layout = prepare_output_layout(Path(self.trainer.cfg.OUTPUT_DIR))
        self.best_checkpoint = layout["checkpoints"] / "model_best.pth"
        self.scheduler = ReduceLROnPlateau(
            self.trainer.optimizer,
            mode="min",
            patience=max(0, self.config.plateau_patience - 1),
            factor=self.config.plateau_factor,
            threshold=self.config.plateau_threshold,
            min_lr=self.config.min_lr,
        )

    def after_step(self) -> None:
        next_iter = self.trainer.iter + 1
        if next_iter % self.iter_per_epoch != 0 and next_iter != self.trainer.max_iter:
            return

        val_loss = latest_scalar(self.trainer.storage, "validation_loss")
        if val_loss is None:
            return

        old_lrs = [group["lr"] for group in self.trainer.optimizer.param_groups]
        self.scheduler.step(val_loss)
        new_lrs = [group["lr"] for group in self.trainer.optimizer.param_groups]
        lr_reduced = old_lrs != new_lrs
        if lr_reduced:
            self._restore_best_checkpoint(new_lrs)
        self.trainer.storage.put_scalar(
            "plateau/lr_reduced", int(lr_reduced), smoothing_hint=False
        )
        self.trainer.storage.put_scalar(
            "plateau/restored_best_checkpoint", int(lr_reduced), smoothing_hint=False
        )

    def _restore_best_checkpoint(self, learning_rates: list[float]) -> None:
        if not self.best_checkpoint or not self.best_checkpoint.exists():
            raise FileNotFoundError(
                "Cannot restore the best model after reducing the learning rate: "
                f"{self.best_checkpoint}"
            )

        checkpointer = DetectionCheckpointer(
            self.trainer.model,
            optimizer=self.trainer.optimizer,
        )
        checkpointer.load(str(self.best_checkpoint))
        for param_group, lr in zip(self.trainer.optimizer.param_groups, learning_rates):
            param_group["lr"] = lr

        if hasattr(self.scheduler, "_last_lr"):
            self.scheduler._last_lr = learning_rates


class EarlyStoppingHook(HookBase):
    """Save the best checkpoint and stop after repeated validation plateaus."""

    def __init__(self, config: TrainingConfig, iter_per_epoch: int):
        self.config = config
        self.iter_per_epoch = iter_per_epoch
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.history = []
        self.checkpoints_dir = None
        self.logs_dir = None

    def before_train(self) -> None:
        layout = prepare_output_layout(Path(self.trainer.cfg.OUTPUT_DIR))
        self.checkpoints_dir = layout["checkpoints"]
        self.logs_dir = layout["logs"]

    def after_step(self) -> None:
        next_iter = self.trainer.iter + 1
        if next_iter % self.iter_per_epoch != 0 and next_iter != self.trainer.max_iter:
            return

        val_loss = latest_scalar(self.trainer.storage, "validation_loss")
        if val_loss is None:
            return

        epoch = math.ceil(next_iter / self.iter_per_epoch)
        lr = self.trainer.optimizer.param_groups[0]["lr"]
        lr_reduced = latest_scalar(self.trainer.storage, "plateau/lr_reduced") == 1
        restored_checkpoint = (
            latest_scalar(self.trainer.storage, "plateau/restored_best_checkpoint")
            == 1
        )
        improved = val_loss < self.best_loss - self.config.early_stopping_min_delta

        if improved:
            previous_best = self.best_loss
            self.best_loss = val_loss
            self.bad_epochs = 0
            self._save_best_checkpoint(epoch, val_loss, lr)
            status = "improved"
            detail = (
                "first validation"
                if math.isinf(previous_best)
                else f"delta={previous_best - val_loss:.6f}"
            )
        else:
            self.bad_epochs += 1
            status = (
                f"no improvement "
                f"({self.bad_epochs}/{self.config.early_stopping_patience})"
            )
            detail = f"best={self.best_loss:.6f}"
            if lr_reduced:
                status = f"{status}; LR reduced"
                detail = f"{detail}; restored best checkpoint"

        self.trainer.storage.put_scalar(
            "early_stopping/bad_epochs", self.bad_epochs, smoothing_hint=False
        )
        self.history.append(
            {
                "epoch": epoch,
                "iter": self.trainer.iter,
                "validation_loss": val_loss,
                "best_loss": self.best_loss,
                "bad_epochs": self.bad_epochs,
                "lr": lr,
                "status": status,
                "detail": detail,
                "improved": improved,
                "lr_reduced": lr_reduced,
                "restored_best_checkpoint": restored_checkpoint,
                "best_checkpoint": str(self.checkpoints_dir / "model_best.pth"),
            }
        )
        self._write_history()
        self._print_recent_history()

        if self.bad_epochs >= self.config.early_stopping_patience:
            raise StopEarly("early stopping triggered")

    def _save_best_checkpoint(self, epoch: int, val_loss: float, lr: float) -> None:
        original_save_dir = self.trainer.checkpointer.save_dir
        self.trainer.checkpointer.save_dir = str(self.checkpoints_dir)
        try:
            self.trainer.checkpointer.save("model_best")
        finally:
            self.trainer.checkpointer.save_dir = original_save_dir

        root_best = Path(self.trainer.cfg.OUTPUT_DIR) / "model_best.pth"
        if root_best.exists():
            root_best.unlink()

        last_checkpoint = self.checkpoints_dir / "last_checkpoint"
        if last_checkpoint.exists():
            last_checkpoint.unlink()

        metadata = {
            "epoch": epoch,
            "iteration": self.trainer.iter,
            "validation_loss": val_loss,
            "learning_rate": lr,
            "checkpoint": str(self.checkpoints_dir / "model_best.pth"),
            "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        with (self.logs_dir / "best_model.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def _write_history(self) -> None:
        with (self.logs_dir / "validation_history.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(self.history, handle, indent=2)

    def _print_recent_history(self) -> None:
        records = self.history[-(self.config.early_stopping_patience + 1) :]
        if console:
            table = Table(title="Recent Validation History")
            table.add_column("Epoch")
            table.add_column("Val Loss")
            table.add_column("Best")
            table.add_column("Bad")
            table.add_column("LR")
            table.add_column("Status")
            for record in records:
                table.add_row(
                    str(record["epoch"]),
                    f"{record['validation_loss']:.6f}",
                    f"{record['best_loss']:.6f}",
                    str(record["bad_epochs"]),
                    f"{record['lr']:.2e}",
                    record["status"],
                )
            console.print(table)
            return

        print("Recent validation history:")
        for record in records:
            print(
                f"  epoch={record['epoch']} val={record['validation_loss']:.6f} "
                f"best={record['best_loss']:.6f} bad={record['bad_epochs']} "
                f"lr={record['lr']:.2e} {record['status']}"
            )


class Trainer(DefaultTrainer):
    """Detectron2 trainer with validation-loss and early-stopping hooks."""

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return None

    def __init__(self, cfg, iter_per_epoch: int, config: TrainingConfig):
        self.iter_per_epoch = iter_per_epoch
        self.config = config
        self.output_dir = Path(cfg.OUTPUT_DIR)
        super().__init__(cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks = [
            hook
            for hook in hooks
            if not isinstance(hook, (EvalHook, LRScheduler, PeriodicCheckpointer))
        ]

        val_loader = build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            DatasetMapper(self.cfg, is_train=True),
        )
        hooks.insert(
            -1, ValidationLossHook(self.model, val_loader, self.iter_per_epoch)
        )
        hooks.insert(-1, ReduceLROnPlateauHook(self.config, self.iter_per_epoch))
        hooks.insert(-1, EarlyStoppingHook(self.config, self.iter_per_epoch))
        return hooks

    def build_writers(self):
        logs_dir = output_layout(self.output_dir)["logs"]
        logs_dir.mkdir(parents=True, exist_ok=True)
        return [
            JSONWriter(str(logs_dir / "metrics.json")),
            TensorboardXWriter(str(logs_dir)),
        ]

    def train(self):
        try:
            return super().train()
        except StopEarly:
            return None


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def epoch_from_iteration(iteration: int, iter_per_epoch: int) -> int:
    return max(1, math.ceil((iteration + 1) / iter_per_epoch))


def training_loss_by_epoch(output_dir: Path, iter_per_epoch: int) -> dict[int, float]:
    metrics = read_json_lines(output_layout(output_dir)["logs"] / "metrics.json")
    losses = {}
    for row in metrics:
        if "total_loss" not in row or "iteration" not in row:
            continue
        epoch = epoch_from_iteration(int(row["iteration"]), iter_per_epoch)
        losses[epoch] = float(row["total_loss"])
    return losses


def validation_history(config: TrainingConfig, iter_per_epoch: int) -> list[dict]:
    layout = output_layout(config.output_dir)
    history = read_json(layout["logs"] / "validation_history.json", [])
    if history:
        return history

    metrics = read_json_lines(layout["logs"] / "metrics.json")
    fallback = []
    best_loss = float("inf")
    bad_epochs = 0
    for row in metrics:
        if "validation_loss" not in row or "iteration" not in row:
            continue
        val_loss = float(row["validation_loss"])
        epoch = epoch_from_iteration(int(row["iteration"]), iter_per_epoch)
        improved = val_loss < best_loss - config.early_stopping_min_delta
        if improved:
            best_loss = val_loss
            bad_epochs = 0
            status = "improved"
        else:
            bad_epochs += 1
            status = f"no improvement ({bad_epochs}/{config.early_stopping_patience})"
        fallback.append(
            {
                "epoch": epoch,
                "iter": int(row["iteration"]),
                "validation_loss": val_loss,
                "best_loss": best_loss,
                "bad_epochs": bad_epochs,
                "lr": row.get("lr"),
                "status": status,
            }
        )
    return fallback


def lr_reduction_windows(
    config: TrainingConfig, history: list[dict]
) -> list[list[dict]]:
    windows = []
    used_indices = set()
    for index, record in enumerate(history):
        lr_reduced = record.get("lr_reduced") or "LR reduced" in record.get(
            "status", ""
        )
        if not lr_reduced:
            continue

        start = max(0, index - config.plateau_patience + 1)
        indices = [
            candidate
            for candidate in range(start, index + 1)
            if candidate not in used_indices
        ]
        if not indices:
            continue

        used_indices.update(indices)
        windows.append([history[candidate] for candidate in indices])
    return windows


def main_branch_records(config: TrainingConfig, history: list[dict]) -> list[dict]:
    branch_epochs = {
        int(record["epoch"])
        for window in lr_reduction_windows(config, history)
        for record in window
    }
    return [record for record in history if int(record["epoch"]) not in branch_epochs]


def plot_training_process(
    config: TrainingConfig, iter_per_epoch: int
) -> Optional[Path]:
    train_losses = training_loss_by_epoch(config.output_dir, iter_per_epoch)
    val_history = validation_history(config, iter_per_epoch)
    if not train_losses and not val_history:
        return None

    layout = output_layout(config.output_dir)
    plot_path = layout["plots"] / "training_process.png"
    layout["plots"].mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    branch_windows = lr_reduction_windows(config, val_history)
    trunk_records = main_branch_records(config, val_history)
    trunk_x = {
        int(record["epoch"]): mapped_epoch
        for mapped_epoch, record in enumerate(trunk_records, start=1)
    }
    branch_colors = [
        "#f97316",
        "#8b5cf6",
        "#14b8a6",
        "#ec4899",
        "#f59e0b",
        "#64748b",
    ]

    if train_losses and trunk_records:
        train_epochs = [
            int(record["epoch"])
            for record in trunk_records
            if int(record["epoch"]) in train_losses
        ]
        ax.plot(
            [trunk_x[epoch] for epoch in train_epochs],
            [train_losses[epoch] for epoch in train_epochs],
            color="#2563eb",
            linewidth=2.2,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="train total loss",
        )

    failed_valid_label_added = False
    lr_label_added = False

    if trunk_records:
        val_epochs = [int(record["epoch"]) for record in trunk_records]
        val_losses = [float(record["validation_loss"]) for record in trunk_records]
        ax.plot(
            [trunk_x[epoch] for epoch in val_epochs],
            val_losses,
            color="#dc2626",
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="valid loss",
        )
        ax.plot(
            [trunk_x[epoch] for epoch in val_epochs],
            [float(record["best_loss"]) for record in trunk_records],
            color="#16a34a",
            linewidth=1.8,
            linestyle="-.",
            marker="s",
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label="best valid loss so far",
        )

    for branch_index, window in enumerate(branch_windows, start=1):
        first_epoch = int(window[0]["epoch"])
        anchor_candidates = [
            record
            for record in trunk_records
            if int(record["epoch"]) < first_epoch
        ]
        if not anchor_candidates:
            continue

        anchor = anchor_candidates[-1]
        anchor_epoch = int(anchor["epoch"])
        anchor_x = trunk_x[anchor_epoch]
        branch_x = [anchor_x] + [
            anchor_x + offset for offset in range(1, len(window) + 1)
        ]

        branch_val_y = [float(anchor["validation_loss"])] + [
            float(record["validation_loss"]) for record in window
        ]
        branch_color = branch_colors[(branch_index - 1) % len(branch_colors)]
        ax.plot(
            branch_x,
            branch_val_y,
            color=branch_color,
            linestyle="--",
            linewidth=2,
            marker="X",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            alpha=0.95,
            label=None if failed_valid_label_added else "failed valid branches",
        )
        failed_valid_label_added = True

        if train_losses and anchor_epoch in train_losses:
            branch_train_epochs = [
                int(record["epoch"])
                for record in window
                if int(record["epoch"]) in train_losses
            ]
            if len(branch_train_epochs) == len(window):
                branch_train_y = [train_losses[anchor_epoch]] + [
                    train_losses[epoch] for epoch in branch_train_epochs
                ]
                ax.plot(
                    branch_x,
                    branch_train_y,
                    color=branch_color,
                    linestyle=":",
                    linewidth=1.5,
                    marker="P",
                    markersize=5,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    alpha=0.55,
                )

        reduction_epoch = int(window[-1]["epoch"])
        ax.scatter(
            branch_x[-1],
            branch_val_y[-1],
            color=branch_color,
            marker="D",
            s=66,
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
            label=None if lr_label_added else "LR reduction",
        )
        lr_label_added = True
        ax.annotate(
            f"LR @ {reduction_epoch}",
            xy=(branch_x[-1], branch_val_y[-1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#6d28d9",
        )

    ax.set_title("Training Process")
    ax.set_xlabel("remapped epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def restore_best_model(trainer: Trainer, output_dir: Path) -> Optional[Path]:
    best = read_json(output_layout(output_dir)["logs"] / "best_model.json", {})
    checkpoint = best.get("checkpoint")
    if not checkpoint:
        return None

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return None

    DetectionCheckpointer(trainer.model).load(str(checkpoint_path))
    return checkpoint_path


def final_training_summary(
    config: TrainingConfig,
    elapsed_seconds: float,
    iter_per_epoch: int,
    plot_path: Optional[Path],
    final_model_checkpoint: Optional[Path],
) -> None:
    layout = output_layout(config.output_dir)
    best = read_json(layout["logs"] / "best_model.json", {})
    history = validation_history(config, iter_per_epoch)
    train_losses = training_loss_by_epoch(config.output_dir, iter_per_epoch)
    final_record = history[-1] if history else {}

    lr_reductions = len(lr_reduction_windows(config, history))
    epochs_completed = int(final_record.get("epoch", 0)) if final_record else 0
    last_train_loss = train_losses.get(max(train_losses)) if train_losses else None
    stopped_early = (
        bool(final_record)
        and int(final_record.get("bad_epochs", 0)) >= config.early_stopping_patience
    )

    rows = [
        ("epochs completed", epochs_completed or "unknown"),
        ("iterations per epoch", iter_per_epoch),
        ("elapsed minutes", f"{elapsed_seconds / 60:.2f}"),
        (
            "last evaluated train loss",
            f"{last_train_loss:.6f}" if last_train_loss is not None else "n/a",
        ),
        (
            "last evaluated valid loss",
            (
                f"{float(final_record['validation_loss']):.6f}"
                if final_record
                else "n/a"
            ),
        ),
        (
            "best valid loss",
            (
                f"{float(best['validation_loss']):.6f}"
                if "validation_loss" in best
                else "n/a"
            ),
        ),
        ("best epoch", best.get("epoch", "n/a")),
        ("LR reductions", lr_reductions),
        ("stopped early", "yes" if stopped_early else "no"),
        (
            "final model",
            relative_path(final_model_checkpoint) if final_model_checkpoint else "n/a",
        ),
        ("final model source", "best checkpoint" if final_model_checkpoint else "n/a"),
        ("plot", relative_path(plot_path) if plot_path else "n/a"),
        ("metrics", relative_path(layout["logs"] / "metrics.json")),
    ]
    print_table("Training Summary", rows)


def write_training_summary(
    config: TrainingConfig,
    elapsed_seconds: float,
    plot_path: Optional[Path],
    final_model_checkpoint: Optional[Path],
    extra: Optional[dict] = None,
) -> dict:
    layout = output_layout(config.output_dir)
    summary = {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60,
        "output_dir": str(config.output_dir),
        "plot": str(plot_path) if plot_path else None,
        "final_model": str(final_model_checkpoint) if final_model_checkpoint else None,
        "final_model_source": "best_checkpoint" if final_model_checkpoint else None,
        "best_model": read_json(layout["logs"] / "best_model.json", {}),
    }
    if extra:
        summary.update(extra)

    with (layout["logs"] / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_training(config: TrainingConfig, extra_summary: Optional[dict] = None) -> dict:
    """Run one complete train/validation/test training scenario."""

    configure_quiet_logging()
    train_name, valid_name, test_name, classes = register_training_datasets(config)
    print_table(
        "Training Inputs",
        [
            ("images", relative_path(config.image_root)),
            ("train annotations", relative_path(config.train_json)),
            ("valid annotations", relative_path(config.valid_json)),
            ("test annotations", relative_path(config.test_json)),
            ("train dataset", train_name),
            ("valid dataset", valid_name),
            ("test dataset", test_name),
            ("classes", ", ".join(classes)),
            ("output", relative_path(config.output_dir)),
        ],
    )

    cfg, iter_per_epoch = build_config(
        config,
        train_name,
        valid_name,
        len(classes),
    )

    start = time.time()
    trainer = Trainer(cfg, iter_per_epoch, config)
    trainer.resume_or_load(resume=config.resume_training)
    trainer.train()
    elapsed_seconds = time.time() - start
    final_model_checkpoint = restore_best_model(trainer, config.output_dir)
    plot_path = plot_training_process(config, iter_per_epoch)

    summary = write_training_summary(
        config,
        elapsed_seconds,
        plot_path,
        final_model_checkpoint,
        extra=extra_summary,
    )
    final_training_summary(
        config,
        elapsed_seconds,
        iter_per_epoch,
        plot_path,
        final_model_checkpoint,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    run_training(default_config())


if __name__ == "__main__":
    main()
