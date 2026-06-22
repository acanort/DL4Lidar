# DL4Lidar: Aerosol and Cloud Segmentation

This repository contains Detectron2 code, COCO-format annotations, selected
lidar images, and trained models for instance segmentation of atmospheric
aerosol and cloud layers.

## Published contents

```text
DL4Lidar/
  data/
    alhambra/
      annotations/                 # aerosol, cloud, and combined annotations
      images/
        1064fta/3e8/               # published image subset
        532fta/2e7/                # published image subset
    mulhacen/
      annotations/                 # all available Mulhacen annotations
      images/                      # all available Mulhacen images
  models/alhambra/
    aerosol/
      aerosol_config.yaml
      aerosol_weights.pth
    cloud/
      cloud_config.yaml
      cloud_weights.pth
  src/
    train.py
    inference.py
  environment.yml
  requirements.txt
  pyproject.toml
```

Only the data paths shown above are tracked. Other local datasets, including
`data/chm15k/`, are intentionally excluded by `.gitignore`. Generated training
and inference outputs are also excluded.

The model weights are stored with Git LFS. Install Git LFS before cloning if you
want Git to download the weight files automatically:

```bash
git lfs install
git clone https://github.com/acanort/DL4Lidar.git
cd DL4Lidar
```

## Environment

The pinned environment targets Python 3.9, PyTorch 2.8, CUDA 12.8, and a
matching Detectron2 build. Create the Conda environment and install the pinned
Python packages with:

```bash
conda env create -f environment.yml
conda activate d2-rtx5070
pip install -r requirements.txt
```

The PyTorch and Detectron2 versions in `requirements.txt` are CUDA-specific.
They may require the corresponding wheel indexes, and must be changed for a
different CUDA or CPU-only environment.

## Dataset layout

The training code expects images and COCO annotation JSON files in this form:

```text
data/{instrument}/
  annotations/{annotation_set}/
    holdout/
      train.json
      valid.json
      test.json
    5_fold_cross_validation/
      train_fold_1.json
      valid_fold_1.json
      ...
      train_fold_5.json
      valid_fold_5.json
      test.json
  images/{channel}/{scale}/
    image files
```

The published Alhambra annotations include `aerosol`, `cloud`, and `combined`
sets. The published Alhambra images are limited to `1064fta/3e8` and
`532fta/2e7`. Mulhacen includes its available annotation and image folders.

## Training

`src/train.py` implements one train/validation/test dataset workflow. It
provides:

- COCO annotation and image validation
- Detectron2 dataset registration and configuration
- validation-loss monitoring
- learning-rate reduction on a plateau
- best-checkpoint restoration
- early stopping
- metric logs, a run summary, and a training plot

There is no command-line interface for training. Edit the user-configuration
block near the top of `src/train.py`, then run:

```bash
python src/train.py
```

The committed defaults select the included Alhambra cloud holdout data at
`data/alhambra/images/1064fta/3e8`. Important settings include:

```python
INSTRUMENT = "alhambra"
ANNOTATION_SET = "cloud"
CHANNEL = "1064fta"
SCALE = "3e8"
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
MAX_EPOCHS = 300
IMS_PER_BATCH = 8
```

Runs are written below `outputs/train/` by default. Each timestamped run can
contain:

```text
checkpoints/model_best.pth
config/config.yaml
logs/metrics.json
logs/best_model.json
logs/training_summary.json
plots/training_process.png
```

To create a holdout or cross-validation variant, copy `src/train.py` locally
and adapt its configuration and orchestration. Files such as
`train_holdout.py` and `train_5_fold_cross_validation.py` are not part of this
repository.

## Inference

`src/inference.py` loads the published model config and weight pair for each
selected class. The default command runs the cloud model on the included
Alhambra `1064fta/3e8` images:

```bash
python src/inference.py
```

Run both aerosol and cloud models on the same published image folder:

```bash
python src/inference.py \
  --classes aerosol,cloud \
  --aerosol-input data/alhambra/images/1064fta/3e8 \
  --cloud-input data/alhambra/images/1064fta/3e8 \
  --output inference/example_run \
  --score-threshold 0.7 \
  --batch-size 1
```

Useful options include:

```bash
python src/inference.py --help
python src/inference.py --classes aerosol
python src/inference.py --output-mode predictions
python src/inference.py --output-mode quick --max-images 10
python src/inference.py --device cpu
```

The output directory always receives `predictions.json`, `predictions.csv`,
and `run_summary.json`. Depending on `--output-mode`, it can also contain PNG
masks, JPEG overlays, and comparison plots. Inference outputs are ignored by
Git.

## Notes

- Model weights are large Git LFS objects: approximately 480 MB for aerosol
  and 953 MB for cloud.
- `outputs/`, `inference/`, caches, virtual environments, and unpublished data
  remain local and are not committed.
- Keep `.gitignore` and this README synchronized whenever the published data or
  source-file selection changes.
