# DL4Lidar: Aerosol and Cloud Segmentation

DL4Lidar uses Detectron2 instance-segmentation models to identify atmospheric
aerosol and cloud layers in lidar images. The repository provides training and
inference code, COCO-format annotations, example image datasets, and trained
Alhambra models.

## Repository structure

```text
DL4Lidar/
  data/
    alhambra/
      annotations/
      images/
        1064fta/3e8/
        532fta/2e7/
    mulhacen/
      annotations/
      images/
  models/
    alhambra/
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

The model weights are stored with Git LFS. Install Git LFS before cloning to
download them automatically:

```bash
git lfs install
git clone https://github.com/acanort/DL4Lidar.git
cd DL4Lidar
```

## Installation

The provided environment targets Python 3.9, PyTorch 2.8, CUDA 12.8, and a
matching Detectron2 build.

```bash
conda env create -f environment.yml
conda activate d2-rtx5070
pip install -r requirements.txt
```

The PyTorch and Detectron2 packages in `requirements.txt` are CUDA-specific.
For a different CUDA version or a CPU-only system, install compatible PyTorch,
Torchvision, Torchaudio, and Detectron2 builds.

## Data

The datasets use COCO-format annotations and the following directory layout:

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

The Alhambra dataset contains aerosol, cloud, and combined annotations, with
images for the `1064fta/3e8` and `532fta/2e7` configurations. The Mulhacen data
contains aerosol and cloud test annotations together with the corresponding
`1064xta/3e8` and `532xpa/1e7` images.

## Training

`src/train.py` implements a complete holdout training workflow with:

- COCO annotation and image validation
- Detectron2 dataset registration and model configuration
- validation-loss monitoring
- learning-rate reduction on a plateau
- early stopping
- best-checkpoint restoration
- metric logs, training summaries, and loss plots

Configure a training run by editing the user-configuration block near the top
of `src/train.py`. The default settings train a cloud model with the Alhambra
`1064fta/3e8` images and holdout annotations:

```python
INSTRUMENT = "alhambra"
ANNOTATION_SET = "cloud"
CHANNEL = "1064fta"
SCALE = "3e8"

ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
MAX_EPOCHS = 300
IMS_PER_BATCH = 8
```

Start training from the repository root:

```bash
python src/train.py
```

Each run is saved under `outputs/train/` in a timestamped directory containing
the model checkpoint, Detectron2 configuration, metric logs, training summary,
and training plot:

```text
checkpoints/model_best.pth
config/config.yaml
logs/metrics.json
logs/best_model.json
logs/training_summary.json
plots/training_process.png
```

## Inference

`src/inference.py` can run the aerosol model, the cloud model, or both models.
The default command runs cloud inference on the Alhambra `1064fta/3e8` images:

```bash
python src/inference.py
```

To run both models:

```bash
python src/inference.py \
  --classes aerosol,cloud \
  --aerosol-input data/alhambra/images/1064fta/3e8 \
  --cloud-input data/alhambra/images/1064fta/3e8 \
  --output inference/example_run \
  --score-threshold 0.7 \
  --batch-size 1
```

Additional examples:

```bash
# Show all options
python src/inference.py --help

# Run only the aerosol model
python src/inference.py --classes aerosol

# Save predictions without visual outputs
python src/inference.py --output-mode predictions

# Process ten images and save a reduced set of visual outputs
python src/inference.py --output-mode quick --max-images 10

# Force CPU inference
python src/inference.py --device cpu
```

Inference creates:

- `predictions.json` with structured predictions
- `predictions.csv` with instance-level results
- `run_summary.json` with model and run settings
- PNG segmentation masks
- translucent and opaque JPEG overlays
- comparison plots

The visual files produced depend on the selected `--output-mode`. Use
`predictions` for tabular results only, `quick` for a reduced set of visuals,
or `full` for all outputs.
