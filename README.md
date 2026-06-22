# DL4Lidar: Aerosol and Cloud Segmentation

DL4Lidar uses Detectron2 instance-segmentation models to identify atmospheric
aerosol and cloud layers in lidar RCS images. The repository provides training
and inference code, COCO-format annotations, example image datasets, and
trained models using ALHAMBRA Raman lidar data (UGR Station, Granada, Spain).

## Repository structure

```text
DL4Lidar/
  data/
    alhambra/
      annotations/
      images/

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

## Installation

The reference setup uses Windows 11, Python 3.9, PyTorch 2.8, CUDA 12.8, and a
precompiled Detectron2 wheel. Before starting, install:

- Git and Git LFS
- Miniconda or Anaconda
- a current NVIDIA driver

Check that the GPU and driver are visible:

```console
nvidia-smi
```

Run the installation commands below in an Anaconda Prompt from the repository
root.

### Reference installation: NVIDIA GPU and CUDA 12.8

Create a fresh environment containing the shared project dependencies:

```console
conda env create -n d2-dl4lidar -f environment.yml
conda activate d2-dl4lidar
python -m pip install --upgrade pip wheel "setuptools<81"
```

Install the official PyTorch 2.8 wheels for CUDA 12.8:

```console
python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Install the matching precompiled Detectron2 wheel:

```console
python -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder "detectron2==0.6+18f6958pt2.8.0cu128"
```

This Detectron2 package is a third-party Windows wheel from
[`torch_packages_builder`](https://github.com/MiroPsota/torch_packages_builder),
not an official Detectron2 binary. It avoids compiling Detectron2's C++ and
CUDA extensions locally.

The Conda environment already installs the remaining packages from
`environment.yml`. Running the following is optional after PyTorch and
Detectron2 are installed, and verifies that all pins in `requirements.txt` are
satisfied:

```console
python -m pip install -r requirements.txt
```

### Verify the installation

```console
python -c "import torch, detectron2; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Detectron2:', detectron2.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python src/inference.py --help
```

Run a one-image inference test:

```console
python src/inference.py --classes cloud --max-images 1 --output-mode predictions --output inference/smoke_test
```

### Other graphics cards and CPU-only systems

The required package variant depends on the operating system, GPU, and NVIDIA
driver—not only on the graphics-card model. Use `nvidia-smi` to check the
driver, then consult the
[official PyTorch installer](https://pytorch.org/get-started/locally/) before
choosing a CUDA build.

Precompiled Windows/Python 3.9 Detectron2 wheels matching PyTorch 2.8 are
available for `cu126`, `cu128`, `cu129`, and `cpu`. PyTorch and Detectron2 must
use the same variant. For example, a CUDA 12.6 installation uses:

```console
python -m pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder "detectron2==0.6+18f6958pt2.8.0cu126"
```

For a Windows system without a compatible NVIDIA GPU, use the CPU builds:

```console
python -m pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder "detectron2==0.6+18f6958pt2.8.0cpu"
```

CPU inference is supported but is substantially slower. On Windows, AMD and
Intel graphics users should use the CPU instructions unless they have a
separately validated PyTorch and Detectron2 stack.

`requirements.txt` pins the reference `cu128` packages. Do not run it unchanged
after selecting `cu126`, `cu129`, or `cpu`, because it would attempt to replace
that selection with the CUDA 12.8 builds. The shared dependencies are already
installed by `environment.yml`.
