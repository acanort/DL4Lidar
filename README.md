# DL4Lidar: Aerosol and Cloud Segmentation from Lidar RCS Data

DL4Lidar provides Detectron2-based instance-segmentation workflows for
identifying aerosol and cloud layers in lidar range-corrected signal (RCS)
images. The repository contains the source code needed to download the
associated datasets and pretrained models, train new models, and run inference
with the trained ALHAMBRA models.

The source repository is intentionally lightweight. The datasets and pretrained
model files are distributed separately through Zenodo and are downloaded into
local folders that are ignored by Git.

## Citation

If you use this project in your research, please cite the associated article:

> Adrian Canella-Ortiz et al., "Generalizable Deep Learning for Aerosol and
> Cloud Segmentation in Ground-Based and Spaceborne Lidar Observations,"
> *IEEE Transactions on Geoscience and Remote Sensing*, 2026.
> DOI: [10.1109/TGRS.2026.3707646](https://doi.org/10.1109/TGRS.2026.3707646)

Please also cite the Zenodo records for any downloaded datasets or pretrained
models used in your work:

- Dataset: [10.5281/zenodo.20156234](https://doi.org/10.5281/zenodo.20156234)
- Pretrained models: [10.5281/zenodo.21379786](https://doi.org/10.5281/zenodo.21379786)

## Repository Contents

The GitHub repository tracks only code, configuration, and documentation:

```text
DL4Lidar/
  src/
    download_data.py
    download_weights.py
    inference.py
    train.py
    train_holdout.py
    train_5_fold_cross_validation.py
  environment.yml
  requirements.txt
  pyproject.toml
  LICENSE
  README.md
```

After setup, the downloader scripts create local `data/` and `models/`
directories. These folders are ignored by Git and should not be committed:

```text
DL4Lidar/
  data/
    alhambra/
    mulhacen/
  models/
    alhambra/
      aerosol/
        aerosol_config.yaml
        aerosol_weights.pth
      cloud/
        cloud_config.yaml
        cloud_weights.pth
```

## Installation

The reference environment uses Windows 11, Python 3.9, PyTorch 2.8, CUDA 12.8,
and a precompiled Detectron2 wheel. Before starting, install:

- Git
- Miniconda or Anaconda
- a current NVIDIA driver, if GPU inference or training will be used
- WinRAR/UnRAR or 7-Zip, because the dataset archive is distributed as RAR

Clone the repository and enter the project directory:

```console
git clone https://github.com/acanort/DL4Lidar.git
cd DL4Lidar
```

Create and activate the Conda environment:

```console
conda env create -n d2-dl4lidar -f environment.yml
conda activate d2-dl4lidar
python -m pip install --upgrade pip wheel "setuptools<81"
```

Install PyTorch 2.8 for CUDA 12.8:

```console
python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Install the matching Detectron2 wheel:

```console
python -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder "detectron2==0.6+18f6958pt2.8.0cu128"
```

This Detectron2 package is a third-party Windows wheel from
[`torch_packages_builder`](https://github.com/MiroPsota/torch_packages_builder),
not an official Detectron2 binary. It avoids compiling Detectron2's C++ and
CUDA extensions locally.

The remaining shared dependencies are installed by `environment.yml`. Running
the following command is optional after PyTorch and Detectron2 are installed,
and verifies that the pinned packages in `requirements.txt` are satisfied:

```console
python -m pip install -r requirements.txt
```

Verify the installation:

```console
python -c "import torch, detectron2; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Detectron2:', detectron2.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python src/inference.py --help
```

## Download Data and Models

Download and prepare the COCO-format dataset from Zenodo DOI
[10.5281/zenodo.20156234](https://doi.org/10.5281/zenodo.20156234):

```console
python src/download_data.py
```

The script downloads `DL4Lidar.rar`, validates the extracted contents, and
creates `data/` in the repository root. It refuses to overwrite an existing
`data/` directory unless `--replace` is provided.

If the archive has already been downloaded, prepare it without accessing
Zenodo:

```console
python src/download_data.py --archive path/to/DL4Lidar.rar
```

Download the pretrained ALHAMBRA aerosol and cloud model assets from Zenodo DOI
[10.5281/zenodo.21379786](https://doi.org/10.5281/zenodo.21379786):

```console
python src/download_weights.py
```

The script creates:

```text
models/alhambra/aerosol/aerosol_config.yaml
models/alhambra/aerosol/aerosol_weights.pth
models/alhambra/cloud/cloud_config.yaml
models/alhambra/cloud/cloud_weights.pth
```


## Dataset Layout

After extraction, the dataset uses the following structure:

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

The ALHAMBRA data include aerosol, cloud, and combined annotations, with images
for the `1064fta/3e8` and `532fta/2e7` configurations. The Mulhacen data
include aerosol and cloud test annotations with the corresponding
`1064xta/3e8` and `532xpa/1e7` images.

## Training

`src/train.py` contains the full holdout training workflow, including COCO
validation, Detectron2 dataset registration, validation-loss monitoring,
learning-rate reduction on plateau, early stopping, best-checkpoint
restoration, metric logs, summaries, and loss plots.

The default settings train a cloud model with the ALHAMBRA `1064fta/3e8`
images and holdout annotations:

```python
INSTRUMENT = "alhambra"
ANNOTATION_SET = "cloud"
CHANNEL = "1064fta"
SCALE = "3e8"
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
MAX_EPOCHS = 300
IMS_PER_BATCH = 8
```

Run the default training workflow from the repository root:

```console
python src/train.py
```

The CLI wrappers expose the same training configuration without editing source
constants:

```console
python src/train_holdout.py --help
python src/train_5_fold_cross_validation.py --help
```

Training outputs are written under `outputs/`, which is ignored by Git. A
typical training run contains:

```text
checkpoints/model_best.pth
config/config.yaml
logs/metrics.json
logs/best_model.json
logs/training_summary.json
plots/training_process.png
```

## Inference

`src/inference.py` runs the pretrained aerosol model, the pretrained cloud
model, or both. Before running inference, download both the dataset and model
assets.

Run the default cloud inference workflow:

```console
python src/inference.py
```

Run both aerosol and cloud inference on the ALHAMBRA `1064fta/3e8` images:

```console
python src/inference.py ^
  --classes aerosol,cloud ^
  --aerosol-input data/alhambra/images/1064fta/3e8 ^
  --cloud-input data/alhambra/images/1064fta/3e8 ^
  --output inference/example_run ^
  --score-threshold 0.7 ^
  --batch-size 1
```

Additional examples:

```console
python src/inference.py --help
python src/inference.py --classes aerosol
python src/inference.py --output-mode predictions
python src/inference.py --output-mode quick --max-images 10
python src/inference.py --device cpu
```

Inference outputs are written under `inference/`, which is ignored by Git. The
available output modes are `predictions`, `quick`, and `full`.

## Other CUDA Versions and CPU-Only Systems

The required PyTorch and Detectron2 package variants depend on the operating
system, GPU, and NVIDIA driver. Use `nvidia-smi` to check the installed driver,
then consult the [official PyTorch installer](https://pytorch.org/get-started/locally/)
before choosing a CUDA build.

Precompiled Windows/Python 3.9 Detectron2 wheels matching PyTorch 2.8 are
available for `cu126`, `cu128`, `cu129`, and `cpu`. PyTorch and Detectron2 must
use the same variant. For CUDA 12.6, for example:

```console
python -m pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder "detectron2==0.6+18f6958pt2.8.0cu126"
```

For a Windows system without a compatible NVIDIA GPU, use CPU builds:

```console
python -m pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder "detectron2==0.6+18f6958pt2.8.0cpu"
```

CPU inference is supported but substantially slower. Do not run
`requirements.txt` unchanged after selecting `cu126`, `cu129`, or `cpu`,
because it pins the reference CUDA 12.8 packages.

## Licensing

The source code in this repository is licensed under the GNU General Public
License, version 3 or later. See [LICENSE](LICENSE) for the complete license
text.

The dataset and pretrained model assets are distributed separately through
Zenodo. Use the citation and license information provided in the corresponding
Zenodo records when using those assets.
