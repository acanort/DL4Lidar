# Detectron2 Aerosol and Cloud Segmentation

This repository contains Detectron2 scripts for training and running instance
segmentation models on atmospheric aerosol/cloud image products. The code is
organized around a small set of master scripts in `src/`; project-specific or
experiment-specific scripts can be added later by copying and adapting these
entry points.

## Master Scripts

The public entry points are:

- `src/train.py`: base training script. It is the canonical script to copy and
  modify for a new training scenario.
- `src/train_holdout.py`: adapted training script for a `train.json`,
  `valid.json`, and `test.json` holdout split.
- `src/train_5_fold_cross_validation.py`: adapted training script for
  one-model-per-fold cross-validation.
- `src/inference.py`: run one or two trained models and export prediction files,
  masks, overlays, and plots.

The base `src/train.py` script is configured by editing the user configuration
block at the top of the file. The scenario scripts also expose command-line
options; run them with `--help` to see their configurable settings.

## Repository Layout

```text
detectron2_project/
  data/          # local datasets; not intended for GitHub by default
  inference/     # local inference outputs
  models/        # local trained/exported model weights
  outputs/       # local training outputs
  src/
    train.py
    inference.py
    train_holdout.py
    train_5_fold_cross_validation.py
  environment.yml
  requirements.txt
```

Large datasets, model weights, and generated outputs should normally stay out of
Git. Keep only lightweight documentation, configs, or small examples in the
repository.

## Environment

This project expects PyTorch, Detectron2, and the scientific Python stack listed
in `requirements.txt`. A typical setup is:

```bash
conda env create -f environment.yml
conda activate d2-rtx5070
pip install -r requirements.txt
```

Detectron2 builds are tightly coupled to the installed PyTorch and CUDA
versions. If installation fails, install the Detectron2 build that matches your
local PyTorch/CUDA environment.

## Dataset Layout

Training scripts expect COCO-style annotation JSON files and image folders in
this structure:

```text
data/{instrument}/
  annotations/
    {annotation_set}/
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
  images/
    {channel}/
      {scale}/
        image_001.jpg
        image_002.jpg
```

For the current Alhambra data, common values are:

- `instrument`: `alhambra`
- `annotation_set`: `aerosol`, `cloud`, or `combined`
- `channel`: `355fpa`, `532fta`, or `1064fta`
- `scale`: for example `1e6`, `5e6`, `1e7`, `3e8`, `5e8`, or `1e9`

## Download Data From Zenodo

Use `src/download_zenodo_data.py` to download the dataset archive from Zenodo DOI
`10.5281/zenodo.20156234` and extract it into `data_zenodo/`. The script keeps
the same internal layout as `data/`, but does not write into `data/`.

```bash
python src/download_zenodo_data.py
```

To test with a local archive, such as the file placed in `data_tmp/`:

```bash
python src/download_zenodo_data.py --archive data_tmp/DL4Lidar.rar
```

Useful safe checks:

```bash
python src/download_zenodo_data.py --archive data_tmp/DL4Lidar.rar --dry-run
python src/download_zenodo_data.py --archive data_tmp/DL4Lidar.rar --overwrite
```

The archive is expected to contain a top-level `data/` directory. During
extraction, that prefix is removed so `data/alhambra/...` becomes
`data_zenodo/alhambra/...`.

## Base Training Script

Use `src/train.py` as the starting point for a new training script. It contains
one complete, ordinary train/validation/test workflow:

- dataset validation
- COCO dataset registration
- Detectron2 config creation
- validation-loss computation
- learning-rate reduction on plateau
- best-checkpoint restoration
- early stopping
- metric logs, run summaries, and training plots

To run the base script as-is, edit the user configuration block near the top of
the file and then run:

```bash
python src/train.py
```

For a different scenario, copy `src/train.py` and modify the dataset path
construction or run orchestration. The holdout and cross-validation scripts in
this repository are examples of that pattern.

## Train With a Holdout Split

```bash
python src/train_holdout.py \
  --instrument alhambra \
  --annotation-set cloud \
  --channel 1064fta \
  --scale 3e8 \
  --architecture mask_rcnn_R_101_FPN_3x \
  --max-epochs 300 \
  --ims-per-batch 8
```

The script writes a timestamped run under `outputs/train_holdout/` by default.
Each run contains:

- `config/config.yaml`: the Detectron2 config used for training
- `checkpoints/model_best.pth`: the best validation-loss checkpoint
- `logs/metrics.json`: Detectron2 scalar logs
- `logs/best_model.json`: metadata for the best checkpoint
- `logs/training_summary.json`: final run summary
- `plots/training_process.png`: training/validation loss plot

Useful options:

```bash
python src/train_holdout.py --help
python src/train_holdout.py --no-amp
python src/train_holdout.py --resume --no-timestamped-output
```

## Train With 5-Fold Cross Validation

```bash
python src/train_5_fold_cross_validation.py \
  --instrument alhambra \
  --annotation-set cloud \
  --channel 1064fta \
  --scale 3e8 \
  --folds 1-5
```

Use `--folds` to train a subset while testing the pipeline:

```bash
python src/train_5_fold_cross_validation.py --folds 1
python src/train_5_fold_cross_validation.py --folds 1,3,5
```

The script writes one output directory per fold plus
`cross_validation_summary.json` at the run root.

## Run Inference

The inference script expects each class model directory to contain:

```text
models/alhambra/aerosol/
  aerosol_config.yaml
  aerosol_weights.pth
models/alhambra/cloud/
  cloud_config.yaml
  cloud_weights.pth
```

Run both aerosol and cloud models:

```bash
python src/inference.py \
  --classes aerosol,cloud \
  --aerosol-input data/chm15k/images/1064nm/percentiles_1.0_80.5 \
  --cloud-input data/chm15k/images/1064nm/percentiles_1.0_95.5 \
  --output inference/example_run \
  --score-threshold 0.7 \
  --batch-size 1
```

Run only one class:

```bash
python src/inference.py \
  --classes cloud \
  --cloud-input data/chm15k/images/1064nm/percentiles_1.0_95.5 \
  --cloud-model-dir models/alhambra/cloud
```

Main outputs are:

- `predictions.json`: full structured predictions
- `predictions.csv`: tabular instance-level predictions
- `masks_png/`: binary and combined masks
- `overlays_translucent_jpg/`: translucent visual overlays
- `overlays_opaque_jpg/`: opaque visual overlays
- `plots/`: quick comparison plots
- `run_summary.json`: inference configuration and model metadata

Use `--output-mode predictions` to skip image generation, or `--output-mode quick`
to save only the most useful visuals.

## Creating Scenario-Specific Scripts

Keep `src/train.py` and `src/inference.py` generic. For a specific experiment,
copy `src/train.py` and adapt the user configuration block or the dataset/run
section. Recommended examples:

- `scripts/train_cloud_1064fta_3e8.ps1`
- `scripts/train_aerosol_532fta_1e7.ps1`
- `scripts/infer_chm15k_dual_model.ps1`

This keeps the GitHub-facing code easy to understand while still allowing
project-specific workflows to live beside it.

## Notes for Publishing

- Do not commit private datasets, generated inference results, or large model
  weights unless you intentionally want to publish them.
- If you publish trained weights, add a release artifact or an external download
  link and document where they should be placed under `models/`.
- Keep `README.md` updated when new scenario-specific scripts are added.
