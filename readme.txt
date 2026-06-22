# Data folder — structure and usage

This document briefly describes the structure of the `data` folder.

Quick summary

- The `data/` folder is organized by instruments (for example, `alhambra` or `mulhacen`).
- Each instruments contains `annotations/` and `images/` subfolders.

Top-level structure

- data/
  - alhambra/
    - annotations/
      - aerosol/
        - 5_fold_cross_validation/
        - holdout/
      - cloud/
        - 5_fold_cross_validation/
        - holdout/
      - combined/
        - 5_fold_cross_validation/
        - holdout/
    - images/
      - 1064fta/
        - 1e6/ 1e7/ 1e8/ 1e9/ 2e7/ 3e7/ 3e8/ 5e6/ 5e7/ 5e8/
      - 355fpa/
        - 1e6/ 1e7/ 1e8/ 1e9/ 2e7/ 3e7/ 3e8/ 5e6/ 5e7/ 5e8/
      - 532fta/
        - 1e6/ 1e7/ 1e8/ 1e9/ 2e7/ 3e7/ 3e8/ 5e6/ 5e7/ 5e8/
  - mulhacen/
    - annotations/
      - aerosol/
        - holdout/
      - cloud/
        - holdout/
    - images/
      - 1064fta/
        - 3e8
      - 532fta/
        - 1e7        

Annotations

- Location: `data/alhambra/annotations/`.
- Subsets: `aerosol`, `cloud`, `combined`.
- Each subset typically includes two grouping strategies:
  - `5_fold_cross_validation/`: per-fold files. Typical names: `train_fold_1.json` ... `train_fold_5.json`, `valid_fold_1.json` ... `valid_fold_5.json`, and `test.json`.
  - `holdout/`: single holdout split with `train.json`, `valid.json`, and `test.json`.
- Format: JSON (COCO-like schema).

Images

- Location: `data/alhambra/images/`.
- Organization: folders for different wavelengths (`1064fta`, `355fpa`, `532fta`) with subfolders for scales (`1e6`, `1e7`, ...).
- Extensions: `.jpg`

Notes

- The MULHACEN data only contains the data used to evaluate it in the paper, this is, with the required wavelengths and scales. 
