# PET Enhancement: Deep Learning for Dose Reduction

Deep learning models (Pix2Pix and CycleGAN) for PET image enhancement. The project preprocesses raw DICOM PET scans and trains models to enhance low-dose images.

## Structure

```
.
├── data/
│   ├── preprocessing_pipeline.py      # DICOM to HDF5 preprocessing
│   └── run_preprocessing.py            # Preprocessing runner
├── pytorch-CycleGAN-and-pix2pix/      # PyTorch GANs implementation (cloned repo)
├── train_pix2pix_cyclegan_multikey.py # Multi-key training orchestrator
└── requirements.txt
```

## Quick Start

### 1. Preprocess DICOM to HDF5

```bash
cd data
python run_preprocessing.py
```

Converts raw DICOM files to HDF5 format with multiple dose reduction factors.

### 2. Train Models

```bash
python train_pix2pix_cyclegan_multikey.py \
    --dataroot train_sample \
    --name my_experiment \
    --model pix2pix \
    --input-keys 1_10 1_50 1_100
```

Trains separate models for each dose reduction factor.

## Files

- **`data/preprocessing_pipeline.py`**: `PETPreprocessor` class for DICOM preprocessing
  - Loads DICOM series by dose factor
  - Per-slice min-max normalization to [-1, 1]
  - Spatial standardization (440×440)
  - Saves to HDF5 with gzip compression

- **`data/run_preprocessing.py`**: Script to run preprocessing on all patients

- **`train_pix2pix_cyclegan_multikey.py`**: Training orchestrator
  - Runs training for multiple dose reduction factors
  - Each factor gets its own model
  - Supported factors: 1_2, 1_5, 1_10, 1_20, 1_50, 1_100
  - Target: full-dose images

## Dataset

### Input (Raw DICOM)
```
uExplorerPART16/Patient/
├── 2.886 x 600 WB NORMAL/     # Full-dose
├── 2.886 x 600 WB D10/        # 10× reduction
└── ...
```

### Output (Preprocessed HDF5)
```
train_sample/Patient.h5
├── full: (N, 440, 440)
├── 1_10: (N, 440, 440)
└── ...
```

## Dependencies

```bash
pip install -r requirements.txt
```


