# mlp_dtof
# Latent-Space Learning of Optical Parameters from Simulated DTOF Signals

## 1. Introduction

This repository implements a structured machine learning pipeline for
the analysis of simulated Distribution of Time-of-Flight (DTOF) signals.
The primary objective is to learn a compact latent representation of
time-resolved photon transport data and subsequently use this
representation, together with geometrical parameters, to predict optical
properties.

The workflow consists of four stages:

1.  Conversion of MATLAB simulation outputs (`.mat`) into compressed
    NumPy archives (`.npz`).
2.  Construction of a PyTorch dataset with deterministic signal
    preprocessing.
3.  Training of a fully-connected autoencoder to obtain a 4-dimensional
    latent embedding.
4.  Supervised classification of discretised optical parameters using
    multilayer perceptrons (MLPs).

At present, the raw simulation data cannot be publicly released. This
document specifies the exact data format required to reproduce the
experiments. The dataset will be made available in a future release.

------------------------------------------------------------------------

## 2. Pipeline Description

### 2.1 MATLAB to NumPy Conversion

The first script:

-   Reads a whitespace-separated metadata file (`legenda.txt`).
-   Loads corresponding MATLAB `.mat` files.
-   Extracts the variable `output_all`.
-   Flattens it into a one-dimensional array.
-   Stores one `.npz` file per sample containing: `output`: the flattened signal vector, `params`: the first six numeric parameters from the metadata row.

This conversion ensures fast loading during training and avoids repeated
MATLAB parsing.

------------------------------------------------------------------------

### 2.2 Dataset Construction and Signal Preprocessing

A custom PyTorch `Dataset` is implemented to:

-   Load `.npz` files.
-   Apply deterministic preprocessing to each signal:

Each sample returned by the dataset consists of:

-   `arr`: preprocessed signal (torch.float32 tensor).
-   `optical_props`: numeric parameter vector read directly from the
    metadata table.

------------------------------------------------------------------------

### 2.3 Autoencoder Architecture

A fully-connected autoencoder is trained to reconstruct the preprocessed
signal.

Architecture:

Input → Linear(32) → ReLU → Linear(4) → Sigmoid\
Linear(32) → ReLU → Linear(input_dim)

Properties:

-   Latent dimension: 4.
-   Loss function: Mean Squared Error (MSE).
-   Optimiser: Adam (learning rate = 5×10⁻³).
-   Training epochs: 10.

The encoder provides a low-dimensional embedding intended to capture the
essential structure of the DTOF signal.

The trained model is saved for subsequent use in supervised learning.

------------------------------------------------------------------------

### 2.4 Latent-Space Classification of Optical Parameters

After training the autoencoder:

1.  Each signal is encoded into a latent vector `z ∈ R⁴`.
2.  Geometrical parameters `T0` and `R` are extracted from metadata.
3.  A 6-dimensional feature vector is constructed:
    `[T0, R, z1, z2, z3, z4]`
Continuous optical parameters are discretised into bins (default:
quantile binning) and treated as classification tasks.

Two separate MLP classifiers are trained:
-   Classifier A predicts `(mua1, mua2)`.
-   Classifier B predicts `(mus1, mus2)`.

Each classifier consists of:
-   Shared trunk: 6 → 64 → 32 (ReLU activations).
-   Two independent linear heads for the two parameters.

Evaluation includes:
-   Per-parameter accuracy.
-   Mean accuracy.
-   Accuracy heatmaps as a function of `(T0, R)` (see files in folder).
-   Row-normalised confusion matricx (see file in folder).

------------------------------------------------------------------------

## 3. Required Data Format

### 3.1 Metadata File (`legenda.txt`)

The file must be whitespace-separated and contain a header line.
Each data row must contain at minimum:
    filename  T0  R  mua1  mua2  mus1  mus2
Requirements:
-   First column: filename (with or without `.mat`).
-   At least six numeric columns after filename.
-   Empty lines and lines starting with `#` are ignored.

Column interpretation in classifier script:

  Numeric Column Index   Parameter
  ---------------------- -----------
  0                      T0
  1                      R
  2                      mua1
  3                      mua2
  4                      mus1
  5                      mus2

Additional columns are allowed but not used by default.

------------------------------------------------------------------------

### 3.2 MATLAB Files

Each `.mat` file must contain a variable named:
    output_all
The variable may have arbitrary shape; it will be flattened to a
one-dimensional vector.

If a different variable name is used, modify:
    VAR_NAME = "output_all"
in the conversion script.

------------------------------------------------------------------------

## 4. Installation

Recommended Python version: 3.9 or newer.
Required packages:
-   numpy
-   scipy
-   pandas
-   matplotlib
-   torch

Install via:
    pip install numpy scipy pandas matplotlib torch

------------------------------------------------------------------------

## 5. Execution Procedure

1.  Convert MATLAB files:
    python FromMatToDict.py

2.  Verify dataset loading:
    python DatasetNpz.py

3.  Train autoencoder:
    python AutoencoderNpz.py

4.  Train classifiers and generate evaluation plots:
    python Classifier_MLPDoubleHead.py

------------------------------------------------------------------------

## 6. Methodological Notes

-   Signal preprocessing is fixed and hard-coded.
-   Latent dimension is set to 4.
-   Continuous targets are discretised for classification.
-   Train/test split uses 80/20 partition with fixed random seed.
-   Accuracy heatmaps assume discrete sets of `T0` and `R` values;
    update if needed.

------------------------------------------------------------------------

## 7. Data Availability

The simulation data and metadata table are not included in this
repository due to current restrictions. They will be released in a
forthcoming update.

Researchers wishing to reproduce the pipeline with independent data must
ensure strict adherence to the format described above.
